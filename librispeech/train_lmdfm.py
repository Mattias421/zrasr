#!/usr/bin/env/python3
"""Recipe for training a wav2vec-based ctc ASR system with librispeech.
The system employs wav2vec as its encoder. Decoding is performed with
k2 through the use of a decoding graph and, optionally, a rescoring LM.
To run this recipe, do the following:
> python train_with_wav2vec.py hparams/train_{hf,sb}_wav2vec.yaml
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens.

Authors
 * Pierre Champion 2023
 * Zeyu Zhao 2023
 * Georgios Karakasidis 2023
 * Rudolf A Braun 2022
 * Titouan Parcollet 2022
 * Sung-Lin Yeh 2021
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.distributions import Independent, Normal
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
import speechbrain.integrations.k2_fsa as sbk2
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

import k2
from flow_matching.utils import ModelWrapper
from flow_matching.solver import Solver, ODESolver
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler

logger = get_logger(__name__)

# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        if self.hparams.audio_lm:
            codec_tokens, codec_token_lens = batch.codec_tokens
            codec_tokens, codec_token_lens = codec_tokens.to(self.device), codec_token_lens.to(self.device)
            x_1 = codec_tokens
            x_1_lens = codec_token_lens
            if x_1.shape[-1] > 1024:
                raise ValueError(f"length of {x_1.shape[-1]} is larger than 1024")
        else:
            x_1, x_1_lens = batch.tokens_bos_eos

        x_0 = torch.randint_like(x_1, high=self.hparams.output_neurons)

        t = torch.rand(x_1.shape[0], device=x_1.device) * (1.0 - self.time_epsilon)
        path_sample = self.hparams.path.sample(t=t, x_0=x_0, x_1=x_1)

        logits = self.modules.dfm_model(x_t=path_sample.x_t, time=path_sample.t)

        if isinstance(self.hparams.loss_fn, torch.nn.CrossEntropyLoss):
            loss = self.hparams.loss_fn(logits.flatten(0, 1), x_1.flatten(0, 1)).mean()
        else:
            # assume MixturePathGeneralizedKL
            loss = self.hparams.loss_fn(
                logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t
            ).mean()

        return loss, x_0, logits, x_1, x_1_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        loss, x_0, logits, x_1, x_1_lens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens

        if stage == sb.Stage.TRAIN:
            train_stats = {"loss_step":loss}
            self.hparams.train_logger.log_stats(stats_meta={"optimizer_step":self.optimizer_step}, train_stats=train_stats)

        loss = loss

        logit_pred = logits.argmax(dim=-1)
        logit_pred[:, -1] = 2 # ensure there is at least 1 eos token per utt
        sample_list = [s.tolist() for s in logit_pred]
        sample_list_filtered = [s[:s.index(2)] for s in sample_list]
        sample_list_filtered = [[i for i in s if i != 0] for s in sample_list_filtered]

        # Decode token terms to words
        predicted_words = [
            tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in sample_list_filtered
        ]

        if stage != sb.Stage.TRAIN:
            # compute ELBO
            linear_scheduler = PolynomialConvexScheduler(n=1.0)
            linear_path = MixtureDiscreteProbPath(scheduler=linear_scheduler)

            generalized_kl_fn = MixturePathGeneralizedKL(path=linear_path, reduction="none")

            class WrappedModel(ModelWrapper):
                def forward(self, x, t, **extras):
                    # Note: logit's precision is important.
                    return torch.softmax(self.model(x_t=x, time=t).float(), -1)

            wrapped_probability_denoiser = WrappedModel(model=self.modules.dfm_model)

            # Time discretization
            sampling_steps = self.hparams.sampling_steps_test if stage == sb.Stage.TEST else self.hparams.sampling_steps_valid
            discretization = (
                torch.linspace(0, 1, sampling_steps + 1, device=self.device)[:-1]
                .view(-1, 1)
                .repeat(1, self.hparams.batch_size_eval)
            )
            # Lower variance estimator for time discretization
            discretization = discretization + torch.rand(
                size=(1, self.hparams.batch_size_eval), device=self.device
            )
            discretization = discretization % 1
            discretization = discretization * (1 - self.time_epsilon)

            for k in discretization[:, : x_1.shape[0]]:
                x_0 = torch.randint_like(x_1, high=self.hparams.output_neurons)
                x_t = linear_path.sample(t=k, x_0=x_0, x_1=x_1).x_t

                t = self.hparams.path.scheduler.kappa_inverse(k)

                logits = wrapped_probability_denoiser(x=x_t, t=t)

                generalized_kl = generalized_kl_fn(logits=logits, x_1=x_1, x_t=x_t, t=k)
                self.n_elements += generalized_kl.numel()

                self.elbo += generalized_kl.sum()

            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):

                solver = MixtureDiscreteEulerSolver(
                    model=wrapped_probability_denoiser,
                    path=self.hparams.path,
                    vocabulary_size=self.hparams.output_neurons
                )


                sample = solver.sample(
                    x_init=x_0,
                    step_size=1 / sampling_steps,
                    verbose=True,
                    dtype_categorical=torch.float64,
                    time_grid=torch.tensor([0.0, 1.0 - self.time_epsilon]),
                )


                sample[:,-1] = 2 # ensure all sequences have at least one eos token
                sample_list = [s.tolist() for s in sample]
                sample_list_filtered = [s[:s.index(2)] for s in sample_list]
                sample_list_filtered = [[i for i in s if i != 0] for s in sample_list_filtered]

                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in sample_list_filtered
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                self.wer_metric.append(ids, predicted_words, target_words)
                print(predicted_words)


            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(logits, x_1, x_1_lens)
            self.acc_metric_pad.append(logits, x_1, x_1_lens)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.acc_metric_pad = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()

            self.elbo = torch.zeros((1,), device=self.device)
            self.n_elements = torch.zeros((1,), device=self.device)

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            stage_stats["ELBO"] = torch.exp(self.elbo / self.n_elements).item()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "optimizer_step": self.optimizer_step,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                meta=stage_stats,
                min_keys=["loss"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"loss": 1.1, "epoch": epoch},
                max_keys=["loss"],
                num_to_keep=1,
            )

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    valtest_datasets = [valid_data] + [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("codec_tokens")
    def audio_pipeline(ID):
        tokens = torch.load(hparams['codec_cache'] + f"/{ID}.pt")
        tokens[0] = hparams["bos_index"]
        tokens[-1] = hparams["eos_index"] # TODO check that performance on final tokens is correct
        return tokens

    sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos_eos = torch.LongTensor([hparams["bos_index"]] + (tokens_list) + [hparams["eos_index"]])
        yield tokens_bos_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "codec_tokens", "wrd", "tokens_bos_eos", "tokens"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_train,
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_valid,
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.time_epsilon = 1e-3 if isinstance(asr_brain.hparams.loss_fn, MixturePathGeneralizedKL) else 0.0

    # adding objects to trainer:
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        collate_fn = None
        if "collate_fn" in train_dataloader_opts:
            collate_fn = train_dataloader_opts["collate_fn"]

        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

        if collate_fn is not None:
            train_dataloader_opts["collate_fn"] = collate_fn

    if valid_bsampler is not None:
        collate_fn = None
        if "collate_fn" in valid_dataloader_opts:
            collate_fn = valid_dataloader_opts["collate_fn"]

        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

        if collate_fn is not None:
            valid_dataloader_opts["collate_fn"] = collate_fn

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing
    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}_{hparams['sampling_steps_test']}_NFE.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            min_key="loss",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
