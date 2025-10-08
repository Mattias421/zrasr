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
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from speechbrain.dataio.batch import PaddedBatch
from torch.distributions import Independent, Normal
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
import speechbrain.integrations.k2_fsa as sbk2
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

from flow_matching.utils import ModelWrapper
from flow_matching.solver import Solver, ODESolver
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler

import kenlm

logger = get_logger(__name__)

# Define training procedure
class ASR(sb.core.Brain):

    def generate_ngram_training_data(self, data):
        kenlm_dir = Path(self.hparams.kenlm_dir)
        kenlm_dir.mkdir(parents=True, exist_ok=True)

        data_loader = DataLoader(train_data, collate_fn=PaddedBatch, batch_size=self.hparams.batch_size_eval, shuffle=False)

        self.checkpointer.recover_if_possible(min_key="loss")

        class WrappedModel(ModelWrapper):
            def forward(self, x, t, **extras):
                # Note: logit's precision is important.
                return torch.softmax(self.model(x_t=x, time=t).float(), -1)

        wrapped_probability_denoiser = WrappedModel(model=self.modules.dfm_model)

        solver = MixtureDiscreteEulerSolver(
            model=wrapped_probability_denoiser,
            path=self.hparams.path,
            vocabulary_size=self.hparams.output_neurons
        )

        n_hyp_tokens = 0
        n_ref_tokens = 0
        
        for batch in data_loader:
            x_1, x_1_lens = batch.tokens
            x_1 = x_1.to(self.device)
            x_0 = torch.randint_like(x_1, high=self.hparams.output_neurons)

            sample = solver.sample(
                x_init=x_0,
                step_size=1 / self.hparams.sampling_steps_test,
                verbose=True,
                dtype_categorical=torch.float64,
                time_grid=torch.tensor([0.0, 1.0 - self.time_epsilon]),
            )

            sample[:,-1] = 2 # ensure all sequences have at least one eos token
            sample_list = [s.tolist() for s in sample]
            sample_list_filtered = [s[:(s.index(2) + 1)] for s in sample_list]
            sample_list_filtered = [[i for i in s if i != 0] for s in sample_list_filtered]

            label_list = [s.tolist() for s in x_1]
            label_list_filtered = [s[:(s.index(2) + 1)] for s in label_list]
            label_list_filtered = [[str(i) for i in s if i != 0] for s in label_list_filtered]

            predicted_words = [
                tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in sample_list_filtered
            ]

            sample_list_filtered = [[str(i) for i in j] for j in sample_list_filtered]

            hyp_file = open(kenlm_dir / "hyp.txt", 'a')
            ref_file = open(kenlm_dir / "ref.txt", 'a')
            wrd_file = open(kenlm_dir / "wrd.txt", 'a')

            for h, r, w in zip(sample_list_filtered, label_list_filtered, predicted_words):
                print(w)
                if w[0] != '' and n_hyp_tokens < self.hparams.target_hyp_tokens:
                    print(' '.join(w), file=wrd_file)
                    print(' '.join(h), file=hyp_file)

                    n_hyp_tokens += len(h)

                if n_ref_tokens < self.hparams.target_ref_tokens:
                    print(' '.join(r), file=ref_file)
                    n_ref_tokens += len(r)


            hyp_file.close()
            wrd_file.close()
            ref_file.close()

            print(f"ref tokens {n_ref_tokens/self.hparams.target_ref_tokens}, hyp tokens {n_hyp_tokens/self.hparams.target_hyp_tokens}")

            if n_ref_tokens >= self.hparams.target_ref_tokens and n_hyp_tokens >= self.hparams.target_hyp_tokens:
                return

    def compute_perplexity(self, data):
        hyp_model = kenlm.Model(f"{self.hparams.kenlm_dir}/hyp.arpa")
        ref_model = kenlm.Model(f"{self.hparams.kenlm_dir}/ref.arpa")

        num_tok = 0
        full_score_hyp = 0
        full_score_ref = 0

        for item in data:
            tok = item['tokens'].tolist()
            num_tok += len(tok) + 1
            tok = ' '.join([str(t) for t in tok])

            full_score_hyp += hyp_model.score(tok)
            full_score_ref += ref_model.score(tok)

        print(f"full_score_hyp: {full_score_hyp}")
        print(f"full_score_ref: {full_score_ref}")
        print(f"num_tok: {num_tok}")
        print(f"perplexity_hyp: {10.0 ** (-full_score_hyp / num_tok)}")
        print(f"perplexity_ref: {10.0 ** (-full_score_ref / num_tok)}")


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    # sort decending
    train_data = train_data.filtered_sorted(
        sort_key="duration", reverse=False
    )
    # when sorting do not shuffle in dataloader ! otherwise is pointless
    hparams["train_dataloader_opts"]["shuffle"] = False

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
    @sb.utils.data_pipeline.provides("tokens")
    def audio_pipeline(ID):
        tokens = torch.load(hparams['codec_cache'] + f"/{ID}.pt")
        tokens[0] = hparams["bos_index"]
        tokens[-1] = hparams["eos_index"] # TODO check that performance on final tokens is correct
        return tokens

    if hparams["audio_lm"]:
        sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

        sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "tokens"
    )
    def text_pipeline(wrd):
        tokens_list = tokenizer.encode_as_ids(wrd)
        tokens_bos_eos = torch.LongTensor([hparams["bos_index"]] + (tokens_list) + [hparams["eos_index"]])
        yield tokens_bos_eos

    if not hparams["audio_lm"]:
        sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "tokens"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None

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

    # Training generation
    asr_brain.generate_ngram_training_data(train_data)

    # Training
    kenlm_save_path = hparams["kenlm_dir"]
    with open(f"./{kenlm_save_path}/hyp.txt", 'r') as hyp_txt, open(f"./{kenlm_save_path}/hyp.arpa", 'w') as hyp_arpa: 
        subprocess.run([hparams["kenlm_lmplz"], "-o3"], stdin=hyp_txt, stdout=hyp_arpa)
    with open(f"./{kenlm_save_path}/ref.txt", 'r') as ref_txt, open(f"./{kenlm_save_path}/ref.arpa", 'w') as ref_arpa: 
        subprocess.run([hparams["kenlm_lmplz"], "-o3"], stdin=ref_txt, stdout=ref_arpa)

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        print(f"testing {k}")
        asr_brain.compute_perplexity(test_datasets[k])

