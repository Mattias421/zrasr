import dualcodec
import torchaudio
import torch

from ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download

class DualCodec():
    def __init__(self, model_id="12hz_v1", n_quantizers=8, sample_rate=24000):
        self.dualcodec_model = dualcodec.get_model(model_id)
        self.dualcodec_inference = dualcodec.Inference(dualcodec_model=self.dualcodec_model, device='cuda')
        self.n_quantizers = n_quantizers
        self.sample_rate = sample_rate

    def to(self, device):
        self.dualcodec_inference = self.dualcodec_inference.to(device)

    @torch.no_grad
    def __call__(self, audio):

        audio = torchaudio.functional.resample(audio, self.sample_rate, 24000)
        audio = audio[:,None,:]
        audio = audio.to(self.dualcodec_inference.device)

        semantic_codes, acoustic_codes = self.dualcodec_inference.encode(audio, n_quantizers=self.n_quantizers)

        return semantic_codes[:,0,:]

class FACodec():
    def __init__(self, codebook_layer):
        self.codebook_layer = codebook_layer

        fa_encoder = FACodecEncoder(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )

        fa_decoder = FACodecDecoder(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )

        encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
        decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

        fa_encoder.load_state_dict(torch.load(encoder_ckpt))
        fa_decoder.load_state_dict(torch.load(decoder_ckpt))

        fa_encoder.eval()
        fa_decoder.eval()

        self.fa_encoder = fa_encoder.cuda()
        self.fa_decoder = fa_decoder.cuda()

    @torch.no_grad
    def __call__(self, audio):
        audio = audio[:, None, :]
        enc_out = self.fa_encoder(audio)

        vq_post_emb, vq_id, _, q, spk = self.fa_decoder(enc_out, eval_vq=False, vq=True)

        return vq_id[self.codebook_layer] # layers 1 and 2 are content, 0 is prosody, rest are acoustic detail

