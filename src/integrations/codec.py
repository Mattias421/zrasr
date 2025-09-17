import dualcodec
import torchaudio
import torch

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

