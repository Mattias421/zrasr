import torch
from torch import nn, Tensor

class CharPrior(nn.Module):
    def __init__(
        self,
        n_chars
    ):
        super().__init__()
        
        self.n_chars = n_chars
        self.register_buffer("counts", torch.zeros(n_chars)) # add counts during training

    def forward(self, chars: Tensor, return_log = True) -> Tensor:
        probs = self.counts[chars] / self.counts.sum()
        return probs.log()

    def update_counts(self, chars):
        self.counts += chars.bincount(minlength=self.n_chars)
