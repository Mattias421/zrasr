import torch
from torch import nn, Tensor

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return torch.sigmoid(x) * x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        time_dim: int = 1,
        class_dim: int = 1,
        hidden_dim: int = 128,
        num_class: int = 30,
        quantise_time: int = 100,
        use_embeddings: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.class_dim = class_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.quantise_time = quantise_time
        self.use_embeddings = use_embeddings

        if use_embeddings:
            self.time_embed = nn.Embedding(quantise_time, hidden_dim)
            self.class_embed = nn.Embedding(num_class, hidden_dim)
            self.x_layer_0 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                Swish()
            )
            main_input_dim = hidden_dim
        else:
            main_input_dim = input_dim + time_dim + class_dim

        self.main = nn.Sequential(
            nn.Linear(main_input_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor, c) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        
        if self.use_embeddings:
            t = (t * self.quantise_time).to(torch.int64)
            h_t = self.time_embed(t)
            h_c = self.class_embed(c)
            h_x = self.x_layer_0(x)
            h = h_t + h_c + h_x
        else:
            t = t.reshape(-1, self.time_dim).float()
            c = c.reshape(-1, self.class_dim).float()
            t = t.reshape(-1, 1).expand(x.shape[0], 1)
            c = c.reshape(-1, 1).expand(x.shape[0], 1)
            h = torch.cat([x, t, c], dim=1)
        
        output = self.main(h)
        return output.reshape(*sz)
