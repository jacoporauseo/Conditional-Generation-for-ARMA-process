import torch 
import torch.nn as nn
import math 


class TimeEmbedding(nn.Module):
    """ Time Embedding module from KAIST course in Diffusion Models"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(k, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param k: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=k.device)
        args = k[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding
    
    def forward(self, k : torch.Tensor):
            if k.ndim == 0:
                k = k.unsqueeze(-1)
            t_freq = self.timestep_embedding(k, self.frequency_embedding_size)
            t_emb = self.mlp(t_freq)
            return t_emb


class TimeLinear(nn.Module):

    def __init__(self, dim_in: int, dim_out: int, diffusion_steps : int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.diffusion_steps = diffusion_steps

        self.time_embedding = TimeEmbedding(dim_out)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, k: torch.Tensor):
        x = self.fc(x)
        alpha = self.time_embedding(k).view(-1, self.dim_out)

        return alpha * x


class DenoisingNet(nn.Module):
    """Conditional Denoising Network e(x^k,y,k)"""
    def __init__(self, dim_in, dim_out, dim_hids, diffusion_steps):
        super().__init__()
        
        dims = [dim_in + 1] + dim_hids + [dim_out]  # +1 for y
        self.tlins = nn.ModuleList([
            TimeLinear(dims[i], dims[i+1], diffusion_steps)
            for i in range(len(dims) - 1)
        ])
        self.act = nn.SiLU()

    def forward(self, x, t, y):
        """Estimate the """
        x = torch.cat([x, y], dim=-1)
        
        for i, layer in enumerate(self.tlins):
            x = layer(x, t)
            if i < len(self.tlins) - 1:
                x = self.act(x)
        return x








