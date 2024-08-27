import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import einsum
import math

class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        dim_hidden = int(dim*mult)
        self.net = nn.Sequential(
            nn.Linear(dim, dim_hidden*2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim)
        )
    def forward(self, x, **kwargs):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        n_heads=8,
        dim_attn=16,
        dropout=0.,
    ):
        super().__init__()

        dim_inner = dim_attn * n_heads
        self.n_heads = n_heads

        self.scale = dim_attn ** -0.5

        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_k = nn.Linear(dim, dim_inner, bias=False)
        self.to_v = nn.Linear(dim, dim_inner, bias=False)

        self.to_out = nn.Linear(dim_inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n_heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.dropout(self.to_out(out)) 

class Encoder(nn.Module):
    def __init__(
            self,
            dim,
            n_heads,
            dim_attn,
            mult_ff,
            dropout_ff,
            dropout_attn,
    ):
        super().__init__()

        self.norm_attn = nn.LayerNorm(dim)
        self.attention = SelfAttention(dim, n_heads, dim_attn, dropout_attn)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult_ff, dropout_ff)

    def forward(self, x):
        # x_norm = self.norm_attn(x)
        out_attn  = self.attention(x)
        x = self.norm_attn(out_attn + x)

        # x_norm = self.norm_ff(x)
        out_ff = self.ff(x)
        x = self.norm_ff(out_ff + x)

        return x 

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        n_heads,
        dim_attn,
        mult_ff,
        dropout_ff,
        dropout_attn,
    ):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([
            Encoder(
                dim=dim,
                n_heads=n_heads,
                dim_attn=dim_attn,
                mult_ff=mult_ff,
                dropout_ff=dropout_ff,
                dropout_attn=dropout_attn
            ) for _ in range(depth)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x, x