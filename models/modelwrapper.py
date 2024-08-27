from typing import Dict
from torch import Tensor
from .coformer import COformer
from .bilstm import BiLSTM
from .transformer import Transformer

import torch
import torch.nn as nn
import torch.nn.init as init
import math


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, ):
        super(LearnedPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.positional_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        position_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand(-1, x.size(1))
        positional_encodings = self.positional_embeddings(position_indices)
        x = x + positional_encodings
        return x


class Augmentation(nn.Module):
    def __init__(self, dim: int, p_mask: float) -> None:
        super().__init__()
        self.emb_mask = nn.Parameter(torch.Tensor(1, dim))
        self.p_mask = p_mask
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.emb_mask.shape[-1])
        init.uniform_(self.emb_mask, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        masked_x = x.clone()

        m = int(self.p_mask * n)

        masks = []
        for i in range(b):
            mask_indices = torch.randperm(n)[:m]
            masks.append(mask_indices)
            masked_x[i, mask_indices, :] = self.emb_mask

        return masked_x, masks

class ModelWrapper(nn.Module):
    def __init__(self, params: Dict, device: str = 'cpu') -> None:
        super(ModelWrapper, self).__init__()
        self.params = params
        self.device = device
        len_tokens = params['len_aminoacids'] + params['len_codons'] + 1
        self.name = params.get('name', '')

        if self.name in ['coformer', 'transformer', 'realjmformer']:
            self.pe = LearnedPositionalEncoding(params['dim_in'])
        else:
            self.pe = nn.Identity()

        self.embedding = nn.Embedding(len_tokens, params['dim_in'])
        self.backbone = self._get_backbone()

        self.to_out = nn.Linear(self.params['dim_in'], self.params['dim_out'])
        self.num_outs = params['len_aminoacids'] + 1
        self.dim_out = params['dim_out']

        self.len_to_codon = params['len_codons'] + 1
        self.len_to_aa = params['len_aminoacids'] + 1

        self.to_codon = nn.Linear(self.params['dim_in'], self.len_to_codon)
        self.to_aa = nn.Linear(self.params['dim_in'], self.len_to_aa)
        self.to_proj = nn.Sequential(nn.Linear(self.params['dim_in'], self.params['dim_in']))
        
        self.mcdropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> Tensor:
        emb = self.embedding(x)
        emb = self.pe(emb)
        emb = self.backbone(emb)
        return self.to_out(emb)

    def _get_backbone(self) -> nn.Module:
        BACKBONES = {
            'coformer': COformer(
                dim=self.params['dim_in'],
                depth=self.params['depth'],
                n_heads=self.params['n_heads'],
                dim_attn=self.params['dim_attn'],
                mult_ff=self.params['mult_ff'],
                dropout_ff=self.params['dropout_ff'],
                dropout_attn=self.params['dropout_attn'],
            ).to(self.device),

            'bilstm': BiLSTM(
                dim=self.params['dim_in'],
                depth=self.params['depth'],
                dropout_ff=self.params['dropout_ff'],
            ).to(self.device),

            'transformer': Transformer(
                dim=self.params['dim_in'],
                depth=self.params['depth'],
                n_heads=self.params['n_heads'],
                dim_attn=self.params['dim_attn'],
                mult_ff=self.params['mult_ff'],
                dropout_ff=self.params['dropout_ff'],
                dropout_attn=self.params['dropout_attn'],
            ).to(self.device),
        }

        return BACKBONES.get(self.name, None)