import os
import random
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, auc as auc3, precision_recall_curve
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn

warnings.filterwarnings(action='ignore')


genetic_code = {
    # RNA to Amino Acid
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'UGU': 'C', 'UGC': 'C',
    'GAU': 'D', 'GAC': 'D',
    'GAA': 'E', 'GAG': 'E',
    'UUU': 'F', 'UUC': 'F',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    'CAU': 'H', 'CAC': 'H',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I',
    'AAA': 'K', 'AAG': 'K',
    'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'AUG': 'M',
    'AAU': 'N', 'AAC': 'N',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAA': 'Q', 'CAG': 'Q',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'UGG': 'W',
    'UAU': 'Y', 'UAC': 'Y',
    'UAA': '*', 'UAG': '*', 'UGA': '*',

    # PAD
    'PAD': 'PAD',

    # DNA to Amino Acid
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TGT': 'C', 'TGC': 'C',
    'GAT': 'D', 'GAC': 'D',
    'GAA': 'E', 'GAG': 'E',
    'TTT': 'F', 'TTC': 'F',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    'CAT': 'H', 'CAC': 'H',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
    'AAA': 'K', 'AAG': 'K',
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATG': 'M',
    'AAT': 'N', 'AAC': 'N',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TGG': 'W',
    'TAT': 'Y', 'TAC': 'Y',
    'TAA': '*', 'TAG': '*', 'TGA': '*',

    'F': ['UUU', 'UUC'], 'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
    'I': ['AUU', 'AUC', 'AUA'], 'M': ['AUG'], 'V': ['GUU', 'GUC', 'GUA', 'GUG'],
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'], 'P': ['CCU', 'CCC', 'CCA', 'CCG'],
    'T': ['ACU', 'ACC', 'ACA', 'ACG'], 'A': ['GCU', 'GCC', 'GCA', 'GCG'],
    'Y': ['UAU', 'UAC'], 'H': ['CAU', 'CAC'], 'Q': ['CAA', 'CAG'],
    'N': ['AAU', 'AAC'], 'K': ['AAA', 'AAG'], 'D': ['GAU', 'GAC'],
    'E': ['GAA', 'GAG'], 'C': ['UGU', 'UGC'], 'W': ['UGG'],
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'], 'G': ['GGU', 'GGC', 'GGA', 'GGG'],
    '*': ['UAA', 'UAG', 'UGA']
}

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device(cuda_num: Optional[int] = None) -> str:
    if cuda_num in [0, 1, 2, 3]:
        return f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu"
    return "cpu"


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_optimizer(params, opt_name: str, lr: float = 1e-4, w_decay: Optional[float] = None) -> optim.Optimizer:
    weight_decay = 0 if w_decay is None else w_decay

    optimizer_dict = {
        'adamw': optim.AdamW,
        'adam': optim.Adam,
        'sgd': optim.SGD
    }
    optimizer_class = optimizer_dict.get(opt_name.lower())
    if optimizer_class:
        return optimizer_class(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Optimizer '{opt_name}' not recognized.")

def dna2rna(seq: str) -> str:
    return seq.replace('T', 'U')

def convert2aminos(seq: str) -> str:
    return ''.join([genetic_code[seq[i:i + 3]] for i in range(0, len(seq), 3)])

def convert2codons(seq: str):
    return [seq[i:i + 3] for i in range(0, len(seq), 3)]
