import argparse
from utils import seed_everything, get_device
from dataset import COData
from trainer import Trainer
import os
import torch
import json
import torch.nn as nn   
from dicts import genetic_dictionary


def convert_to_python_types(obj):
    """
    Recursively convert tensors in a nested structure (like dict or list) to standard Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    else:
        return obj

def run(args_dict: dict) -> None:
    """
    Run the training process based on the provided arguments.
    """
    seed_everything(args_dict['seed'])
    device = get_device(args_dict['device'])
    
    loss_fn = nn.CrossEntropyLoss()

    file_path = str(args_dict['data_name']) + '/' + str(args_dict['species']) + '/' + str(args_dict['name']) + '/' + str(args_dict['file_name']) + '/' + str(args_dict['seed']) + '/'
    trainer = Trainer(
        seed=args_dict['seed'],
        device=device,
        file_name=file_path,
        loss_fn=loss_fn,
        genetic_dictionary=genetic_dictionary
    )

    data = COData(
        data_name=args_dict['data_name'],
        species=args_dict['species'], 
        clip_size=args_dict['clip_size'], 
        seed=args_dict['seed'],
        genetic_dictionary=genetic_dictionary,
    )
    data.load_data()
    trainer._set_data(data)
    trainer.define_model(args_dict)
    trainer.set_loaders(clip_size=args_dict['clip_size'], batch_size=args_dict['batchsize'])

    trainer.fit(
        epochs=args_dict['epoch'],
        optimizer=args_dict['optimizer'],
        lr=args_dict['lr'],
        earlystop=args_dict['earlystop'],
        wo_pad=args_dict['wo_pad'],
        verbose=args_dict['verbose'],
        lambda_codon=args_dict['lambda_codon'],
        lambda_emb=args_dict['lambda_emb'],
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model based on provided arguments.")

    parser.add_argument('--data_name', type=str, default='all', choices=['all', 'genscript', 'thermofisher', 'novopro'])
    parser.add_argument('--species', type=str, default='homosapiens', choices=['homosapiens'])
    parser.add_argument('--clip_size', type=int, default=512, help='Clip size for sequences.')
    
    # Model arguments
    parser.add_argument('--name', type=str, default='coformer', choices=['coformer', 'bilstm', 'transformer' ], help='Model name.')
    parser.add_argument('--depth', type=int, default=4, help='Depth (layers) of backbone model.')
    parser.add_argument('--dim_in', type=int, default=32, help='input embedding dimension')
    parser.add_argument('--dim_attn', type=int, default=16, help='attention embedding dimension')
    parser.add_argument('--dim_out', type=int, default=64, help='output embedding dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='number of heads')
    parser.add_argument('--dropout_attn', type=int, default=0.1, help='dropout rate for attention')
    parser.add_argument('--mult_ff', type=int, default=2, help='multiplication for feedforward')
    parser.add_argument('--dropout_ff', type=int, default=0.1, help='dropout rate for feedforward')

    # Training arguments
    parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--batchsize', type=int, default=512, help='Training batch size.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd'], help='Optimizer choice.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--earlystop', type=int, default=30, help='Early stopping.')
    parser.add_argument('--wo_pad', action='store_false', help='Flag for padding.')
    parser.add_argument('--verbose', action='store_false', help='Flag for verbosity.')
    parser.add_argument('--file_name', type=str, default='best_model', help='file name for saving the best model.')

    # for contrastive loss
    parser.add_argument('--lambda_codon', type=float, default=0.0, help='Lambda for codon loss.')
    parser.add_argument('--lambda_emb', type=float, default=0.0, help='Lambda for embedding loss.')
    parser.add_argument('--window', type=int, default=64, help='Window size for ce loss.')
    parser.add_argument('--step_size', type=int, default=32, help='Step size for ce loss.')

    # Misc arguments
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility.')
    parser.add_argument('--device', type=int, default=-1, choices=[-1, 0, 1], help='Device choice (-1 for CPU, 0 or 1 for GPU).')

    args = parser.parse_args()
    args_dict = vars(args)
    run(args_dict)