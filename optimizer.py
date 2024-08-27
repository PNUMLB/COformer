import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import warnings
from typing import List, Tuple
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from dataset import CODataset, Preprocessing
from models.modelwrapper import ModelWrapper
from analyzer import AnalyzerBias, AnalyzerAdaptation, AnalyzerStability

warnings.filterwarnings(action='ignore')

class Optimizer:
    def __init__(self, seed=1, device='cpu', species='homosapiens', genetic_dictionary=None):
        if genetic_dictionary is None:
            raise ValueError("Genetic dictionary must be provided.")
        self.seed = seed
        self.device = device
        self.species = species
        self.genetic_dicts = genetic_dictionary

        self._setup_seed()
        self._initialize_dictionaries()

        self.preprocessing = Preprocessing(self.seq2index, self.label2index, self.index2codon)
        self.analyzer_bias = AnalyzerBias(species, genetic_dictionary=genetic_dictionary)
        self.analyzer_preference = AnalyzerAdaptation(species, genetic_dictionary=genetic_dictionary)
        self.analyzer_stability = AnalyzerStability(species, genetic_dictionary=genetic_dictionary)

        self.verbose = True
        self.ticks = ['\\', '-', '/', '|']
        self.iter_mcdropout = 0

    def _setup_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _initialize_dictionaries(self):
        try:
            self.genetic_code = self.genetic_dicts['genetic_code']
            self.seq2index = self.genetic_dicts['seq2index']
            self.label2index = self.genetic_dicts['label2index']
            self.index2codon = self.genetic_dicts['index2codon']
            self.index2seq = {index: seq for seq, index in self.seq2index.items()}
            self.index2aminoacid = self.genetic_dicts['index2aminoacid']

            self.list_codon = self.genetic_dicts['list_seqs']['list_codon']
            self.list_pad = self.genetic_dicts['list_seqs']['list_pad']
            self.list_aminoacid = self.genetic_dicts['list_seqs']['list_aminoacid']
            self.list_seq = self.list_codon + self.list_pad + self.list_aminoacid

            self.all2index = {seq: index for index, seq in enumerate(self.list_seq)}
            self.index2all = {index: seq for index, seq in enumerate(self.list_seq)}

        except KeyError as e:
            raise ValueError(f"Key not found in genetic dictionary: {e}")

        self.index_of_pad = self.seq2index.get('PAD', -1)
        self.len_codon_char = len(self.index2codon['PAD'])
        self.len_aminoacid_char = len(self.list_aminoacid)

    def define_model(self, file_name):
        self.dir_path = f'./saved_models/{file_name}/'
        self.model_params_path = f'{self.dir_path}params.json'

        with open(self.model_params_path, 'r') as file:
            params = json.load(file)

        self.model = ModelWrapper(params).to(self.device)
        try:
            self.model.load_state_dict(torch.load(f'{self.dir_path}model_state.pth', map_location=self.device))
        except:
            self.model.embedding.load_state_dict(torch.load(f'{self.dir_path}embedding_state.pth', map_location=self.device))
            self.model.backbone.load_state_dict(torch.load(f'{self.dir_path}backbone_state.pth', map_location=self.device))
            self.model.to_out.load_state_dict(torch.load(f'{self.dir_path}to_out_state.pth', map_location=self.device))
        self.model.eval()

    def forward(self, model, x):
        n_iter = self.iter_mcdropout if self.iter_mcdropout > 0 else 1
        emb = model.pe(model.embedding(x))
        emb_context, _ = model.backbone(emb)
        emb_dropout = self.mcdropout(emb_context)
        out = model.to_out(emb_dropout)
        outs = out

        for _ in range(n_iter - 1):
            outs += model.to_out(self.mcdropout(emb_context))

        return outs / n_iter, emb_context

    def get_optimized_codons(self, model, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        df, contexts = self._calculate_optimization_metrics(model, df)
        df = self._calculate_codon_optimization_metrics(df)
        return df

    def _calculate_optimization_metrics(self, model, df):
        df['sequence_index'] = df['Sequence'].apply(self.sequence_to_indices)
        for i, row in df.iterrows():
            df_iter = row.to_frame().T
            len_seq = len(df_iter['Sequence'].values[0])
            aminoacid, n_pads = self.preprocessing.process_sequences(
                df_iter['sequence_index'].tolist(),
                self.preprocessing.index_of_pad_input,
                len_seq
            )
            loader = DataLoader(CODataset(aminoacid, aminoacid, aminoacid, n_pads), batch_size=None, shuffle=False)
            aminoacids, preds, pads, contexts = self.inference(model, loader)
            preds_labels = torch.argmax(preds, dim=-1)
            preds_codons = self.index2codon_opt(aminoacids.cpu().numpy(), preds_labels.cpu().numpy())

            opt_seq = ''.join(codon for pred_codon in preds_codons for codon in pred_codon if codon != 'PAD')
            df.at[i, 'OptimizedRNA'] = opt_seq
        return df, contexts

    def sequence_to_indices(self, sequence: str) -> List[int]:
        return [self.seq2index[s] for s in sequence]

    def inference(self, model, loader):
        model.eval() if self.iter_mcdropout == 0 else model.train()
        with torch.no_grad():
            input_sequence, pred_sequence, contexts, pads = [], [], [], []
            for data in loader:
                input_batch = data[0].to(self.device)
                pad_batch = data[3]
                input_batch = input_batch.unsqueeze(0)
                pred, context = self.forward(model, input_batch)
                input_sequence.append(input_batch.cpu())
                pred_sequence.append(pred.cpu())
                contexts.append(context)
                pad_batch = pad_batch.view(-1) if len(pad_batch.shape) == 0 else pad_batch
                pads.append(pad_batch.cpu())

            pred_sequence = torch.cat(pred_sequence, dim=0)
            input_sequence = torch.cat(input_sequence, dim=0)
            pads = torch.cat(pads, dim=0)
        return input_sequence, pred_sequence, pads, contexts

    def index2codon_opt(self, input_sequence, preds):
        input_sequence = self.index_to_aminoacids(input_sequence)
        seq_codon = []
        for i, pred in enumerate(preds):
            codon = []
            for j in range(len(pred)):
                tmp = self.index2codon[input_sequence[i][j]][pred[j]]
                codon.append(tmp)
                if tmp not in self.list_codon:
                    print(f'\r {tmp} not in list_codon', end=' ')
                    break
            seq_codon.append(codon)
        return seq_codon

    def index_to_aminoacids(self, sequences: str):
        return [[self.index2seq[sequences[i][j]] for j in range(len(sequences[i]))] for i in range(len(sequences))]

    def _calculate_codon_optimization_metrics(self, df):
        df['OptimizedCodon'] = df['OptimizedRNA'].apply(self.convert2codonseq)
        df['CAI'] = df['OptimizedCodon'].apply(self.analyzer_preference.calculate_cai)
        df['GCContent'] = df['OptimizedCodon'].apply(self.analyzer_stability.calculate_gc_content_position)
        df['GC3'] = df['OptimizedCodon'].apply(lambda x: self.analyzer_stability.calculate_gc_content_position(x, position=3))
        return df

    def convert2codonseq(self, seq: str):
        return [seq[i:i + 3] for i in range(0, len(seq), 3)]
