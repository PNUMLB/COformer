from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from Bio import SeqIO
from sklearn.model_selection import train_test_split

import os
import torch
import numpy as np
import pandas as pd


class COData:
    VALID_NUCLEOTIDES = {'A', 'C', 'U', 'G'}
    def __init__(self, data_name=None, species='ecoli', genetic_dictionary=None, clip_size=256, seed=0):
        self.data_name = data_name
        self.seed = seed
        self.clip_size = clip_size
        self.species = species
        self.genetic_dicts = genetic_dictionary
        self.initialize_paths()
        self.initialize_dictionaries()
        # self.df = self.load_data()
        self.preprocessing = Preprocessing(self.seq2index, self.label2index, self.index2codon)

    def initialize_paths(self):
        self.dir_path = './data/'+self.species+'/'
        if self.data_name is not None:
            self.data_csv_path = os.path.join(self.dir_path, f'{self.data_name}.csv')
        self.data_faa_path = os.path.join(self.dir_path, f'{self.species}.faa')

    def initialize_dictionaries(self):
        try:
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

        self.index_of_pad_input = self.seq2index.get('PAD', -1)
        self.index_of_pad_label = self.label2index.get('PAD', -1)
        self.len_labels = len(self.index2codon['A'])
        self.len_codons = len(self.list_codon)
        self.len_aminoacids = len(self.list_aminoacid)

    def fasta2df(self, file_path, save=True) -> pd.DataFrame:
        samples = []
        for dna in SeqIO.parse(file_path, "fasta"):
            name = dna.name
            rna = dna.seq.transcribe()
            protein = str(rna.translate())
            aminoacid = [a for a in protein]
            codon = [str(rna[i:i+3]) for i in range(0, len(rna), 3)]

            if set(str(rna)).issubset(self.VALID_NUCLEOTIDES):
                samples.append([name, dna, rna, codon, protein, aminoacid, len(codon), len(aminoacid)])
            
        column_names = ['Name', 'DNA', 'Protein']
        df = pd.DataFrame(samples, columns=column_names)
        df = df[df['len_AminoAcid'] == df['len_Codon']].reset_index(drop=True)
        df['length'] = df['len_AminoAcid']
        return df

    def load_data(self, data_name=None):
        if data_name is not None:
            self.data_csv_path = os.path.join(self.dir_path, f'{data_name}.csv')

        if data_name in ['all', None, 'None']:
            df_novopro = pd.read_csv(self.dir_path+'/novopro.csv')
            df_genscript = pd.read_csv(self.dir_path+'/genscript.csv')
            df_thermofisher = pd.read_csv(self.dir_path+'/thermofisher.csv')
            df = pd.concat([df_novopro, df_genscript, df_thermofisher], axis = 0)
            df = df.reset_index(drop=True)
        else:
            df = pd.read_csv(self.data_csv_path)
        self.df = self.preprocess_df(df)

    def preprocess_df(self, df):
        df['RNA'] = df['DNA'].apply(self.dna2rna)
        df['AminoAcid'] = df['Protein'].apply(self.seq2aa)
        df['len_AminoAcid'] = df['AminoAcid'].apply(len)

        df['Codon'] = df['RNA'].apply(self.seq2codon)
        df['len_Codon'] = df['Codon'].apply(len)
        
        df['length'] = df['Protein'].apply(len)
        return df.reset_index(drop=True)

    def dna2rna(self, seq: str):
        return seq.replace('T', 'U')
    
    def seq2codon(self, seq:str):
        return [seq[i:i+3] for i in range(0, len(seq), 3)] 
    
    def seq2aa(self, seq:str):
        return [seq[i] for i in range(len(seq))]

    def split_dataset(self, split_rate=[0.6, 0.2, 0.2]) -> None:
        list_name = self.df.Name.unique()
        list_name = np.array(list_name)

        train_name, tmp_name = train_test_split(list_name, test_size=1-split_rate[0], random_state=self.seed)
        valid_name, test_name = train_test_split(tmp_name, test_size=split_rate[1] / (split_rate[1] + split_rate[2]), random_state=self.seed)

        self.df_train = self.df[self.df['Name'].isin(train_name)].reset_index(drop=True)
        self.df_valid = self.df[self.df['Name'].isin(valid_name)].reset_index(drop=True)
        self.df_test = self.df[self.df['Name'].isin(test_name)].reset_index(drop=True)

    def prepare_train_valid_test(self, df_train=None, df_valid=None, df_test=None, clip_size=None):

        df_train = self.df_train if df_train is None else df_train
        df_valid = self.df_valid if df_valid is None else df_valid
        df_test = self.df_test if df_test is None else df_test

        data_train = self.prepare_datasets_from_df(df_train, clip_size)
        data_valid = self.prepare_datasets_from_df(df_valid, clip_size)
        data_test = self.prepare_datasets_from_df(df_test, clip_size)
       
        return data_train, data_valid, data_test
    
    def prepare_datasets_from_df(self, df, clip_size=None):
        clip_size = clip_size or self.clip_size

        aa, pads = self.preprocessing.preprocess_aminoacid(df, clip_size)
        codon, _ = self.preprocessing.preprocess_codon(df, clip_size)
        label, _ = self.preprocessing.preprocess_label(df, clip_size)

        return [aa, codon, label, pads]

class Preprocessing:
    def __init__(self, seq2index: Dict[str, int], label2index: Dict[str, int], index2codon: Dict[str, Dict[int, str]]):
        self.seq2index = seq2index
        self.label2index = label2index
        self.index_of_pad_input = self.seq2index['PAD']
        self.index_of_pad_label = self.label2index['PAD']
        self.value_of_pad_n_codon = len(index2codon['A'])

    def clip_and_pad(self, sequence: List[int], index_of_pad: int, clip_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        chunks = [sequence[i:i+clip_size] for i in range(0, len(sequence), clip_size)]
        pad_counts = []
        for i in range(len(chunks)):
            if len(chunks[i]) < clip_size:
                pad_count = clip_size - len(chunks[i])
                chunks[i].extend([index_of_pad] * pad_count)
            else:
                pad_count = 0
            pad_counts.append(pad_count)

        return np.array(chunks), pad_counts

    def sequence_to_indices(self, sequence: str) -> List[int]:
        result = []
        for s in sequence:
            result.append(self.seq2index[s])
        return result

    def label_to_indices(self, sequence: str) -> List[int]:
        result = []
        for s in sequence:
            result.append(self.label2index[s])
        return result

    def process_sequences(self, sequences: List[List[int]], index_of_pad:int, clip_size = None) -> Tuple[np.ndarray, np.ndarray]:
        all_chunks = []
        all_pad_counts = []

        for sequence in sequences:
            chunks, pad_counts = self.clip_and_pad(sequence, index_of_pad, clip_size)
            all_chunks.extend(chunks)
            all_pad_counts.extend(pad_counts)

        return np.array(all_chunks), np.array(all_pad_counts)
    
    def preprocess_aminoacid(self, df: pd.DataFrame, clip_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        df['AminoAcid_index'] = df['AminoAcid'].apply(self.sequence_to_indices)
        aminoacid, n_pads = self.process_sequences(df['AminoAcid_index'].tolist(), self.index_of_pad_input, clip_size)
        return [aminoacid, n_pads]
    
    def preprocess_codon(self, df: pd.DataFrame, clip_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        df['Codon_index'] = df['Codon'].apply(self.sequence_to_indices)
        codon, n_pads = self.process_sequences(df['Codon_index'].tolist(), self.index_of_pad_input, clip_size)
        return [codon, n_pads]
    
    def preprocess_label(self, df: pd.DataFrame, clip_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        df['Label_index'] = df['Codon'].apply(self.label_to_indices)
        label, n_pads = self.process_sequences(df['Label_index'].tolist(), self.index_of_pad_label, clip_size)
        return [label, n_pads]
    
    def preprocess_n_codon_label(self, df: pd.DataFrame, clip_size: int = 512) -> np.ndarray:
        df = df.copy()
        df['n_codon_label'] = df['Codon'].apply(lambda x: len(x))
        n_codon_label, n_pads = self.process_sequences(df['Label_index'].tolist(), self.value_of_pad_n_codon, clip_size)
        return [n_codon_label, n_pads]


class CODataset(Dataset):
    def __init__(
            self, 
            aa_sequence: torch.Tensor, 
            input_sequence: torch.Tensor, 
            label_sequence: torch.Tensor, 
            n_pads: torch.Tensor,
    ) -> None:
        self.aa_sequence = aa_sequence
        self.input_sequence = input_sequence
        self.label_sequence = label_sequence
        self.n_pads = n_pads
    
    def __len__(self) -> int:
        return len(self.n_pads)

    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.aa_sequence[idx], self.input_sequence[idx], self.label_sequence[idx], self.n_pads[idx]