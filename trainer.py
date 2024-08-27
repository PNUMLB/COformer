from math import inf
from torch.utils.data import DataLoader
from models.modelwrapper import ModelWrapper, Augmentation
from utils import get_optimizer
from dataset import COData, CODataset
from optimizer import Optimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json

class Trainer:
    def __init__(
            self, 
            species='homosapiens',
            genetic_dictionary=None,
            seed=1, 
            device='cpu', 
            file_name='best_model', 
            loss_fn=nn.CrossEntropyLoss()
    ):
        self.seed = seed
        self.device = device
        self._setup_seed()
        self.loss_fn = loss_fn
        self.optim = None
        self.best_model_state = None
        if genetic_dictionary is None:
            raise ValueError("Genetic dictionary must be provided.")
        self.optimizer = Optimizer(seed=seed, device=device, species=species, genetic_dictionary=genetic_dictionary)
        self._set_file_paths(file_name)
        self.verbose = True
        self.ticks=['\\', '-', '/', '|']

    def _setup_seed(self):
        """Sets the seed for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def _create_directory(self, path):
        """Create directory if it does not exist."""
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

    def _set_file_paths(self, file_name):
        """Set file paths for saving and loading models."""
        self.dir_path = f'./saved_models/{file_name}/'
        self._create_directory(self.dir_path)
                
        # Model
        self.model_state_path = f'{self.dir_path}model_state.pth'
        self.model_path = f'{self.dir_path}model.pth'
        # Embedding
        self.embedding_state_path = f'{self.dir_path}embedding_state.pth'
        self.embedding_path = f'{self.dir_path}embedding.pth'
        # Backbone
        self.backbone_state_path = f'{self.dir_path}backbone_state.pth'
        self.backbone_path = f'{self.dir_path}backbone_state.pth'
        # To out
        self.to_out_state_path = f'{self.dir_path}to_out_state.pth'
        self.to_out_path = f'{self.dir_path}to_out.pth'
        # pretrained
        self.model_state_pt_path = f'{self.dir_path}model_pt_state.pth'
        self.model_pt_path = f'{self.dir_path}model_pt.pth'
        # Parameters
        self.model_params_path = f'{self.dir_path}params.json'

    def _save_params(self, params): # Save the model parameters
        with open(self.model_params_path, 'w') as file:
            json.dump(params, file, indent=4)

    def split_dataset(self, df, split_rate=[0.6, 0.2, 0.2]):
        list_entry = df['Entry'].unique()
        list_train, tmp_data = train_test_split(list_entry, test_size=split_rate[1]+split_rate[2], random_state=self.seed)    
        list_valid, list_test = train_test_split(tmp_data, test_size=0.5, random_state=self.seed)
        df_train = df[df['Entry'].isin(list_train)]
        df_valid = df[df['Entry'].isin(list_valid)]
        df_test = df[df['Entry'].isin(list_test)]
        return df_train, df_valid, df_test
        
    def _set_data(self, data, split_rate=[0.6, 0.2, 0.2]):
        self.data = data
        self.data.split_dataset(split_rate)
        self.df_train, self.df_valid, self.df_test = self.data.df_train, self.data.df_valid, self.data.df_test

    def set_loaders(self, clip_size=512, batch_size=512):
        """Sets up the DataLoader for training, validation, and testing."""
        self.train_loader =[]
        self.valid_loader = []
        self.test_loader = []

        data_train, data_valid, data_test = self.data.prepare_train_valid_test(self.df_train, self.df_valid, self.df_test, clip_size=clip_size)

        self.train_loader.append(DataLoader(CODataset(data_train[0], data_train[1], data_train[2], data_train[3]), batch_size, shuffle=True))
        self.valid_loader.append(DataLoader(CODataset(data_valid[0], data_valid[1], data_valid[2], data_valid[3]), batch_size, shuffle=False))
        self.test_loader.append(DataLoader(CODataset(data_test[0], data_test[1], data_test[2], data_test[3]), batch_size, shuffle=False))

    def define_model(self, params, len_aminoacids=None, len_codons=None, p_mask=0.4):
        if len_aminoacids is None or len_codons is None:
            params['len_aminoacids'] = self.data.len_aminoacids
            params['len_codons'] = self.data.len_codons
            self.len_aminoacids = params['len_aminoacids']
            self.len_codons = params['len_codons']

        params['dim_out'] = self.data.len_labels
        self.dim_out = params['dim_out']
        self.model = ModelWrapper(params).to(self.device)
        self.best_model_state = self.model.state_dict()
        self.augmentation = Augmentation(dim=self.model.params['dim_in'], p_mask=p_mask).to(self.device)
        self._save_params(params)

    def define_pretraining_modules(self, p_mask=0.4):
        if self.model is None:
            raise ValueError("Model must be defined first. Use define_model() method.")
        self.model._set_pretraining_modules(device=self.device)
        self.best_model_pt_state = self.model.state_dict()
    
    def reset_pretraining_modeuls(self):
        self.model.to_codon = None
        self.model.to_aa = None
        self.model.to_contrast = None
        self.augmentation = None
        
    def save_model(self):
        torch.save(self.model, self.model_path)
        torch.save(self.model.embedding, self.embedding_path)
        torch.save(self.model.backbone, self.backbone_path)
        torch.save(self.model.to_out, self.to_out_path)

    def load_model(self):
        self.model = torch.load(self.model_path)
        self.model.eval()
        self.model.embedding = torch.load(self.embedding_path)
        self.model.embedding.eval()
        self.model.backbone = torch.load(self.backbone_path)
        self.model.backbone.eval()
        self.model.to_out = torch.load(self.to_out_path)
        self.model.to_out.eval()

    def save_module(self, module, path):
        torch.save(module, path)

    def load_module(self, path):
        self.model = torch.load(path)
        self.model.eval()

    def save_model_state(self, path):
        self.best_model_state = copy.deepcopy(self.model.state_dict())
        self.best_embedding_state = copy.deepcopy(self.model.embedding.state_dict())
        self.best_backbone_state = copy.deepcopy(self.model.backbone.state_dict())
        self.best_to_out_state = copy.deepcopy(self.model.to_out.state_dict())

        torch.save(self.best_model_state, path)
        torch.save(self.best_embedding_state, self.embedding_state_path)
        torch.save(self.best_backbone_state, self.backbone_state_path)
        torch.save(self.best_to_out_state, self.to_out_state_path)

    def load_model_state(self, module_name=None, path=None):
        if module_name == 'embedding':
            self.model.embedding.load_state_dict(torch.load(path), strict=False)
        elif module_name == 'backbone':
            self.model.backbone.load_state_dict(torch.load(path), strict=False)
        elif module_name == 'to_out':
            self.model.to_out.load_state_dict(torch.load(path), strict=False)
        else:
            if path is None:
                path = self.model_state_path
            self.model.load_state_dict(torch.load(path), strict=False)

    def set_mcdropout(self, n_iter=0, rate=0.0):
        self.mcdropout = nn.Dropout(p=0.0)
        self.optimizer.mcdropout = nn.Dropout(p=0.0)
        self.iter_mcdropout = n_iter
        self.optimizer.iter_mcdropout = self.iter_mcdropout
        if n_iter > 0:
            self.mcdropout = nn.Dropout(p=rate)
            self.optimizer.mcdropout = nn.Dropout(p=rate)
   
    def logging_loss(self, losses, verbose=True):
        if verbose:
            print('\r|  total  |aminoacid|  codon  |  cosine |    l1   |    l2   |   cov   |   var   |')
            print(f'\r| {losses[0]:.5f} | {losses[1]:.5f} | {losses[2]:.5f} | {losses[3]:.5f} | {losses[4]:.5f} | {losses[5]:.5f} | {losses[6]:.5f} | {losses[7]:.5f} |')
        return {'total': losses[0], 'aa': losses[1], 'codon': losses[2], 'cosine': losses[3], 'l1': losses[4], 'l2': losses[5], 'cov': losses[6], 'var': losses[7]}
    
    def logging_score(self, scores, stats, verbose=True):
        if verbose:
            print('\r| accuracy| f1 macro|f1 weight|   mcc   |   CAI   |  GC   |   GC3   |')
            print(f'\r| {scores[0]:.5f} | {scores[1]:.5f} | {scores[2]:.5f} | {scores[3]:.5f} | {stats[0]:.5f} | {stats[1]:.5f} | {stats[2]:.5f} |')
        return {'acc': scores[0], 'f1_macro': scores[1], 'f1_weight': scores[2], 'mcc': scores[3], 'CAI': stats[0], 'GC': stats[1],  'GC3': stats[2]}

    def fit(
            self, 
            epochs=100, 
            optimizer='adamw', 
            lr=1e-3, 
            earlystop=15, 
            wo_pad=False, 
            verbose=True, 
            lambda_codon=0.0,
            lambda_emb=0.0,
            lambda_l1=0.0,
            lambda_l2=0.0,
            lambda_cov=0.0,
            lambda_var=0.0,
            window=16,
            step_size=8,
            mcdropout_rate=0.0,
            iter_mcdropout=0
        ):
        self.log_fit = {'train': {}, 'valid': {}, 'test': {}}
        self.epochs = epochs
        self.wo_pad = wo_pad
        self.verboes = verbose
        self.lambda_codon = lambda_codon
        self.lambda_emb = lambda_emb
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_cov = lambda_cov
        self.lambda_var = lambda_var
        self.window = window
        self.step_size = step_size

        self.set_mcdropout(iter_mcdropout, mcdropout_rate)
        self.optim = get_optimizer(params=self.model.parameters(), opt_name=optimizer, lr=lr)

        earlystop = earlystop if earlystop is not None else epochs
        best_valid, patience = 0, 0
        
        for epoch in range(1, epochs + 1):
            
            self.train_mode = True
            self.model.train()
            l_train, s_train = self.step_epoch(self.train_loader)

            if self.verbose:
                print(f'\r[Train][{epoch}/{epochs}]_______________________________________________________________')
            self.log_fit['train'][epoch] = {'losses': self.logging_loss(l_train)}

            self.train_mode = False
            self.model.eval()
            with torch.no_grad():
                l_valid, s_valid = self.step_epoch(self.valid_loader)
                stat_valid= self.get_statistics(self.data.df_valid)
                if self.verbose:
                    print(f'\r[Valid][{epoch}/{epochs}]______________________________________________________________')
                self.log_fit['valid'][epoch] = {'losses': self.logging_loss(l_valid, verbose=False), 'scores': self.logging_score(s_valid, stat_valid)}
                

                score_valid = s_valid[0]
                isBest = True if score_valid > best_valid else False 
                if isBest:
                    e_best, patience = epoch, 0
                    best_valid = score_valid
                    l_b_valid, s_b_valid = l_valid, s_valid
                    l_test, s_test = self.step_epoch(self.test_loader)
                    stat_test= self.get_statistics(self.data.df_test)
                    if self.verbose:
                        print(f'\r[test][{epoch}/{epochs}]_______________________________________________________________')
                    self.log_fit['test'][e_best] = {'losses': self.logging_loss(l_test, verbose=False), 'scores': self.logging_score(s_test, stat_test)}
                    self.save_model_state(self.model_state_path)
                patience += 1
                                                     
            if verbose:
                print(f'---------------------------------------- [Best] ----------------------------------------')
                print('| Data  |Epoch| | loss | accuracy | f1 macro|f1 weight|   mcc   |   CAI   |  GC   |   GC3   |')
                print(f'| Valid |  {e_best}  | {l_b_valid[0]:.5f} | {s_b_valid[0]:.5f} | {s_b_valid[1]:.5f} | {s_b_valid[2]:.5f} |', end='')
                print(f' {s_b_valid[3]:.5f} | {stat_valid[0]:.5f} | {stat_valid[1]:.5f} | {stat_valid[2]:.5f} |')
                print(f'| test  |  {e_best}  | {l_test[0]:.5f} | {s_test[0]:.5f} | {s_test[1]:.5f} | {s_test[2]:.5f} |', end='')
                print(f' {s_test[3]:.5f} | {stat_test[0]:.5f} | {stat_test[1]:.5f} | {stat_test[2]:.5f} |')
                print(f'---------------------------------------------------------------------------------------')
                print()

            if patience > earlystop:
                if verbose:
                    print('Early-stopping....!')
                break

        self.load_model_state()
        
    def step_epoch(self, data_loader):
        losses_epoch = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        score_epoch = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        for loader in data_loader:
            losses, _, _, labels, preds, pads = self.step_loader(loader)
            score = self.get_scores(labels, preds, pads)
            
            losses_epoch = [losses_epoch[i] + losses[i] for i in range(len(losses))]
            score_epoch = [score_epoch[i] + score[i] for i in range(len(score))]

        losses_epoch = [losses_epoch[i]/len(data_loader) for i in range(len(losses_epoch))]
        score_epoch = [score_epoch[i]/len(data_loader) for i in range(len(score_epoch))]
        return losses, score_epoch
    
    def step_loader(self, loader):
        seqs_aa, seqs_codon, seqs_label, pads, seqs_pred = torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
        losses = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Total, AA, Codon, Contrastive, l1, l2, cov, var
        for i, data in enumerate(loader):
            loss, loss_aa, loss_codon, loss_contrastive = 0.0, 0.0, 0.0, 0.0

            seq_aa, seq_codon, seq_label, pad = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device)
            if self.train_mode:
                self.optim.zero_grad()
            
            seq_pred, context_aa = self.forward(seq_aa)
            loss_aa = self.get_loss(seq_pred, seq_label, pad)
            loss += loss_aa

            if self.train_mode:
                lambda_value = self.lambda_codon + self.lambda_emb
                if lambda_value > 0.0:
                    emb = self.model.embedding(seq_codon)
                    emb = self.model.pe(emb)
                    emb_out, context_codon = self.model.backbone(emb)

                    if self.lambda_codon > 0.0:
                        seq_pred_codon = self.model.to_out(emb_out)
                        loss_codon = self.get_loss(seq_pred_codon, seq_label, pad)
                        loss = loss + (self.lambda_codon * loss_codon)


                    if self.lambda_emb > 0.0:
                        context_aa = self.model.to_proj(context_aa)
                        context_codon = self.model.to_proj(context_codon)
                        loss_contrastive = self.CELOSS_window(context_aa, context_codon, tau=0.5, window=self.window, step_size=self.step_size)
                        loss = loss + (self.lambda_emb * loss_contrastive)
                
                loss.backward()
                self.optim.step()
            
            losses[0] += loss
            losses[1] += loss_aa
            losses[2] += loss_codon
            losses[3] += loss_contrastive

            if self.verbose:
                print('\r', self.ticks[(i%4)], f'[{i+1}/{len(loader)}] Loss: {loss:.5f} [batch: {len(seq_aa)} length: {len(seq_aa[0])}]', end = '')

            seqs_aa = torch.cat([seqs_aa, seq_aa.cpu()], dim=0)
            seqs_codon = torch.cat([seqs_codon, seq_aa.cpu()], dim=0)
            seqs_label = torch.cat([seqs_label, seq_label.cpu()], dim=0)
            seqs_pred = torch.cat([seqs_pred, seq_pred.cpu()], dim=0)
            pads = torch.cat([pads, pad.cpu()], dim=0)          
            
        losses = [losses[i]/len(loader) for i in range(len(losses))]
        return losses, seqs_aa, seqs_codon, seqs_label, seqs_pred, pads

    def forward(self, x):
        n_iter=1
        if self.iter_mcdropout > 0:
            n_iter = 1 if self.train_mode else self.iter_mcdropout

        emb = self.model.embedding(x)
        emb = self.model.pe(emb)
        emb_out, emb_context = self.model.backbone(emb)
        out = self.model.to_out(self.mcdropout(emb_out))

        for _ in range(n_iter-1):
            out += self.model.to_out(self.mcdropout(emb_out))

        return out / n_iter, emb_context
    
    def without_pads(self, pred, codon, n_pads, amino=None):
        amino_wo_pads, pred_wo_pads, codon_wo_pads = [], [], []
        for i, n_pad in enumerate(n_pads):
            if n_pad > 0:
                if amino is not None:
                    amino_wo_pads.append(amino[i, :-int(n_pad.item())])
                codon_wo_pads.append(codon[i, :-int(n_pad.item())])
                pred_wo_pads.append(pred[i, :-int(n_pad.item())])
            else:
                if amino is not None:
                    amino_wo_pads.append(amino[i, :])
                pred_wo_pads.append(pred[i, :])
                codon_wo_pads.append(codon[i, :])

        if amino is not None:
            amino_wo_pads = torch.cat(amino_wo_pads).to(self.device)
        else:
            amino_wo_pads = None

        pred_wo_pads = torch.cat(pred_wo_pads).to(self.device)
        codon_wo_pads = torch.cat(codon_wo_pads).to(self.device)
        return pred_wo_pads, codon_wo_pads, amino_wo_pads

    def get_loss(self, predictions, targets, n_pads):
        if self.wo_pad:
            predictions, targets, _ = self.without_pads(predictions, targets, n_pads)
        else:
            _, _, d = predictions.shape
            predictions = predictions.view(-1, d)
            targets = targets.view(-1)

        loss = self.loss_fn(predictions, targets)
        return loss

    def get_scores(self, codons, preds, n_pads):
        print('\r Calculating...', end='')
        if self.wo_pad:
            preds, codons, _ = self.without_pads(preds, codons, n_pads)
        
        print('ACC ', end=' ')
        codons_np = codons.view(-1).long().cpu().detach().numpy()
        preds_np = torch.argmax(preds, dim=-1).view(-1).cpu().numpy()
        correct = preds_np == codons_np
        acc = correct.mean()

        print('f1m ', end=' ')
        f1_ma = f1_score(codons_np, preds_np, average='macro')
        print('f1w ', end=' ')
        f1_w = f1_score(codons_np, preds_np, average='weighted')
        print('MCC ', end=' ')
        mcc = matthews_corrcoef(codons_np, preds_np)
        return [acc, f1_ma, f1_w, mcc]

    def get_statistics(self, df):
        df = df[['Name', 'AminoAcid']].rename(columns={'Name':'Name', 'AminoAcid':'Sequence'})
        df['Type'] = ['Protein']*df.shape[0]
        df = df[['Name', 'Sequence', 'Type']]
        df = df.drop_duplicates(subset='Name')
        df = df.reset_index(drop=True)

        print('\r Calculating...', end='')
        df_stat = self.optimizer.get_optimized_codons(self.model, df)
        df_score = self.optimizer._calculate_codon_optimization_metrics(df_stat)
        print('CAI ',end=' ')
        cai=df_score['CAI'].mean()
        print('GC ',end=' ')
        gc=df_score['GCContent'].mean()
        print('GC3 ',end=' ')
        gc3=df_score['GC3'].mean()

        return [cai, gc, gc3]

    def CELOSS_window(self, out1, out2, tau=0.5, window=16, step_size=16):
        total_loss = 0.0
        num_windows = 0

        seq_len = out1.size(1)
        for i in range(0, seq_len, step_size):
            if i + window <= seq_len:
                out1_window = out1[:, i:i + window, :]
                out2_window = out2[:, i:i + window, :]
            else:
                # Handle the last window if seq_len is not perfectly divisible by window size
                out1_window = out1[:, -window:, :]
                out2_window = out2[:, -window:, :]

            loss = self.CELoss(out1_window, out2_window, tau)
            total_loss += loss
            num_windows += 1

            # Debugging output
            if torch.any(loss < 0):
                print(f"Negative loss detected in window [{i}:{i + window}]. Loss: {loss.item()}")

        # Ensure the total loss is non-negative
        final_loss = total_loss / num_windows if num_windows > 0 else 0.0
        if final_loss < 0:
            print("Final loss is negative. This should not happen.")
            print("total_loss:", total_loss)
            print("num_windows:", num_windows)
            print("final_loss:", final_loss)

        return final_loss
    
    def CELoss(self, out1, out2, tau=0.5):
        eps = 1e-10
        sim = F.cosine_similarity(out1, out2, dim=2)
        
        # Add numerical stability check for sim values
        sim = torch.clamp(sim, min=-1 + eps, max=1 - eps)

        exp_sim = torch.exp(sim / tau)

        # Ensure numerical stability
        exp_sim_sum = exp_sim.sum(dim=1, keepdim=True) + eps
        exp_sim_diagonal = torch.diagonal(exp_sim, dim1=-2, dim2=-1) + eps

        # Compute contrasts and handle potential negative values in logarithm
        contrasts = -torch.log(exp_sim_diagonal / exp_sim_sum)
        contrasts = torch.clamp(contrasts, min=0.0)

        # Compute the mean loss
        loss = contrasts.mean()
        return loss