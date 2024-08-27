from collections import Counter, defaultdict
import warnings
import numpy as np
import pandas as pd
import math
import RNA

warnings.filterwarnings(action='ignore')

class Analyzer:
    def __init__(self, species: str = "homosapiens", genetic_dictionary=None):
        self.species = species

        if genetic_dictionary is None:
            raise ValueError("Genetic dictionary not provided.")
        else:
            self.genetic_dictionary = genetic_dictionary

        self.list_codon_usage = self.genetic_dictionary['usages'][species]
        self.initialize_codon_usage(self.list_codon_usage)

        self.list_codon = self.genetic_dictionary['list_seqs']['list_codon']
        self.list_aminoacid = self.genetic_dictionary['list_seqs']['list_aminoacid']
        
        self.dict_aa_to_codon = self.genetic_dictionary['aa_to_codon']
        self.dict_codon_to_aa = self.genetic_dictionary['codon_to_aa']

        self.amino_acid_degeneracy = {}
        self.initialize_relative_adaptiveness()
        self.initialize_reference_nucleotide_frequencies()
        
    def initialize_codon_usage(self, codon_usage):
        self.df_codon_usage = pd.DataFrame(codon_usage, columns=['codon', 'amino acid', 'fraction', 'frequency', 'number'])
        self.dict_codon_counts = self.df_codon_usage.set_index('codon')['number'].to_dict()
        self.dict_codon_frequency = self.df_codon_usage.set_index('codon')['frequency'].to_dict()
        self.dict_codon_weights = self.df_codon_usage.set_index('codon')['fraction'].to_dict()
        self.dict_codon_usage = self.df_codon_usage.set_index('codon').to_dict(orient='index')

    def initialize_reference_nucleotide_frequencies(self):
        position_counts = [defaultdict(int), defaultdict(int), defaultdict(int)]
        
        for codon, _, _, _, absolute_frequency in self.list_codon_usage:
            for i, nucleotide in enumerate(codon):
                position_counts[i][nucleotide] += absolute_frequency
        
        self.reference_nucleotide_frequencies = []
        for counts in position_counts:
            total = sum(counts.values())
            frequencies = {nucleotide: count / total for nucleotide, count in counts.items()}
            self.reference_nucleotide_frequencies.append(frequencies)

    def transcription(self, dna: str) -> str:
        return dna.replace('T', 'U')
        
    def convert_to_codon(self, seq: str):
        rna_seq = self.transcription(seq)
        return [rna_seq[i:i + 3] for i in range(0, len(rna_seq) - 2, 3)]

    def translation(self, codon):
        return self.dict_codon_to_aa.get(codon, None)

    def initialize_relative_adaptiveness(self):
        self.relative_adaptiveness = {}
        for aa, codons in self.dict_aa_to_codon.items():
            max_weight = max(self.dict_codon_weights[codon] for codon in codons)
            for codon in codons:
                self.relative_adaptiveness[codon] = self.dict_codon_weights[codon] / max_weight
    
    def initialize_optimal_codons(self):
        optimal_codons = self.df_codon_usage.loc[self.df_codon_usage.groupby('amino acid')['frequency'].idxmax()]
        self.optimal_codon_dict = dict(zip(optimal_codons['amino acid'], optimal_codons['codon']))
        for aa in self.df_codon_usage['amino acid'].unique():
            self.amino_acid_degeneracy[aa] = self.df_codon_usage[self.df_codon_usage['amino acid'] == aa].shape[0]
    

    def calculate_nucleotide_frequencies_position(self, codon_seq):
        position_counts = [defaultdict(int), defaultdict(int), defaultdict(int)]
        total_codons = len(codon_seq)

        for codon in codon_seq:
            if len(codon) == 3:
                for i, nucleotide in enumerate(codon):
                    position_counts[i][nucleotide] += 1

        position_frequencies = []
        for counts in position_counts:
            total = sum(counts.values())
            frequencies = {nucleotide: count / total for nucleotide, count in counts.items()}
            position_frequencies.append(frequencies)

        return position_frequencies
    
    def calculate_relative_adaptiveness_observed(self, codon_seq):
        observed_codon_counts = {}
        for codon in codon_seq:
            if codon in observed_codon_counts:
                observed_codon_counts[codon] += 1
            else:
                observed_codon_counts[codon] = 1

        relative_adaptiveness_observed = {}
        for codon in observed_codon_counts:
            aa = self.dict_codon_to_aa[codon]
            codons_for_aa = self.dict_aa_to_codon[aa]
            max_count = max(observed_codon_counts.get(c, 0) for c in codons_for_aa)
            relative_adaptiveness_observed[codon] = observed_codon_counts[codon] / max_count if max_count > 0 else 0

        return relative_adaptiveness_observed

    def calculate_codon_frequencies(self, codon_seq, as_counts=False):
        codon_count = {}
        for codon in codon_seq:
            if codon in codon_count:
                codon_count[codon] += 1
            else:
                codon_count[codon] = 1
    
        if as_counts:
            return codon_count
        total_codons = len(codon_seq)
        return {codon: count / total_codons for codon, count in codon_count.items()}


    def calculate_gc_content_position(self, codon_seq, position=None):
        if position is None:
            gc_count = sum(codon.count('G') + codon.count('C') for codon in codon_seq)
            total_bases = len(codon_seq) * 3
            return (gc_count / total_bases) if total_bases > 0 else 0
        else:
            if position not in [1, 2, 3]:
                raise ValueError("Position must be None, 1, 2, or 3")
            gc_count = sum(codon[position-1] in ['G', 'C'] for codon in codon_seq)
            total_bases = len(codon_seq)
            return (gc_count / total_bases) if total_bases > 0 else 0

    def calculate_windowed_gc_content(self, codon_seq, window_size=10, step_size=1, position=None):
        sequence = ''.join(codon_seq)
        gc_content = []

        if position is None:
            for i in range(0, len(sequence) - window_size + 1, step_size):
                window = sequence[i:i + window_size]
                gc_count = sum(nuc in ['G', 'C'] for nuc in window)
                gc_content.append(gc_count / window_size)
        else:
            if position not in [1, 2, 3]:
                raise ValueError("Position must be 1, 2, or 3")
            start_index = position - 1
            for i in range(0, len(sequence) - window_size + 1, step_size):
                window = sequence[i:i + window_size]
                selected_positions = window[start_index::3]
                gc_count = sum(nuc in ['G', 'C'] for nuc in selected_positions)
                gc_content.append(gc_count / len(selected_positions) if len(selected_positions) > 0 else 0)
        
        return gc_content

    def calculate_purine_content_position(self, codon_seq, position=None):
        if position is None:
            ag_count = sum(codon.count('A') + codon.count('G') for codon in codon_seq)
            total_bases = len(codon_seq) * 3
            return (ag_count / total_bases) if total_bases > 0 else 0
        else:
            if position not in [1, 2, 3]:
                raise ValueError("Position must be None, 1, 2, or 3")
            ag_count = sum(codon[position-1] in ['A', 'G'] for codon in codon_seq)
            total_bases = len(codon_seq)
            return (ag_count / total_bases) if total_bases > 0 else 0

    def calculate_windowed_purine_content(self, codon_seq, window_size=10, step_size=1, position=None):
        sequence = ''.join(codon_seq)
        ag_content = []

        if position is None:
            for i in range(0, len(sequence) - window_size + 1, step_size):
                window = sequence[i:i + window_size]
                ag_count = sum(nuc in ['A', 'G'] for nuc in window)
                ag_content.append(ag_count / window_size)
        else:
            if position not in [1, 2, 3]:
                raise ValueError("Position must be 1, 2, or 3")
            start_index = position - 1
            for i in range(0, len(sequence) - window_size + 1, step_size):
                window = sequence[i:i + window_size]
                selected_positions = window[start_index::3]
                ag_count = sum(nuc in ['A', 'G'] for nuc in selected_positions)
                ag_content.append(ag_count / len(selected_positions) if len(selected_positions) > 0 else 0)
        
        return ag_content

    def calculate_uridine_content_position(self, codon_seq, position=None):
        if position is None:
            u_count = sum(codon.count('U') for codon in codon_seq)
            total_bases = len(codon_seq) * 3
            return (u_count / total_bases) if total_bases > 0 else 0
        else:
            if position not in [1, 2, 3]:
                raise ValueError("Position must be None, 1, 2, or 3")
            u_count = sum(codon[position-1] == 'U' for codon in codon_seq)
            total_bases = len(codon_seq)
            return (u_count / total_bases) if total_bases > 0 else 0

    def calculate_windowed_uridine_content(self, codon_seq, window_size=10, step_size=1, position=None):
        sequence = ''.join(codon_seq)
        u_content = []

        if position is None:
            for i in range(0, len(sequence) - window_size + 1, step_size):
                window = sequence[i:i + window_size]
                u_count = sum(nuc == 'U' for nuc in window)
                u_content.append(u_count / window_size)
        else:
            if position not in [1, 2, 3]:
                raise ValueError("Position must be 1, 2, or 3")
            start_index = position - 1
            for i in range(0, len(sequence) - window_size + 1, step_size):
                window = sequence[i:i + window_size]
                selected_positions = window[start_index::3]
                u_count = sum(nuc == 'U' for nuc in selected_positions)
                u_content.append(u_count / len(selected_positions) if len(selected_positions) > 0 else 0)
        
        return u_content

    
class AnalyzerBias(Analyzer):
    def calculate_cbi(self, codon_seq):
        if not hasattr(self, 'optimal_codon_dict'):
            self.initialize_optimal_codons()
            
        codon_count = Counter(codon_seq)
        total_codons = sum(codon_count.values())

        N_prt = sum(codon_count[codon] for codon in codon_count if codon in self.optimal_codon_dict.values())

        N_rand = sum(
            codon_count[codon] / self.amino_acid_degeneracy[self.dict_codon_to_aa[codon]]
            for codon in codon_count if codon in self.optimal_codon_dict.values()
        )

        if total_codons > 0:
            cbi = (N_prt - N_rand) / (total_codons - N_rand)
        else:
            cbi = 0

        return cbi

    def calculate_codon_pair_frequencies(self, codon_seq):
        pair_frequencies = Counter(zip(codon_seq, codon_seq[1:]))
        total_pairs = len(codon_seq) - 1
        for pair in pair_frequencies:
            pair_frequencies[pair] /= total_pairs
        return pair_frequencies
    
    def calculate_amino_acid_pair_frequencies(self, codon_seq, codon_to_amino_acid):
        amino_acids = [codon_to_amino_acid.get(codon) for codon in codon_seq if codon in codon_to_amino_acid]
        amino_acids = [aa for aa in amino_acids if aa is not None]
        amino_acid_pairs = Counter(zip(amino_acids, amino_acids[1:]))
        total_aa_pairs = len(amino_acids) - 1
        return {pair: count / total_aa_pairs for pair, count in amino_acid_pairs.items()} if total_aa_pairs > 0 else {}

    def calculate_cps(self, codon_seq):
        codon_frequencies = self.calculate_codon_frequencies(codon_seq)
        pair_frequencies = self.calculate_codon_pair_frequencies(codon_seq)
        amino_acid_pairs = self.calculate_amino_acid_pair_frequencies(codon_seq, self.dict_codon_to_aa)
    
        cps_scores = {}
        for (codon1, codon2), pair_freq in pair_frequencies.items():
            f_a = codon_frequencies.get(codon1, 0)
            f_b = codon_frequencies.get(codon2, 0)
            amino1, amino2 = self.dict_codon_to_aa.get(codon1), self.dict_codon_to_aa.get(codon2)
            f_xy = amino_acid_pairs.get((amino1, amino2), 0)
    
            expected_freq = f_a * f_b * f_xy if amino1 and amino2 else 0
            if expected_freq > 0:
                cps_scores[(codon1, codon2)] = np.log(pair_freq / expected_freq)
            else:
                cps_scores[(codon1, codon2)] = float('-inf')
        
        cps_values = list(cps_scores.values())
        cpb = sum(cps_values) / (len(cps_values) - 1) if cps_values else 0  # Handle case with no valid pairs

        return cps_scores, cpb

    def calculate_dcbs(self, codon_seq):
        nucleotide_frequencies = self.calculate_nucleotide_frequencies_position(codon_seq)
        codon_frequencies = self.calculate_codon_frequencies(codon_seq)
    
        if not codon_frequencies or not nucleotide_frequencies:
            return 0  # Return 0 if there's no valid data to process
    
        d_values = []
        for codon in codon_seq:
            if len(codon) == 3:  
                f_xyz = codon_frequencies.get(codon, 0)
                f_x = nucleotide_frequencies[0].get(codon[0], 0)
                f_y = nucleotide_frequencies[1].get(codon[1], 0)
                f_z = nucleotide_frequencies[2].get(codon[2], 0)
    
                if f_x > 0 and f_y > 0 and f_z > 0 and f_xyz > 0:
                    ratio1 = f_xyz / (f_x * f_y * f_z)
                    ratio2 = (f_x * f_y * f_z) / f_xyz
                    d = max(ratio1, ratio2)
                    d_values.append(d)
                else:
                    d_values.append(0.0)  
        dcbs_gene = sum(d_values) / len(codon_seq) if d_values else 0
        return dcbs_gene, d_values

    def calculate_nucleotide_contents(self, s, r):
        nucleotide_contents = {
            'A': (1 - s) * r,
            'U': (1 - s) * (1 - r),
            'G': s * r,
            'C': s * (1 - r)
        }
        return nucleotide_contents

    def calculate_expected_codon_usage(self, nucleotide_contents):
        codons = [a+b+c for a in 'AUGC' for b in 'AUGC' for c in 'AUGC']
        expected_usage = {}
        total = 0
    
        for codon in codons:
            x, y, z = codon
            expected_usage[codon] = nucleotide_contents[x] * nucleotide_contents[y] * nucleotide_contents[z]
            total += expected_usage[codon]
                
        for codon in expected_usage:
            expected_usage[codon] /= total
    
        return expected_usage

    def calculate_observed_codon_usage(self, codon_seq):
        observed_usage = {codon: 0 for codon in self.dict_codon_weights}
        total_codons = len(codon_seq)

        for codon in codon_seq:
            observed_usage[codon] += 1

        for codon in observed_usage:
            observed_usage[codon] /= total_codons

        return observed_usage
        
    def calculate_cdc(self, codon_seq):
        s = self.calculate_gc_content_position(codon_seq)
        r = self.calculate_purine_content_position(codon_seq)
        nucleotide_contents = self.calculate_nucleotide_contents(s, r)
        expected_usage = self.calculate_expected_codon_usage(nucleotide_contents)
        observed_usage = self.calculate_observed_codon_usage(codon_seq)
    
        dot_product = sum(expected_usage[codon] * observed_usage.get(codon, 0) for codon in expected_usage)
        mag_expected = np.sqrt(sum(val**2 for val in expected_usage.values()))
        mag_observed = np.sqrt(sum(val**2 for val in observed_usage.values()))
    
        cdc = 1 - dot_product / (mag_expected * mag_observed)
        return cdc

    def calculate_scuo(self, codon_seq):
        gene_codon_freq = self.calculate_codon_frequencies(codon_seq)
        total_codons = sum(gene_codon_freq.values())
        O_i_values, weights = [], []

        for aa, codons in self.dict_aa_to_codon.items():
            k = len(codons)
            if k > 1:
                H_max = math.log(k)
                p = [gene_codon_freq.get(codon, 0) / total_codons for codon in codons]
                H = -sum(pi * math.log(pi) for pi in p if pi > 0)
                O_i = 1 - H / H_max if H_max > 0 else 0
                O_i_values.append(O_i)
                weight = sum(gene_codon_freq.get(codon, 0) for codon in codons) / total_codons
                weights.append(weight)    
        if O_i_values and weights:
            O_w = sum(O * w for O, w in zip(O_i_values, weights))
            return O_w
        return 0

    def calculate_rcdi(self, codon_seq):
        gene_codon_freq = self.calculate_codon_frequencies(codon_seq)
        rcdi_values = []
        for codon, freq in gene_codon_freq.items():
            weight = self.dict_codon_weights.get(codon)
            if weight and weight > 0:
                rcdi_values.append(freq / weight)
        
        if rcdi_values:
            geometric_mean = np.prod(rcdi_values) ** (1 / len(rcdi_values))
        else:
            geometric_mean = 0
        
        return geometric_mean
        
    def calculate_enc(self, codon_seq):
        codon_frequencies = self.calculate_codon_frequencies(codon_seq)
        Fs = []

        for aa, codons in self.dict_aa_to_codon.items():
            if len(codons) > 1:
                n = sum(codon_frequencies.get(codon, 0) * len(codon_seq) for codon in codons)
                if n > 1:
                    pi_squared_sum = sum((codon_frequencies.get(codon, 0) / (n / len(codon_seq))) ** 2 for codon in codons)
                    F_AA = (n * pi_squared_sum - 1) / (n - 1)
                    Fs.append(F_AA)

        if Fs:
            F_values = {
                2: [F for F, codons in zip(Fs, self.dict_aa_to_codon.values()) if len(codons) == 2],
                3: [F for F, codons in zip(Fs, self.dict_aa_to_codon.values()) if len(codons) == 3],
                4: [F for F, codons in zip(Fs, self.dict_aa_to_codon.values()) if len(codons) == 4],
                6: [F for F, codons in zip(Fs, self.dict_aa_to_codon.values()) if len(codons) == 6]
            }

            Nc = 2
            if len(F_values[2]) > 0 and sum(F_values[2]) > 0:
                Nc += 9 / (sum(F_values[2]) / len(F_values[2]))
            if len(F_values[3]) > 0 and sum(F_values[3]) > 0:
                Nc += 1 / (sum(F_values[3]) / len(F_values[3]))
            if len(F_values[4]) > 0 and sum(F_values[4]) > 0:
                Nc += 5 / (sum(F_values[4]) / len(F_values[4]))
            if len(F_values[6]) > 0 and sum(F_values[6]) > 0:
                Nc += 3 / (sum(F_values[6]) / len(F_values[6]))

            return min(max(Nc, 20), 61)
        return float('inf')


    def calculate_rcbs(self, codon_seq):
        codon_frequencies = self.calculate_codon_frequencies(codon_seq)
        nucleotide_frequencies = self.calculate_nucleotide_frequencies_position(codon_seq)
        
        rcbs_values = {}
        for codon in codon_frequencies:
            x, y, z = codon[0], codon[1], codon[2]
            f_xyz = codon_frequencies[codon]
            f1_x = nucleotide_frequencies[0][x]
            f2_y = nucleotide_frequencies[1][y]
            f3_z = nucleotide_frequencies[2][z]
            
            expected_freq = f1_x * f2_y * f3_z
            if expected_freq != 0:
                d_xyz = (f_xyz - expected_freq) / expected_freq
            else:
                d_xyz = 0
            
            rcbs_values[codon] = d_xyz
        
        rcbs_gene = np.exp(np.mean([np.log(1 + d) for d in rcbs_values.values()])) - 1
        
        return rcbs_gene, rcbs_values


    def calculate_p2(self, codon_seq):

        codon_counts = Counter(codon_seq)
        
        WWC = ['AAC', 'UUC', 'AUC', 'UAC']
        SSU = ['CCU', 'GGU', 'CGU', 'GCU']
        WWY = ['AAC', 'UUC', 'AUC', 'UAC',
              'AAU', 'UUU', 'AUU', 'UAU']
        SSY = ['CCC', 'GGC', 'CGC', 'GCC',
              'CCU', 'GGU', 'CGU', 'GCU']
        
        # Calculate frequencies of each category
        WWC_count = sum(codon_counts.get(codon, 0) for codon in WWC)
        SSU_count = sum(codon_counts.get(codon, 0) for codon in SSU)
        WWY_count = sum(codon_counts.get(codon, 0) for codon in WWY)
        SSY_count = sum(codon_counts.get(codon, 0) for codon in SSY)
        
        # Calculate P2 index
        P2 = (WWC_count + SSU_count) / (WWY_count + SSY_count) if (WWY_count + SSY_count) > 0 else 0
        return P2

class AnalyzerAdaptation(Analyzer):
    def calculate_cai(self, codon_seq: str):
        cai = 1.0
        for codon in codon_seq:
            if codon in self.relative_adaptiveness:
                cai *= self.relative_adaptiveness[codon]
            else:
                cai *= 1e-3  # Assuming a very low value for non-standard codons
        cai **= (1 / len(codon_seq))
        return cai

    def calculate_windowed_cai(self, codon_seq, window_size=10, step_size=1):
        windowed_cai_values = []
    
        for i in range(0, len(codon_seq) - window_size + 1, step_size):
            window = codon_seq[i:i + window_size]
            cai_value = self.calculate_cai(window)
            windowed_cai_values.append(cai_value)
    
        return windowed_cai_values

    def calculate_rscu(self, codon_seq):
        codon_count = {codon: 0 for codon in self.dict_codon_weights.keys()}
        aa_counts = {aa: 0 for aa in self.dict_aa_to_codon.keys()}
        
        for codon in codon_seq:
            if codon in codon_count:
                codon_count[codon] += 1
        
        for codon, count in codon_count.items():
            aa = self.dict_codon_to_aa[codon]
            aa_counts[aa] += count
        
        rscu_values = {}
        for aa, codons in self.dict_aa_to_codon.items():
            total = sum(codon_count[codon] for codon in codons)
            if total > 0:
                for codon in codons:
                    rscu_values[codon] = codon_count[codon] / (total / len(codons))
            else:
                for codon in codons:
                    rscu_values[codon] = 0
        return rscu_values

    def calculate_fop(self, codon_seq):
        if not hasattr(self, 'optimal_codon_dict'):
            self.initialize_optimal_codons()
        total_codons = len(codon_seq)
        if total_codons == 0:
            return 0

        optimal_codon_count = sum(1 for codon in codon_seq if codon in self.optimal_codon_dict.values())
        fop = optimal_codon_count / total_codons
        return fop

    def calculate_rca(self, codon_seq):
        nucleotide_freq = self.calculate_nucleotide_frequencies_position(codon_seq)
        
        RCA_values = []
        
        for codon in codon_seq:
            x, y, z = codon[0], codon[1], codon[2]
            
            f_xyz = nucleotide_freq[0][x] * nucleotide_freq[1][y] * nucleotide_freq[2][z]
            
            f_ref = (
                self.reference_nucleotide_frequencies[0].get(x, 1e-6) *
                self.reference_nucleotide_frequencies[1].get(y, 1e-6) *
                self.reference_nucleotide_frequencies[2].get(z, 1e-6)
            )
            
            RCA_codon = f_xyz / f_ref if f_ref != 0 else 1e-6
            RCA_values.append(RCA_codon)
        
        if RCA_values:
            geometric_mean = np.exp(np.mean(np.log(RCA_values)))
        else:
            geometric_mean = 0
        
        return geometric_mean, RCA_values
        
    def kl_divergence(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    
    def calculate_cufs(self, p_codon, q_codon):
        p_freq = self.calculate_codon_frequencies(p_codon)
        q_freq = self.calculate_codon_frequencies(q_codon)
        
        all_codons = set(p_freq.keys()).union(q_freq.keys())

        p_vector = np.array([p_freq.get(codon, 0) for codon in all_codons])
        q_vector = np.array([q_freq.get(codon, 0) for codon in all_codons])
    
        p_vector /= np.sum(p_vector)
        q_vector /= np.sum(q_vector)
    
        m_vector = 0.5 * (p_vector + q_vector)
    
        dkl_p_m = self.kl_divergence(p_vector, m_vector)
        dkl_q_m = self.kl_divergence(q_vector, m_vector)
        d_es = np.sqrt(dkl_p_m + dkl_q_m)
    
        return d_es
    
class AnalyzerStability(Analyzer):
    def calculate_gc_content_position(self, codon_seq, position=None):
        if position is None:
            gc_count = sum(codon.count('G') + codon.count('C') for codon in codon_seq)
            total_bases = len(codon_seq) * 3
            return (gc_count / total_bases) if total_bases > 0 else 0
        else:
            if position not in [1, 2, 3]:
                raise ValueError("Position must be None, 1, 2, or 3")
            gc_count = sum(codon[position-1] in ['G', 'C'] for codon in codon_seq)
            total_bases = len(codon_seq)
            return (gc_count / total_bases) if total_bases > 0 else 0

    def calculate_windowed_gc_content(self, codon_seq, window_size=10, step_size=1, position=None):
        sequence = ''.join(codon_seq)
        gc_content = []

        if position is None:
            for i in range(0, len(sequence) - window_size + 1, step_size):
                window = sequence[i:i + window_size]
                gc_count = sum(nuc in ['G', 'C'] for nuc in window)
                gc_content.append(gc_count / window_size)
        else:
            if position not in [1, 2, 3]:
                raise ValueError("Position must be 1, 2, or 3")
            start_index = position - 1
            for i in range(0, len(sequence) - window_size + 1, step_size):
                window = sequence[i:i + window_size]
                selected_positions = window[start_index::3]
                gc_count = sum(nuc in ['G', 'C'] for nuc in selected_positions)
                gc_content.append(gc_count / len(selected_positions) if len(selected_positions) > 0 else 0)
        
        return gc_content

    def calculate_mfe(self, sequence):
        # Use RNAfold or similar tool to calculate Minimum Free Energy (MFE) for RNA sequence
        mfe, _ = RNA.fold(sequence)
        return mfe

    def calculate_pf(self, sequence):
        # Use RNAfold or similar tool to calculate Partition Function (PF) for RNA sequence
        md = RNA.md()
        md.uniq_ML = 1
        fc = RNA.fold_compound(sequence, md)
        (ss, mfe) = fc.mfe()
        (pf, _, _) = fc.pf()
        return pf


    def calculate_codon_frequencies(self, codon_seq, as_counts=False):
        codon_count = {}
        for codon in codon_seq:
            if codon in codon_count:
                codon_count[codon] += 1
            else:
                codon_count[codon] = 1
    
        if as_counts:
            return codon_count
        total_codons = len(codon_seq)
        return {codon: count / total_codons for codon, count in codon_count.items()}

    def calculate_ew(self, codon_seq):
        gene_codon_freq = self.calculate_codon_frequencies(codon_seq)
        total_codons = sum(gene_codon_freq.values())
    
        E_values, weights = [], []
    
        for aa, codons in self.dict_aa_to_codon.items():
            k = len(codons)
            if k > 1:
                p = [gene_codon_freq.get(codon, 0) / total_codons for codon in codons]
                H = -sum(pi * math.log(pi) for pi in p if pi > 0)
                H_max = math.log(k)
                E = H / H_max if H_max > 0 else 0
                E_values.append(E)
                weight = sum(gene_codon_freq.get(codon, 0) for codon in codons) / total_codons
                weights.append(weight)
        
        Ew = sum(E * w for E, w in zip(E_values, weights)) if weights else 0
        return Ew



class AnalyzerRNAFold:
    def __init__(self, rna_seq, noLP=True):
        self.md = RNA.md()
        self.md.noLP = noLP
        self.seq = rna_seq
        self.fc = RNA.fold_compound(rna_seq, self.md)

    def calculate_mfe(self):
        self.mfe, self.mfe_structure = self.fc.mfe()
        return self.mfe, self.mfe_structure
    
    def calculate_pf(self):
        self.pf, self.pf_structure = self.fc.pf()
        return self.pf, self.pf_structure
    
    def calculate_centroid(self):
        self.cetroid, self.centroid_structure = self.fc.centroid()
        return self.cetroid, self.centroid_structure
    
    def calculate_entropy(self):
        self.entropy, self.entropy_structure = self.fc.entropy()
        return self.entropy, self.entropy_structure
    
    def preprocessing_mountains(self, structure):
        level = 0
        heights = []
        stack = []

        for char in structure:
            if char == '(':
                level += 1
                stack.append(level)
            elif char == ')':
                level = stack.pop() if stack else 0

            heights.append(level)

        return np.array(heights)

    def plot_mountain(self):
        positions = np.arange(1, len(self.mfe_structure)+1)
        self.mfe_mountain = self.preprocessing_mountains(self.mfe_structure)
        self.pf_moutain = self.preprocessing_mountains(self.pf_structure)
        self.centroid_moutain = self.preprocessing_mountains(self.centroid_structure)


        

