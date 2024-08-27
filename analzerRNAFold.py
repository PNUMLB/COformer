from collections import Counter, defaultdict
import warnings
import numpy as np
import pandas as pd
import math
import RNA

warnings.filterwarnings(action='ignore')

class AnalyzerRNAFold:
    def __init__(self, rna_seq, noLP=True):
        self.md = RNA.md()
        self.md.noLP = noLP
        self.seq = rna_seq
        self.fc = RNA.fold_compound(rna_seq, self.md)

    def calculate_mfe(self):
        self.mfe_structure, self.mfe = self.fc.mfe()
        return self.mfe, self.mfe_structure
    
    def calculate_pf(self):
        self.pf_structure, self.pf = self.fc.pf()
        return self.pf, self.pf_structure
    
    def calculate_centroid(self):
        self.centroid_structure, self.cetroid = self.fc.centroid()
        return self.cetroid, self.centroid_structure
    
    def calculate_entropy(self):
        entropy = self.fc.positional_entropy()
        self.entropy, self.entropy_structure = entropy[0], entropy[1:]
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

    def get_values_mountain(self):
        self.mfe_mountain = self.preprocessing_mountains(self.mfe_structure)
        self.pf_moutain = self.preprocessing_mountains(self.pf_structure)
        self.centroid_moutain = self.preprocessing_mountains(self.centroid_structure)

