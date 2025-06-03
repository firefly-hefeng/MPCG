#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPCG-Codon: Multi-Modal Physics-Constrained Guided Codon Optimization
Full model implementation (with five-species prior data)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
from Bio.Data import CodonTable

def get_mask_value(dtype: torch.dtype) -> float:
    """
    Get a mask value safe for the current dtype.
    
    Args:
        dtype: Tensor data type.
    
    Returns:
        Safe mask value.
    """
    if dtype in [torch.float16, torch.bfloat16]:
        # float16/bfloat16 range is approximately ±65504
        return -65000.0
    else:
        # float32/float64 can use a larger value
        return -1e9

# ==================== Config Class ====================
@dataclass
class MPCGConfig:
    """MPCG-Codon model configuration."""
    # Basic parameters
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 2048
    
    # Sparse attention parameters
    local_window: int = 64
    global_ratio: float = 0.1
    
    # Physics constraint parameters
    lambda_energy: float = 0.1
    lambda_pause: float = 0.05
    lambda_gc: float = 0.01
    
    # Multi-modal parameters
    aux_dim: int = 64
    structure_dim: int = 128
    dynamics_dim: int = 64
    
    # Other parameters
    use_neural_folder: bool = True


# ==================== Five-Species Codon Usage Data ====================
class FiveSpeciesCodonData:
    """
    Codon usage frequency, RSCU, and tRNA abundance data for five species.
    Data source: Kazusa Codon Usage Database and related literature.
    """
    
    def __init__(self):
        self.genetic_code = CodonTable.unambiguous_dna_by_name["Standard"]
        
        # Initialize data for five species
        self._initialize_codon_frequencies()
        self._initialize_trna_abundances()
        
        # Compute RSCU and CAI weights
        self.species_rscu = {}
        self.species_cai_weights = {}
        
        for species in self.species_list:
            self.species_rscu[species] = self._calculate_rscu(
                self.codon_frequencies[species]
            )
            self.species_cai_weights[species] = self._calculate_cai_weights(
                self.codon_frequencies[species]
            )
    
    def _initialize_codon_frequencies(self):
        """Initialize codon frequencies for five species."""
        
        # Species list
        self.species_list = [
            'Homo sapiens',
            'Mus musculus',
            'Escherichia coli',
            'Saccharomyces cerevisiae',
            'Pichia angusta'
        ]
        
        # Codon frequency dictionary
        self.codon_frequencies = {}
        
        # 1. Homo sapiens (Human) - from highly expressed genes
        self.codon_frequencies['Homo sapiens'] = {
            # Phe
            'TTT': 0.45, 'TTC': 0.55,
            # Leu
            'TTA': 0.07, 'TTG': 0.13, 'CTT': 0.13, 'CTC': 0.20, 'CTA': 0.07, 'CTG': 0.41,
            # Ile
            'ATT': 0.36, 'ATC': 0.48, 'ATA': 0.16,
            # Met
            'ATG': 1.00,
            # Val
            'GTT': 0.18, 'GTC': 0.24, 'GTA': 0.11, 'GTG': 0.47,
            # Ser
            'TCT': 0.18, 'TCC': 0.22, 'TCA': 0.15, 'TCG': 0.05, 'AGT': 0.15, 'AGC': 0.24,
            # Pro
            'CCT': 0.28, 'CCC': 0.33, 'CCA': 0.27, 'CCG': 0.11,
            # Thr
            'ACT': 0.24, 'ACC': 0.36, 'ACA': 0.28, 'ACG': 0.12,
            # Ala
            'GCT': 0.26, 'GCC': 0.40, 'GCA': 0.23, 'GCG': 0.11,
            # Tyr
            'TAT': 0.43, 'TAC': 0.57,
            # His
            'CAT': 0.41, 'CAC': 0.59,
            # Gln
            'CAA': 0.25, 'CAG': 0.75,
            # Asn
            'AAT': 0.46, 'AAC': 0.54,
            # Lys
            'AAA': 0.42, 'AAG': 0.58,
            # Asp
            'GAT': 0.46, 'GAC': 0.54,
            # Glu
            'GAA': 0.42, 'GAG': 0.58,
            # Cys
            'TGT': 0.45, 'TGC': 0.55,
            # Trp
            'TGG': 1.00,
            # Arg
            'CGT': 0.08, 'CGC': 0.19, 'CGA': 0.11, 'CGG': 0.21, 'AGA': 0.20, 'AGG': 0.20,
            # Gly
            'GGT': 0.16, 'GGC': 0.34, 'GGA': 0.25, 'GGG': 0.25,
            # Stop
            'TAA': 0.28, 'TAG': 0.20, 'TGA': 0.52,
        }
        
        # 2. Mus musculus (Mouse)
        self.codon_frequencies['Mus musculus'] = {
            'TTT': 0.43, 'TTC': 0.57,
            'TTA': 0.06, 'TTG': 0.12, 'CTT': 0.12, 'CTC': 0.20, 'CTA': 0.07, 'CTG': 0.43,
            'ATT': 0.33, 'ATC': 0.51, 'ATA': 0.15,
            'ATG': 1.00,
            'GTT': 0.17, 'GTC': 0.24, 'GTA': 0.11, 'GTG': 0.48,
            'TCT': 0.18, 'TCC': 0.23, 'TCA': 0.14, 'TCG': 0.05, 'AGT': 0.14, 'AGC': 0.26,
            'CCT': 0.29, 'CCC': 0.34, 'CCA': 0.26, 'CCG': 0.11,
            'ACT': 0.23, 'ACC': 0.38, 'ACA': 0.27, 'ACG': 0.12,
            'GCT': 0.26, 'GCC': 0.41, 'GCA': 0.22, 'GCG': 0.11,
            'TAT': 0.42, 'TAC': 0.58,
            'CAT': 0.40, 'CAC': 0.60,
            'CAA': 0.24, 'CAG': 0.76,
            'AAT': 0.44, 'AAC': 0.56,
            'AAA': 0.40, 'AAG': 0.60,
            'GAT': 0.44, 'GAC': 0.56,
            'GAA': 0.40, 'GAG': 0.60,
            'TGT': 0.43, 'TGC': 0.57,
            'TGG': 1.00,
            'CGT': 0.09, 'CGC': 0.20, 'CGA': 0.11, 'CGG': 0.21, 'AGA': 0.19, 'AGG': 0.20,
            'GGT': 0.17, 'GGC': 0.35, 'GGA': 0.24, 'GGG': 0.24,
            'TAA': 0.27, 'TAG': 0.19, 'TGA': 0.54,
        }
        
        # 3. Escherichia coli K12
        self.codon_frequencies['Escherichia coli'] = {
            'TTT': 0.58, 'TTC': 0.42,
            'TTA': 0.14, 'TTG': 0.13, 'CTT': 0.12, 'CTC': 0.10, 'CTA': 0.04, 'CTG': 0.47,
            'ATT': 0.51, 'ATC': 0.42, 'ATA': 0.07,
            'ATG': 1.00,
            'GTT': 0.28, 'GTC': 0.20, 'GTA': 0.17, 'GTG': 0.35,
            'TCT': 0.17, 'TCC': 0.15, 'TCA': 0.14, 'TCG': 0.14, 'AGT': 0.16, 'AGC': 0.28,
            'CCT': 0.18, 'CCC': 0.12, 'CCA': 0.20, 'CCG': 0.49,
            'ACT': 0.19, 'ACC': 0.40, 'ACA': 0.17, 'ACG': 0.25,
            'GCT': 0.18, 'GCC': 0.26, 'GCA': 0.23, 'GCG': 0.33,
            'TAT': 0.59, 'TAC': 0.41,
            'CAT': 0.57, 'CAC': 0.43,
            'CAA': 0.34, 'CAG': 0.66,
            'AAT': 0.49, 'AAC': 0.51,
            'AAA': 0.74, 'AAG': 0.26,
            'GAT': 0.63, 'GAC': 0.37,
            'GAA': 0.68, 'GAG': 0.32,
            'TGT': 0.46, 'TGC': 0.54,
            'TGG': 1.00,
            'CGT': 0.36, 'CGC': 0.36, 'CGA': 0.07, 'CGG': 0.10, 'AGA': 0.07, 'AGG': 0.04,
            'GGT': 0.35, 'GGC': 0.37, 'GGA': 0.13, 'GGG': 0.15,
            'TAA': 0.61, 'TAG': 0.09, 'TGA': 0.30,
        }
        
        # 4. Saccharomyces cerevisiae (Yeast)
        self.codon_frequencies['Saccharomyces cerevisiae'] = {
            'TTT': 0.59, 'TTC': 0.41,
            'TTA': 0.28, 'TTG': 0.29, 'CTT': 0.13, 'CTC': 0.06, 'CTA': 0.14, 'CTG': 0.11,
            'ATT': 0.46, 'ATC': 0.26, 'ATA': 0.27,
            'ATG': 1.00,
            'GTT': 0.39, 'GTC': 0.21, 'GTA': 0.21, 'GTG': 0.19,
            'TCT': 0.26, 'TCC': 0.16, 'TCA': 0.21, 'TCG': 0.10, 'AGT': 0.16, 'AGC': 0.11,
            'CCT': 0.31, 'CCC': 0.15, 'CCA': 0.42, 'CCG': 0.12,
            'ACT': 0.35, 'ACC': 0.22, 'ACA': 0.30, 'ACG': 0.14,
            'GCT': 0.38, 'GCC': 0.22, 'GCA': 0.29, 'GCG': 0.11,
            'TAT': 0.56, 'TAC': 0.44,
            'CAT': 0.64, 'CAC': 0.36,
            'CAA': 0.69, 'CAG': 0.31,
            'AAT': 0.59, 'AAC': 0.41,
            'AAA': 0.58, 'AAG': 0.42,
            'GAT': 0.65, 'GAC': 0.35,
            'GAA': 0.70, 'GAG': 0.30,
            'TGT': 0.63, 'TGC': 0.37,
            'TGG': 1.00,
            'CGT': 0.14, 'CGC': 0.06, 'CGA': 0.07, 'CGG': 0.04, 'AGA': 0.48, 'AGG': 0.21,
            'GGT': 0.47, 'GGC': 0.19, 'GGA': 0.22, 'GGG': 0.12,
            'TAA': 0.47, 'TAG': 0.23, 'TGA': 0.30,
        }
        
        # 5. Pichia angusta (similar to Pichia pastoris, methylotrophic yeast)
        self.codon_frequencies['Pichia angusta'] = {
            'TTT': 0.56, 'TTC': 0.44,
            'TTA': 0.24, 'TTG': 0.30, 'CTT': 0.15, 'CTC': 0.08, 'CTA': 0.13, 'CTG': 0.10,
            'ATT': 0.44, 'ATC': 0.28, 'ATA': 0.28,
            'ATG': 1.00,
            'GTT': 0.38, 'GTC': 0.23, 'GTA': 0.20, 'GTG': 0.19,
            'TCT': 0.28, 'TCC': 0.18, 'TCA': 0.20, 'TCG': 0.09, 'AGT': 0.15, 'AGC': 0.10,
            'CCT': 0.33, 'CCC': 0.16, 'CCA': 0.40, 'CCG': 0.11,
            'ACT': 0.36, 'ACC': 0.24, 'ACA': 0.28, 'ACG': 0.12,
            'GCT': 0.40, 'GCC': 0.24, 'GCA': 0.27, 'GCG': 0.09,
            'TAT': 0.58, 'TAC': 0.42,
            'CAT': 0.62, 'CAC': 0.38,
            'CAA': 0.67, 'CAG': 0.33,
            'AAT': 0.57, 'AAC': 0.43,
            'AAA': 0.60, 'AAG': 0.40,
            'GAT': 0.63, 'GAC': 0.37,
            'GAA': 0.68, 'GAG': 0.32,
            'TGT': 0.61, 'TGC': 0.39,
            'TGG': 1.00,
            'CGT': 0.16, 'CGC': 0.08, 'CGA': 0.08, 'CGG': 0.05, 'AGA': 0.45, 'AGG': 0.18,
            'GGT': 0.45, 'GGC': 0.21, 'GGA': 0.23, 'GGG': 0.11,
            'TAA': 0.45, 'TAG': 0.25, 'TGA': 0.30,
        }
    
    def _initialize_trna_abundances(self):
        """Initialize tRNA abundance data (based on tRNA gene copy number and expression data)."""
        
        self.trna_abundances = {}
        
        # 1. Human tRNA abundance
        self.trna_abundances['Homo sapiens'] = {
            'TTT': 0.8, 'TTC': 1.2, 'TTA': 0.3, 'TTG': 0.6,
            'CTT': 0.5, 'CTC': 0.9, 'CTA': 0.3, 'CTG': 1.4,
            'ATT': 0.9, 'ATC': 1.3, 'ATA': 0.5, 'ATG': 1.6,
            'GTT': 0.6, 'GTC': 0.8, 'GTA': 0.4, 'GTG': 1.3,
            'TCT': 0.7, 'TCC': 0.9, 'TCA': 0.6, 'TCG': 0.3, 'AGT': 0.6, 'AGC': 0.9,
            'CCT': 0.8, 'CCC': 1.0, 'CCA': 0.7, 'CCG': 0.4,
            'ACT': 0.7, 'ACC': 1.1, 'ACA': 0.8, 'ACG': 0.4,
            'GCT': 0.8, 'GCC': 1.2, 'GCA': 0.7, 'GCG': 0.4,
            'TAT': 0.9, 'TAC': 1.1, 'CAT': 0.8, 'CAC': 1.1,
            'CAA': 0.6, 'CAG': 1.3, 'AAT': 0.9, 'AAC': 1.1,
            'AAA': 0.9, 'AAG': 1.2, 'GAT': 0.9, 'GAC': 1.1,
            'GAA': 0.9, 'GAG': 1.2, 'TGT': 0.8, 'TGC': 1.1,
            'TGG': 1.0, 'CGT': 0.4, 'CGC': 0.7, 'CGA': 0.4,
            'CGG': 0.6, 'AGA': 0.7, 'AGG': 0.6, 'GGT': 0.6,
            'GGC': 1.1, 'GGA': 0.8, 'GGG': 0.7,
            'TAA': 0.0, 'TAG': 0.0, 'TGA': 0.0,
        }
        
        # 2. Mouse tRNA abundance
        self.trna_abundances['Mus musculus'] = {
            'TTT': 0.8, 'TTC': 1.3, 'TTA': 0.3, 'TTG': 0.5,
            'CTT': 0.5, 'CTC': 0.9, 'CTA': 0.3, 'CTG': 1.5,
            'ATT': 0.8, 'ATC': 1.4, 'ATA': 0.5, 'ATG': 1.7,
            'GTT': 0.6, 'GTC': 0.8, 'GTA': 0.4, 'GTG': 1.4,
            'TCT': 0.7, 'TCC': 1.0, 'TCA': 0.6, 'TCG': 0.3, 'AGT': 0.6, 'AGC': 1.0,
            'CCT': 0.8, 'CCC': 1.1, 'CCA': 0.7, 'CCG': 0.4,
            'ACT': 0.7, 'ACC': 1.2, 'ACA': 0.8, 'ACG': 0.4,
            'GCT': 0.8, 'GCC': 1.3, 'GCA': 0.7, 'GCG': 0.4,
            'TAT': 0.9, 'TAC': 1.2, 'CAT': 0.8, 'CAC': 1.2,
            'CAA': 0.6, 'CAG': 1.4, 'AAT': 0.9, 'AAC': 1.2,
            'AAA': 0.8, 'AAG': 1.3, 'GAT': 0.9, 'GAC': 1.2,
            'GAA': 0.8, 'GAG': 1.3, 'TGT': 0.8, 'TGC': 1.2,
            'TGG': 1.0, 'CGT': 0.4, 'CGC': 0.8, 'CGA': 0.4,
            'CGG': 0.7, 'AGA': 0.7, 'AGG': 0.6, 'GGT': 0.6,
            'GGC': 1.2, 'GGA': 0.8, 'GGG': 0.7,
            'TAA': 0.0, 'TAG': 0.0, 'TGA': 0.0,
        }
        
        # 3. E. coli tRNA abundance
        self.trna_abundances['Escherichia coli'] = {
            'TTT': 1.0, 'TTC': 1.2, 'TTA': 0.3, 'TTG': 0.8,
            'CTT': 0.6, 'CTC': 0.5, 'CTA': 0.2, 'CTG': 1.5,
            'ATT': 1.3, 'ATC': 1.1, 'ATA': 0.4, 'ATG': 1.8,
            'GTT': 0.9, 'GTC': 0.7, 'GTA': 0.5, 'GTG': 1.2,
            'TCT': 0.8, 'TCC': 0.7, 'TCA': 0.6, 'TCG': 0.5, 'AGT': 0.7, 'AGC': 0.9,
            'CCT': 0.7, 'CCC': 0.5, 'CCA': 0.8, 'CCG': 1.3,
            'ACT': 0.8, 'ACC': 1.2, 'ACA': 0.7, 'ACG': 0.9,
            'GCT': 0.9, 'GCC': 1.1, 'GCA': 0.8, 'GCG': 1.0,
            'TAT': 1.1, 'TAC': 0.9, 'CAT': 1.0, 'CAC': 0.8,
            'CAA': 0.7, 'CAG': 1.3, 'AAT': 1.0, 'AAC': 0.9,
            'AAA': 1.4, 'AAG': 0.8, 'GAT': 1.2, 'GAC': 0.9,
            'GAA': 1.3, 'GAG': 0.9, 'TGT': 0.8, 'TGC': 0.9,
            'TGG': 1.0, 'CGT': 1.1, 'CGC': 1.0, 'CGA': 0.3,
            'CGG': 0.4, 'AGA': 0.3, 'AGG': 0.2, 'GGT': 1.1,
            'GGC': 1.2, 'GGA': 0.6, 'GGG': 0.7,
            'TAA': 0.0, 'TAG': 0.0, 'TGA': 0.0,
        }
        
        # 4. S. cerevisiae tRNA abundance
        self.trna_abundances['Saccharomyces cerevisiae'] = {
            'TTT': 1.2, 'TTC': 0.9, 'TTA': 0.8, 'TTG': 0.9,
            'CTT': 0.6, 'CTC': 0.3, 'CTA': 0.5, 'CTG': 0.5,
            'ATT': 1.1, 'ATC': 0.7, 'ATA': 0.7, 'ATG': 1.5,
            'GTT': 1.1, 'GTC': 0.7, 'GTA': 0.6, 'GTG': 0.6,
            'TCT': 0.9, 'TCC': 0.6, 'TCA': 0.7, 'TCG': 0.4, 'AGT': 0.6, 'AGC': 0.5,
            'CCT': 0.9, 'CCC': 0.5, 'CCA': 1.1, 'CCG': 0.4,
            'ACT': 1.0, 'ACC': 0.7, 'ACA': 0.8, 'ACG': 0.5,
            'GCT': 1.0, 'GCC': 0.7, 'GCA': 0.8, 'GCG': 0.4,
            'TAT': 1.1, 'TAC': 0.8, 'CAT': 1.2, 'CAC': 0.7,
            'CAA': 1.2, 'CAG': 0.6, 'AAT': 1.1, 'AAC': 0.8,
            'AAA': 1.1, 'AAG': 0.8, 'GAT': 1.2, 'GAC': 0.7,
            'GAA': 1.3, 'GAG': 0.7, 'TGT': 1.1, 'TGC': 0.7,
            'TGG': 1.0, 'CGT': 0.6, 'CGC': 0.3, 'CGA': 0.3,
            'CGG': 0.2, 'AGA': 1.1, 'AGG': 0.6, 'GGT': 1.2,
            'GGC': 0.7, 'GGA': 0.7, 'GGG': 0.5,
            'TAA': 0.0, 'TAG': 0.0, 'TGA': 0.0,
        }
        
        # 5. Pichia angusta tRNA abundance
        self.trna_abundances['Pichia angusta'] = {
            'TTT': 1.1, 'TTC': 0.9, 'TTA': 0.7, 'TTG': 0.9,
            'CTT': 0.7, 'CTC': 0.4, 'CTA': 0.5, 'CTG': 0.5,
            'ATT': 1.0, 'ATC': 0.7, 'ATA': 0.7, 'ATG': 1.5,
            'GTT': 1.0, 'GTC': 0.7, 'GTA': 0.6, 'GTG': 0.7,
            'TCT': 0.9, 'TCC': 0.7, 'TCA': 0.7, 'TCG': 0.4, 'AGT': 0.6, 'AGC': 0.5,
            'CCT': 0.9, 'CCC': 0.6, 'CCA': 1.0, 'CCG': 0.4,
            'ACT': 1.0, 'ACC': 0.8, 'ACA': 0.8, 'ACG': 0.5,
            'GCT': 1.1, 'GCC': 0.8, 'GCA': 0.8, 'GCG': 0.4,
            'TAT': 1.1, 'TAC': 0.9, 'CAT': 1.1, 'CAC': 0.8,
            'CAA': 1.1, 'CAG': 0.7, 'AAT': 1.0, 'AAC': 0.9,
            'AAA': 1.1, 'AAG': 0.8, 'GAT': 1.1, 'GAC': 0.8,
            'GAA': 1.2, 'GAG': 0.8, 'TGT': 1.0, 'TGC': 0.8,
            'TGG': 1.0, 'CGT': 0.7, 'CGC': 0.4, 'CGA': 0.4,
            'CGG': 0.3, 'AGA': 1.0, 'AGG': 0.6, 'GGT': 1.1,
            'GGC': 0.8, 'GGA': 0.7, 'GGG': 0.5,
            'TAA': 0.0, 'TAG': 0.0, 'TGA': 0.0,
        }
    
    def _calculate_rscu(self, codon_freq: Dict[str, float]) -> Dict[str, float]:
        """Calculate RSCU values."""
        rscu = {}
        
        # Group by amino acid
        aa_codons = defaultdict(list)
        for codon, freq in codon_freq.items():
            if codon in self.genetic_code.forward_table:
                aa = self.genetic_code.forward_table[codon]
            elif codon in self.genetic_code.stop_codons:
                aa = '*'
            else:
                continue
            aa_codons[aa].append((codon, freq))
        
        # Calculate RSCU
        for aa, codon_list in aa_codons.items():
            n_synonymous = len(codon_list)
            total_freq = sum(freq for _, freq in codon_list)
            
            for codon, freq in codon_list:
                if total_freq > 0:
                    rscu[codon] = (freq * n_synonymous) / total_freq
                else:
                    rscu[codon] = 1.0
        
        return rscu
    
    def _calculate_cai_weights(self, codon_freq: Dict[str, float]) -> Dict[str, float]:
        """Calculate CAI weights."""
        weights = {}
        
        # Group by amino acid
        aa_codons = defaultdict(list)
        for codon, freq in codon_freq.items():
            if codon in self.genetic_code.forward_table:
                aa = self.genetic_code.forward_table[codon]
            elif codon in self.genetic_code.stop_codons:
                aa = '*'
            else:
                continue
            aa_codons[aa].append((codon, freq))
        
        # Calculate weights
        for aa, codon_list in aa_codons.items():
            max_freq = max(freq for _, freq in codon_list)
            
            for codon, freq in codon_list:
                if max_freq > 0:
                    weights[codon] = freq / max_freq
                else:
                    weights[codon] = 1.0
        
        return weights
    
    def get_species_index(self, species_name: str) -> int:
        """Get species index."""
        try:
            return self.species_list.index(species_name)
        except ValueError:
            return 0
    
    def get_codon_freq(self, species: str) -> Dict[str, float]:
        return self.codon_frequencies.get(species, {})
    
    def get_rscu(self, species: str) -> Dict[str, float]:
        return self.species_rscu.get(species, {})
    
    def get_cai_weights(self, species: str) -> Dict[str, float]:
        return self.species_cai_weights.get(species, {})
    
    def get_trna_abundance(self, species: str) -> Dict[str, float]:
        return self.trna_abundances.get(species, {})


# ==================== Neural RNA Folder ====================
class NeuralRNAFolder(nn.Module):
    """Neural network RNA folding predictor."""
    
    def __init__(self, d_model: int = 512, n_layers: int = 4):
        super().__init__()
        self.d_model = d_model
        
        # Nucleotide embedding (0=pad, 1=A, 2=C, 3=G, 4=U/T)
        self.nucleotide_embed = nn.Embedding(5, d_model, padding_idx=0)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # MFE prediction head
        self.mfe_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # Structure prediction head
        self.struct_head = nn.Linear(d_model, d_model)
    
    def forward(self, nucleotide_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            nucleotide_seq: [B, L] nucleotide sequence
        Returns:
            mfe: [B] minimum folding energy
            pairing_matrix: [B, L, L] pairing probability matrix
        """
        # Embedding
        x = self.nucleotide_embed(nucleotide_seq)  # [B, L, D]
        
        # Padding mask
        mask = (nucleotide_seq == 0)
        
        

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # MFE prediction
        mfe = self.mfe_head(x.mean(dim=1)).squeeze(-1)  # [B]
        
        # Pairing probability matrix
        struct_repr = self.struct_head(x)  # [B, L, D]
        pairing_matrix = torch.matmul(struct_repr, struct_repr.transpose(1, 2))
        pairing_matrix = torch.sigmoid(pairing_matrix)
        
        # Symmetrization
        pairing_matrix = (pairing_matrix + pairing_matrix.transpose(1, 2)) / 2
        
        return mfe, pairing_matrix


# ==================== Translation Dynamics Model ====================
class TranslationDynamicsModel(nn.Module):
    """Translation dynamics model."""
    
    def __init__(self, d_model: int, trna_vocab_size: int = 65):
        super().__init__()
        
        # tRNA abundance embedding
        self.trna_embed = nn.Embedding(trna_vocab_size, d_model // 2)
        
        # Pause probability predictor
        self.pause_predictor = nn.Sequential(
            nn.Linear(d_model + d_model // 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Elongation rate predictor
        self.elongation_predictor = nn.Sequential(
            nn.Linear(d_model + d_model // 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Softplus()
        )
    
    def forward(
        self,
        codon_embed: torch.Tensor,
        trna_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            codon_embed: [B, L, D]
            trna_ids: [B, L]
        Returns:
            dict with pause_prob, translation_time, cumulative_time
        """
        # tRNA features
        trna_features = self.trna_embed(trna_ids)  # [B, L, D/2]
        
        # Combined features
        combined = torch.cat([codon_embed, trna_features], dim=-1)
        
        # Pause probability
        pause_prob = self.pause_predictor(combined).squeeze(-1)  # [B, L]
        
        # Elongation rate
        elongation_rate = self.elongation_predictor(combined).squeeze(-1)  # [B, L]
        
        # Translation time
        translation_time = 1.0 / (elongation_rate + 1e-6)
        translation_time = translation_time * (1 + pause_prob * 10)
        
        # Cumulative time
        cumulative_time = torch.cumsum(translation_time, dim=1)
        
        return {
            'pause_prob': pause_prob,
            'elongation_rate': elongation_rate,
            'translation_time': translation_time,
            'cumulative_time': cumulative_time
        }


# ==================== Kinetic Positional Encoding ====================
class PositionalEncoding(nn.Module):
    """Kinetic positional encoding based on translation time."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Standard positional encoding
        pe_standard = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe_standard[:, 0::2] = torch.sin(position * div_term)
        pe_standard[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe_standard', pe_standard)
    
    def forward(
        self,
        x: torch.Tensor,
        translation_times: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            translation_times: [B, L]
        """
        B, L, D = x.shape
        
        # Standard encoding
        pe_std = self.pe_standard[:L].unsqueeze(0).expand(B, -1, -1)
        
        if translation_times is None:
            return self.dropout(x + pe_std)
        
        # Dynamic encoding
        cum_times = torch.cumsum(translation_times, dim=1)
        max_time = cum_times[:, -1:] + 1e-6
        normalized_times = cum_times / max_time
        
        pe_dynamic = torch.zeros(B, L, D, device=x.device, dtype=x.dtype)
        div_term = torch.exp(
            torch.arange(0, D, 2, device=x.device, dtype=x.dtype) *
            (-math.log(10000.0) / D)
        )
        
        times_expanded = normalized_times.unsqueeze(-1)
        pe_dynamic[:, :, 0::2] = torch.sin(times_expanded * div_term)
        pe_dynamic[:, :, 1::2] = torch.cos(times_expanded * div_term)
        
        # Mix
        alpha = torch.sigmoid(self.alpha)
        pe_mixed = alpha * pe_std + (1 - alpha) * pe_dynamic
        
        return self.dropout(x + pe_mixed)

