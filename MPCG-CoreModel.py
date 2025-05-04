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
        
