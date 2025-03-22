#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Enhanced CodonFormer-Encoder with Sparse Attention and Multi-objective Loss
"""

import math, json, argparse, pathlib, random, itertools
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from scipy.stats import entropy

# RNA structure prediction (optional)
try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False
    print("⚠️  ViennaRNA not found, MFE calculation will use approximation")

# ==================== Global Dictionary Definitions (Auto-generated via Biopython) ====================
from Bio.Data import CodonTable

# Retrieve standard genetic code table
standard_table = CodonTable.unambiguous_dna_by_name["Standard"]

# 1. Amino acid dictionary
AA_TOKENS = ["<pad>", "<bos>", "<eos>"] + list("ACDEFGHIKLMNPQRSTVWY*")
AA2ID = {aa: i for i, aa in enumerate(AA_TOKENS)}
ID2AA = {v: k for k, v in AA2ID.items()}

# 2. Codon dictionary (including stop codons)
ALL_CODONS = list(standard_table.forward_table.keys()) + standard_table.stop_codons
CODON_TOKENS = ["<pad>"] + ALL_CODONS
CODON2ID = {c: i for i, c in enumerate(CODON_TOKENS)}
ID2CODON = {i: c for c, i in CODON2ID.items()}

# 3. Synonymous codon set SYN_CODON
SYN_CODON = {}
for codon, aa in standard_table.forward_table.items():
    SYN_CODON.setdefault(aa, []).append(codon)
for stop in standard_table.stop_codons:
    SYN_CODON.setdefault("*", []).append(stop)

# ==================== Auxiliary Conversion Functions ====================
def aa_to_ids(aa_seq):
    """Convert amino acid sequence to ID list"""
    ids = [AA2ID["<bos>"]]
    for aa in aa_seq:
        ids.append(AA2ID.get(aa, AA2ID.get("*", 0)))
    ids.append(AA2ID["<eos>"])
    return ids


def codon_to_ids(codon_seq):
    """Convert codon sequence to ID list"""
    return [CODON2ID.get(codon, 0) for codon in codon_seq]


def synonym_mask(logits, aa_ids):
    """
    Apply synonymous codon mask to ensure only predicting synonymous codons corresponding to the amino acid
    
    Args:
        logits: [B, L, V_codon] model output logits
        aa_ids: [B, L] amino acid ID sequence
    
    Returns:
        masked_logits: logits after applying mask
    """
    B, L, V = logits.shape
    mask = torch.ones_like(logits) * -1e9
    
    for b in range(B):
        for i in range(1, L-1):  # skip <bos> and <eos>
            aa_id = aa_ids[b, i].item()
            aa = ID2AA.get(aa_id)
            
            if aa and aa in SYN_CODON:
                # get all synonymous codons for this amino acid
                valid_codons = SYN_CODON[aa]
                for codon in valid_codons:
                    if codon in CODON2ID:
                        codon_id = CODON2ID[codon]
                        mask[b, i, codon_id] = 0
    
    return logits + mask


def ids_to_codons(codon_ids):
    """Convert codon ID list back to codon sequence"""
    return [ID2CODON.get(cid, "<pad>") for cid in codon_ids]


def ids_to_aa(aa_ids):
    """Convert amino acid ID list back to amino acid sequence"""
    return [ID2AA.get(aid, "<pad>") for aid in aa_ids]


# ==================== Positional Encoding ====================
class PosEnc(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: [B, L, D] or [B, L] (if ID sequence)
        """
        if x.dim() == 2:
            # if input is ID sequence, return corresponding positional encoding
            B, L = x.shape
            return self.pe[:, :L, :]
        else:
            # if input is embedding vector, add positional encoding directly
            return x + self.pe[:, :x.size(1), :]


# ==================== Data Preprocessing Augmentation ====================
class BiologicalFeatureExtractor:
    """Biological feature extractor"""
    
    def __init__(self):
        self.gc_target = 0.5
        self.optimal_cai = 1.0
        
    def calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content"""
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence) if len(sequence) > 0 else 0.0
    
    def calculate_gc_variance(self, sequence: str, window_size: int = 30) -> float:
        """Calculate local GC content variance"""
        if len(sequence) < window_size:
            return 0.0
        
        gc_contents = []
        for i in range(0, len(sequence) - window_size + 1, 10):
            window = sequence[i:i + window_size]
            gc_contents.append(self.calculate_gc_content(window))
        
        return np.var(gc_contents) if gc_contents else 0.0
    
    def calculate_mfe(self, sequence: str) -> float:
        """Calculate minimum folding energy"""
        if HAS_VIENNA:
            try:
                fc = RNA.fold_compound(sequence)
                structure, mfe = fc.mfe()
                return mfe
            except:
                pass
        
        # simplified estimation method
        gc_content = self.calculate_gc_content(sequence)
        return -0.5 * len(sequence) * gc_content  # rough estimation
    
    def calculate_rscu(self, codons: List[str]) -> Dict[str, float]:
        """Calculate relative synonymous codon usage (RSCU)"""
        # count frequency of each codon
        codon_counts = {}
        aa_counts = {}
        
        for codon in codons:
            if codon in CODON2ID and codon != "<pad>":
                aa = self.codon_to_aa(codon)
                codon_counts[codon] = codon_counts.get(codon, 0) + 1
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        # calculate RSCU
        rscu = {}
        for codon, count in codon_counts.items():
            aa = self.codon_to_aa(codon)
            synonymous_count = len(SYN_CODON.get(aa, [codon]))
            expected_freq = aa_counts[aa] / synonymous_count
            rscu[codon] = count / expected_freq if expected_freq > 0 else 1.0
        
        return rscu
    
    def codon_to_aa(self, codon: str) -> str:
        """Codon to amino acid mapping"""
        if codon in standard_table.forward_table:
            return standard_table.forward_table[codon]
        elif codon in standard_table.stop_codons:
            return '*'
        else:
            return 'X'
