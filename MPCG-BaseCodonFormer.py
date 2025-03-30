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


# ==================== Sparse Attention Mechanism ====================
class SparseAttention(nn.Module):
    """Sparse attention mechanism - combining local and global attention"""
    
    def __init__(self, d_model, n_heads, window_size=64, global_ratio=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.global_ratio = global_ratio
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        # project to Q, K, V
        Q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(B, L, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(B, L, self.n_heads, self.head_dim)
        
        # for short sequences, use standard attention
        if L <= self.window_size:
            output = self._standard_attention(Q, K, V, mask)
        else:
            # local attention (sliding window)
            local_attn = self._local_attention(Q, K, V, mask)
            
            # global attention (random sampling)
            global_attn = self._global_attention(Q, K, V, mask)
            
            # fuse local and global attention
            output = 0.7 * local_attn + 0.3 * global_attn
        
        return self.out_proj(output.reshape(B, L, D))
    
    def _get_neg_inf(self, dtype):
        """Return appropriate negative infinity value based on data type"""
        if dtype == torch.float16:
            return -65000.0  # close to FP16 minimum value
        elif dtype == torch.bfloat16:
            return -3e38  # close to BF16 minimum value
        else:
            return -1e9  # FP32
    
    def _standard_attention(self, Q, K, V, mask):
        """Standard full attention"""
        B, L, H, D = Q.shape
        
        # [B, H, L, L]
        scores = torch.matmul(
            Q.transpose(1, 2),  # [B, H, L, D]
            K.transpose(1, 2).transpose(-2, -1)  # [B, H, D, L]
        ) / math.sqrt(D)
        
        if mask is not None:
            # mask: [B, L] -> [B, 1, 1, L]
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)
            neg_inf = self._get_neg_inf(scores.dtype)
            scores = scores.masked_fill(mask_expanded, neg_inf)
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V.transpose(1, 2))  # [B, H, L, D]
        
        return output.transpose(1, 2)  # [B, L, H, D]
    
    def _local_attention(self, Q, K, V, mask):
        """Local sliding window attention"""
        B, L, H, D = Q.shape
        output = torch.zeros_like(Q)
        neg_inf = self._get_neg_inf(Q.dtype)
        
        # compute local window for each position
        half_window = self.window_size // 2
        
        for i in range(L):
            # determine window range for current position
            start_pos = max(0, i - half_window)
            end_pos = min(L, i + half_window + 1)
            
            # query at current position
            q_i = Q[:, i:i+1, :, :]  # [B, 1, H, D]
            
            # key and value within window
            k_window = K[:, start_pos:end_pos, :, :]  # [B, window_len, H, D]
            v_window = V[:, start_pos:end_pos, :, :]  # [B, window_len, H, D]
            
            # compute attention scores
            # [B, 1, H, D] @ [B, H, D, window_len] -> [B, H, 1, window_len]
            scores = torch.matmul(
                q_i.transpose(1, 2),  # [B, H, 1, D]
                k_window.transpose(1, 2).transpose(-2, -1)  # [B, H, D, window_len]
            ) / math.sqrt(D)
            
            # apply mask
            if mask is not None:
                mask_window = mask[:, start_pos:end_pos].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, window_len]
                scores = scores.masked_fill(mask_window, neg_inf)
            
            # attention weights
            attn_weights = F.softmax(scores, dim=-1)  # [B, H, 1, window_len]
            
            # weighted sum
            # [B, H, 1, window_len] @ [B, H, window_len, D] -> [B, H, 1, D]
            local_out = torch.matmul(
                attn_weights,
                v_window.transpose(1, 2)
            ).transpose(1, 2)  # [B, 1, H, D]
            
            output[:, i:i+1, :, :] = local_out
        
        return output
    
    def _global_attention(self, Q, K, V, mask):
        """Global random attention"""
        B, L, H, D = Q.shape
        n_global = max(1, int(L * self.global_ratio))
        neg_inf = self._get_neg_inf(Q.dtype)
        
        # randomly select global positions
        global_indices = torch.randperm(L, device=Q.device)[:n_global].sort()[0]
        
        # use K and V at global positions
        k_global = K[:, global_indices, :, :]  # [B, n_global, H, D]
        v_global = V[:, global_indices, :, :]  # [B, n_global, H, D]
        
        # compute attention between Q at all positions and global K
        # [B, L, H, D] @ [B, H, D, n_global] -> [B, H, L, n_global]
        scores = torch.matmul(
            Q.transpose(1, 2),  # [B, H, L, D]
            k_global.transpose(1, 2).transpose(-2, -1)  # [B, H, D, n_global]
        ) / math.sqrt(D)
        
        if mask is not None:
            # mask for global positions
            mask_global = mask[:, global_indices].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, n_global]
            scores = scores.masked_fill(mask_global, neg_inf)
        
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, L, n_global]
        
        # [B, H, L, n_global] @ [B, H, n_global, D] -> [B, H, L, D]
        output = torch.matmul(
            attn_weights,
            v_global.transpose(1, 2)
        ).transpose(1, 2)  # [B, L, H, D]
        
        return output

