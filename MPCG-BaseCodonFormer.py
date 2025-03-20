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
