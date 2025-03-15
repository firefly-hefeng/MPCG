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

