#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Specialized module for secreted protein expression optimization
Secretion Protein Expression Adapter (SPEA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

# Import base modules
from MPCG_BaseCodonFormer import AA2ID, ID2AA, CODON2ID, ID2CODON, SYN_CODON
from MPCG_CoreModel import FiveSpeciesCodonData


# ==================== Configuration Class ====================
@dataclass
class SPEAConfig:
    """SPEA Configuration"""
    d_model: int = 512
    n_heads: int = 8
    dropout: float = 0.1
    
    # Signal peptide related
    max_signal_length: int = 35
    signal_weight: float = 0.3
    
    # Disulfide bond related
    disulfide_context_window: int = 5
    cys_optimization_strength: float = 0.4
    
    # Solubility related
    solubility_threshold: float = 0.5
    hydrophobicity_penalty: float = 0.5
    
    # E. coli specific
    ecoli_adaptation_weight: float = 0.2
    rare_codon_threshold: float = 0.1


# ==================== E. coli Signal Peptide Database ====================
class SignalPeptideDB:
    """E. coli Signal Peptide Database"""
    
    def __init__(self):
        self.peptides = {
            'PelB': {
                'sequence': 'MKYLLPTAAAGLLLLAAQPAMA',
                'cleavage_site': 22,
                'efficiency': 0.95,
                'suitable_for': ['periplasm', 'disulfide_rich']
            },
            'OmpA': {
                'sequence': 'MKKTAIAIAVALAGFATVAQA',
                'cleavage_site': 21,
                'efficiency': 0.90,
                'suitable_for': ['periplasm', 'general']
            },
            'StII': {
                'sequence': 'MKKNIAFLLASMFVFSIATNAYA',
                'cleavage_site': 23,
                'efficiency': 0.85,
                'suitable_for': ['heat_stable', 'small_proteins']
            },
            'PhoA': {
                'sequence': 'MKQSTIALALLPLLFTPVTKA',
                'cleavage_site': 21,
                'efficiency': 0.88,
                'suitable_for': ['periplasm', 'phosphatase']
            },
            'MalE': {
                'sequence': 'MKIKTGARILALSALTTMMFSASALA',
                'cleavage_site': 26,
                'efficiency': 0.92,
                'suitable_for': ['periplasm', 'fusion_proteins']
            },
            'DsbA': {
                'sequence': 'MKKIWLALAGLVLAFSASA',
                'cleavage_site': 19,
                'efficiency': 0.93,
                'suitable_for': ['disulfide_formation', 'oxidizing']
            },
            'TorA': {
                'sequence': 'MNNNDLFQASRRRFLAQLGGLTVAGMLGPSLLTPRRATAAQAA',
                'cleavage_site': 44,
                'efficiency': 0.80,
                'suitable_for': ['tat_pathway', 'folded_proteins']
            }
        }
        
        # E. coli codon usage frequency (highly expressed genes)
        self.ecoli_optimal_codons = {
            'A': ['GCT', 'GCC'],  # Ala
            'R': ['CGT', 'CGC'],  # Arg
            'N': ['AAC'],         # Asn
            'D': ['GAT', 'GAC'],  # Asp
            'C': ['TGC'],         # Cys
            'Q': ['CAG'],         # Gln
            'E': ['GAA'],         # Glu
            'G': ['GGT', 'GGC'],  # Gly
            'H': ['CAT'],         # His
            'I': ['ATT', 'ATC'],  # Ile
            'L': ['CTG'],         # Leu
            'K': ['AAA'],         # Lys
            'M': ['ATG'],         # Met
            'F': ['TTC'],         # Phe
            'P': ['CCG'],         # Pro
            'S': ['TCT', 'AGC'],  # Ser
            'T': ['ACC'],         # Thr
            'W': ['TGG'],         # Trp
            'Y': ['TAC'],         # Tyr
            'V': ['GTT', 'GTC'],  # Val
            '*': ['TAA']          # Stop
        }
    
    def select_signal_peptide(self, protein_features: Dict) -> str:
        """Select the best signal peptide based on protein features"""
        scores = {}
        
        for name, info in self.peptides.items():
            score = info['efficiency']
            
            # Adjust score based on protein features
            if protein_features.get('has_disulfide', False):
                if 'disulfide_rich' in info['suitable_for']:
                    score += 0.1
                if 'disulfide_formation' in info['suitable_for']:
                    score += 0.15
            
            if protein_features.get('size', 0) < 20000:  # Less than 20kDa
                if 'small_proteins' in info['suitable_for']:
                    score += 0.05
            
            if protein_features.get('need_oxidizing', False):
                if 'oxidizing' in info['suitable_for']:
                    score += 0.1
            
            scores[name] = score
        
        # Return the highest-scoring signal peptide
        return max(scores, key=scores.get)

