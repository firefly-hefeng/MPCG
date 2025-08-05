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


# ==================== Signal Peptide Aware Adapter ====================
class SecretionSignalAdapter(nn.Module):
    """Signal peptide region optimization adapter"""
    
    def __init__(self, config: SPEAConfig):
        super().__init__()
        self.config = config
        self.signal_db = SignalPeptideDB()
        
        # Signal peptide region detector
        self.region_classifier = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 0: padding, 1: signal, 2: mature
        )
        
        # Signal peptide optimizer
        self.signal_optimizer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        
        # Cleavage site predictor
        self.cleavage_predictor = nn.Sequential(
            nn.Linear(config.d_model * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # E. coli specific adjustment
        self.ecoli_codon_bias = nn.Parameter(
            torch.randn(len(CODON2ID), config.d_model) * 0.02
        )
        
        # Positional encoding (for signal peptide)
        self.signal_pos_encoding = nn.Parameter(
            torch.randn(config.max_signal_length, config.d_model) * 0.02
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        aa_ids: torch.Tensor,
        position_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            hidden_states: [B, L, D]
            aa_ids: [B, L]
            position_labels: [B, L] 1=signal, 2=mature
        """
        B, L, D = hidden_states.shape
        device = hidden_states.device
        
        # Region classification
        region_logits = self.region_classifier(hidden_states)
        region_probs = F.softmax(region_logits, dim=-1)
        
        # Use ground truth labels if provided; otherwise use predictions
        if position_labels is not None:
            is_signal = (position_labels == 1).float()
        else:
            is_signal = (region_probs.argmax(dim=-1) == 1).float()
        
        # Signal peptide region mask
        signal_mask = is_signal.unsqueeze(-1)  # [B, L, 1]
        
        # Apply special optimization to signal peptide region
        signal_optimized = self.signal_optimizer(hidden_states)
        
        # Add positional encoding (signal peptide region only)
        for b in range(B):
            signal_positions = torch.where(is_signal[b] > 0)[0]
            if len(signal_positions) > 0:
                max_pos = min(len(signal_positions), self.config.max_signal_length)
                hidden_states[b, signal_positions[:max_pos]] += \
                    self.signal_pos_encoding[:max_pos]
        
        # Mix original and optimized representations
        hidden_states = hidden_states + signal_mask * self.config.signal_weight * signal_optimized
        
        # Predict cleavage site
        cleavage_scores = []
        for i in range(1, L):
            context = torch.cat([hidden_states[:, i-1], hidden_states[:, i]], dim=-1)
            score = self.cleavage_predictor(context)
            cleavage_scores.append(score)
        
        if cleavage_scores:
            cleavage_probs = torch.cat(cleavage_scores, dim=1)
        else:
            cleavage_probs = torch.zeros(B, L-1, device=device)
        
        # Add E. coli codon preference
        ecoli_adjustment = self._apply_ecoli_bias(hidden_states, aa_ids)
        hidden_states = hidden_states + self.config.ecoli_adaptation_weight * ecoli_adjustment
        
        outputs = {
            'hidden_states': hidden_states,
            'region_probs': region_probs,
            'cleavage_probs': cleavage_probs,
            'signal_mask': signal_mask
        }
        
        return hidden_states, outputs
    
    def _apply_ecoli_bias(self, hidden_states: torch.Tensor, aa_ids: torch.Tensor) -> torch.Tensor:
        """Apply E. coli codon preference"""
        B, L, D = hidden_states.shape
        bias = torch.zeros_like(hidden_states)
        
        for b in range(B):
            for l in range(L):
                aa_id = aa_ids[b, l].item()
                if aa_id > 2:  # Skip special tokens
                    aa = ID2AA.get(aa_id, '')
                    if aa in self.signal_db.ecoli_optimal_codons:
                        optimal_codons = self.signal_db.ecoli_optimal_codons[aa]
                        # Add preference for optimal codons
                        for codon in optimal_codons:
                            if codon in CODON2ID:
                                codon_id = CODON2ID[codon]
                                bias[b, l] += self.ecoli_codon_bias[codon_id]
        
        return bias / (len(CODON2ID) + 1e-8)


# ==================== Disulfide Bond Aware Module ====================
class DisulfideBondAwareModule(nn.Module):
    """Disulfide bond formation optimization module"""
    
    def __init__(self, config: SPEAConfig):
        super().__init__()
        self.config = config
        
        # Cysteine pair pairing prediction
        self.cys_pair_predictor = nn.Bilinear(
            config.d_model, config.d_model, 128
        )
        self.pair_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Redox environment optimization
        self.redox_optimizer = nn.LSTM(
            config.d_model,
            config.d_model // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout
        )
        
        # Local structure stabilization
        self.local_structure_conv = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=2 * config.disulfide_context_window + 1,
            padding=config.disulfide_context_window
        )
        
        # DsbA/DsbC cofactor simulation
        self.dsb_factor = nn.Parameter(torch.randn(config.d_model) * 0.1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        aa_ids: torch.Tensor,
        signal_outputs: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Optimize expression of disulfide-bond-containing proteins
        """
        B, L, D = hidden_states.shape
        device = hidden_states.device
        
        # Find all cysteine positions
        cys_positions_batch = []
        for b in range(B):
            cys_pos = (aa_ids[b] == AA2ID.get('C', -1)).nonzero(as_tuple=False).squeeze(-1)
            cys_positions_batch.append(cys_pos)
        
        # Predict disulfide bond pairings
        disulfide_pairs = self._predict_disulfide_pairs(
            hidden_states, cys_positions_batch
        )
        
        # Optimize redox environment
        optimized_hidden, _ = self.redox_optimizer(hidden_states)
        
        # Enhance optimization around cysteines
        cys_enhanced = hidden_states.clone()
        for b, cys_positions in enumerate(cys_positions_batch):
            if len(cys_positions) > 0:
                for pos in cys_positions:
                    # Define context window
                    start = max(0, pos - self.config.disulfide_context_window)
                    end = min(L, pos + self.config.disulfide_context_window + 1)
                    
                    # Apply optimization
                    cys_enhanced[b, start:end] = (
                        hidden_states[b, start:end] + 
                        self.config.cys_optimization_strength * optimized_hidden[b, start:end]
                    )
                    
                    # Add DsbA factor (promotes disulfide bond formation)
                    cys_enhanced[b, pos] += self.dsb_factor
        
        # Local structure stabilization
        structure_enhanced = self.local_structure_conv(
            cys_enhanced.transpose(1, 2)
        ).transpose(1, 2)
        
        # Mix enhancements
        output = hidden_states + 0.3 * (cys_enhanced - hidden_states) + 0.2 * (structure_enhanced - hidden_states)
        
        outputs = {
            'hidden_states': output,
            'disulfide_pairs': disulfide_pairs,
            'cys_positions': cys_positions_batch,
            'redox_optimized': optimized_hidden
        }
        
        return output, outputs
    
    def _predict_disulfide_pairs(
        self,
        hidden_states: torch.Tensor,
        cys_positions_batch: List[torch.Tensor]
    ) -> List[List[Tuple[int, int, float]]]:
        """Predict disulfide bond pairings"""
        all_pairs = []
        
        for b, cys_positions in enumerate(cys_positions_batch):
            pairs = []
            n_cys = len(cys_positions)
            
            if n_cys >= 2:
                # Calculate all possible pairings
                for i in range(n_cys):
                    for j in range(i + 1, n_cys):
                        pos_i = cys_positions[i].item()
                        pos_j = cys_positions[j].item()
                        
                        # Extract features
                        feat_i = hidden_states[b, pos_i]
                        feat_j = hidden_states[b, pos_j]
                        
                        # Predict pairing probability
                        pair_features = self.cys_pair_predictor(
                            feat_i.unsqueeze(0), feat_j.unsqueeze(0)
                        )
                        pair_prob = self.pair_classifier(pair_features).item()
                        
                        # Distance constraint (disulfide bonds usually have some distance in sequence)
                        distance = abs(pos_j - pos_i)
                        if distance < 2:  # Cysteines too close are unlikely to form disulfide bonds
                            pair_prob *= 0.1
                        
                        pairs.append((pos_i, pos_j, pair_prob))
                
                # Sort and select most likely pairings (avoid conflicts)
                pairs.sort(key=lambda x: x[2], reverse=True)
                selected_pairs = []
                used_positions = set()
                
                for pos_i, pos_j, prob in pairs:
                    if pos_i not in used_positions and pos_j not in used_positions:
                        if prob > 0.5:  # Threshold
                            selected_pairs.append((pos_i, pos_j, prob))
                            used_positions.add(pos_i)
                            used_positions.add(pos_j)
                
                pairs = selected_pairs
            
            all_pairs.append(pairs)
        
        return all_pairs

