#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biologically-informed loss functions (five-species version)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from collections import defaultdict

from MPCG_CoreModel import FiveSpeciesCodonData


# ==================== CAI Calculator ====================
class CAICalculator(nn.Module):
    """CAI Calculator"""
    
    def __init__(self, codon_data: FiveSpeciesCodonData, device: str = 'cpu'):
        super().__init__()
        self.codon_data = codon_data
        self.device = device
        
        # Precompute weight matrices
        self._precompute_weight_matrices()
    
    def _precompute_weight_matrices(self):
        """Precompute CAI weight matrices for all species"""
        from MPCG_BaseCodonFormer import CODON2ID
        
        n_codons = len(CODON2ID)
        n_species = len(self.codon_data.species_list)
        
        weight_matrix = torch.zeros(n_species, n_codons)
        
        for sp_idx, species in enumerate(self.codon_data.species_list):
            weights = self.codon_data.get_cai_weights(species)
            
            for codon, weight in weights.items():
                if codon in CODON2ID:
                    codon_idx = CODON2ID[codon]
                    weight_matrix[sp_idx, codon_idx] = weight
        
        self.register_buffer('weight_matrix', weight_matrix)
        self.species_list = self.codon_data.species_list
    
    def forward(
        self,
        codon_ids: torch.Tensor,
        species_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate CAI
        
        Args:
            codon_ids: [B, L]
            species_ids: [B]
            mask: [B, L]
        Returns:
            cai: [B]
        """
        B, L = codon_ids.shape
        
        # Gather weights
        weights = torch.zeros(B, L, device=codon_ids.device)
        
        for b in range(B):
            sp_idx = species_ids[b].item()
            if 0 <= sp_idx < len(self.species_list):
                for l in range(L):
                    codon_idx = codon_ids[b, l].item()
                    if codon_idx < self.weight_matrix.size(1):
                        weights[b, l] = self.weight_matrix[sp_idx, codon_idx]
        
        # Avoid log(0)
        weights = torch.clamp(weights, min=1e-8)
        
        # CAI = exp(mean(log(weights)))
        log_weights = torch.log(weights)
        
        if mask is not None:
            masked_log_weights = log_weights * mask.float()
            sum_log_weights = masked_log_weights.sum(dim=1)
            valid_counts = mask.float().sum(dim=1).clamp(min=1)
            mean_log_weights = sum_log_weights / valid_counts
        else:
            mean_log_weights = log_weights.mean(dim=1)
        
        cai = torch.exp(mean_log_weights)
        
        return cai

