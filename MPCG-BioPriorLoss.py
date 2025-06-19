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


# ==================== RSCU Calculator ====================
class RSCUCalculator(nn.Module):
    """RSCU Calculator"""
    
    def __init__(self, codon_data: FiveSpeciesCodonData, device: str = 'cpu'):
        super().__init__()
        self.codon_data = codon_data
        self.device = device
        
        # Precompute reference distributions
        self._precompute_reference_distributions()
    
    def _precompute_reference_distributions(self):
        """Precompute reference RSCU distributions"""
        from MPCG_BaseCodonFormer import CODON2ID
        
        n_codons = len(CODON2ID)
        n_species = len(self.codon_data.species_list)
        
        ref_distributions = torch.zeros(n_species, n_codons)
        
        for sp_idx, species in enumerate(self.codon_data.species_list):
            rscu_dict = self.codon_data.get_rscu(species)
            
            for codon, rscu_value in rscu_dict.items():
                if codon in CODON2ID:
                    codon_idx = CODON2ID[codon]
                    ref_distributions[sp_idx, codon_idx] = rscu_value
        
        self.register_buffer('ref_distributions', ref_distributions)
        self.species_list = self.codon_data.species_list
    
    def compute_sequence_rscu(
        self,
        codon_ids: torch.Tensor,
        aa_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute sequence RSCU distribution"""
        from MPCG_BaseCodonFormer import ID2AA, CODON2ID, SYN_CODON
        
        B, L = codon_ids.shape
        n_codons = len(CODON2ID)
        
        rscu_dist = torch.zeros(B, n_codons, device=codon_ids.device)
        
        for b in range(B):
            codon_counts = torch.zeros(n_codons, device=codon_ids.device)
            
            if mask is not None:
                valid_positions = mask[b].nonzero(as_tuple=True)[0]
            else:
                valid_positions = torch.arange(L, device=codon_ids.device)
            
            for pos in valid_positions:
                codon_idx = codon_ids[b, pos].item()
                if codon_idx > 0:
                    codon_counts[codon_idx] += 1
            
            # Group by amino acid
            aa_codon_counts = defaultdict(lambda: defaultdict(int))
            
            for pos in valid_positions:
                aa_idx = aa_ids[b, pos].item()
                codon_idx = codon_ids[b, pos].item()
                
                if aa_idx <= 2 or codon_idx == 0:
                    continue
                
                aa = ID2AA.get(aa_idx)
                if aa:
                    aa_codon_counts[aa][codon_idx] = codon_counts[codon_idx].item()
            
            # Compute RSCU
            for aa, codons_dict in aa_codon_counts.items():
                if aa not in SYN_CODON:
                    continue
                
                synonymous_codons = SYN_CODON[aa]
                n_synonymous = len(synonymous_codons)
                
                total_count = sum(codons_dict.values())
                
                if total_count > 0:
                    for codon_idx, count in codons_dict.items():
                        rscu_value = (count * n_synonymous) / total_count
                        rscu_dist[b, codon_idx] = rscu_value
        
        return rscu_dist
    
    def forward(
        self,
        pred_codon_ids: torch.Tensor,
        target_codon_ids: torch.Tensor,
        aa_ids: torch.Tensor,
        species_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute RSCU KL divergence"""
        # Predicted and target RSCU
        pred_rscu = self.compute_sequence_rscu(pred_codon_ids, aa_ids, mask)
        target_rscu = self.compute_sequence_rscu(target_codon_ids, aa_ids, mask)
        
        # Species reference distribution
        B = pred_codon_ids.size(0)
        ref_rscu = torch.zeros_like(pred_rscu)
        for b in range(B):
            sp_idx = species_ids[b].item()
            if 0 <= sp_idx < len(self.species_list):
                ref_rscu[b] = self.ref_distributions[sp_idx]
        
        # Combined target
        combined_target = 0.7 * target_rscu + 0.3 * ref_rscu
        
        # Smoothing
        pred_rscu = pred_rscu + 1e-8
        combined_target = combined_target + 1e-8
        
        # Normalization
        pred_dist = pred_rscu / pred_rscu.sum(dim=1, keepdim=True)
        target_dist = combined_target / combined_target.sum(dim=1, keepdim=True)
        
        # KL divergence
        kl_div = (target_dist * torch.log(target_dist / pred_dist)).sum(dim=1)
        
        return kl_div

