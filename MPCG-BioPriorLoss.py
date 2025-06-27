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


# ==================== Comprehensive Loss Function ====================
class BiologicallyInformedLoss(nn.Module):
    """Multi-objective loss based on biological priors"""
    
    def __init__(
        self,
        codon_data: FiveSpeciesCodonData,
        weights: Optional[Dict[str, float]] = None,
        device: str = 'cpu'
    ):
        super().__init__()
        self.codon_data = codon_data
        self.device = device
        
        self.weights = weights or {
            'ce': 1.0,
            'cai': 0.4,
            'rscu': 0.3,
            'gc': 0.1,
            'structure': 0.15,
            'dynamics': 0.1,
            'rare_codon': 0.2,
            'manufacturability': 0.05
        }
        
        self.cai_calculator = CAICalculator(codon_data, device)
        self.rscu_calculator = RSCUCalculator(codon_data, device)
        
        self.rare_threshold = 0.6
    
    def forward(
        self,
        logits: torch.Tensor,
        target_codon_ids: torch.Tensor,
        aa_ids: torch.Tensor,
        species_ids: torch.Tensor,
        features: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute comprehensive loss
        
        Args:
            logits: [B, L, V_codon]
            target_codon_ids: [B, L]
            aa_ids: [B, L]
            species_ids: [B]
            features: dict
            mask: [B, L]
        """
        losses = {}
        
        pred_codon_ids = logits.argmax(dim=-1)
        
        # 1. Cross-entropy
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_codon_ids.reshape(-1),
            ignore_index=0,
            reduction='mean'
        )
        losses['ce'] = ce_loss
        
        # 2. CAI loss
        pred_cai = self.cai_calculator(pred_codon_ids, species_ids, mask)
        target_cai = self.cai_calculator(target_codon_ids, species_ids, mask)
        cai_loss = F.relu(target_cai - pred_cai).mean()
        losses['cai'] = cai_loss
        
        # 3. RSCU loss
        rscu_loss = self.rscu_calculator(
            pred_codon_ids, target_codon_ids, aa_ids, species_ids, mask
        ).mean()
        losses['rscu'] = rscu_loss
        
        # 4. GC content loss
        if features and 'gc_pred' in features:
            target_gc = 0.5
            gc_loss = F.mse_loss(
                features['gc_pred'].mean(dim=1),
                torch.full((features['gc_pred'].size(0),), target_gc, device=self.device)
            )
            losses['gc'] = gc_loss
        else:
            losses['gc'] = torch.tensor(0.0, device=self.device)
        
        # 5. RNA structure loss
        if features and 'mfe' in features:
            target_mfe = -20.0
            structure_loss = F.mse_loss(
                features['mfe'],
                torch.full_like(features['mfe'], target_mfe)
            )
            losses['structure'] = structure_loss
        else:
            losses['structure'] = torch.tensor(0.0, device=self.device)
        
        # 6. Translation dynamics loss
        if features and 'pause_prob' in features:
            pause_target = 0.1
            pause_loss = F.mse_loss(
                features['pause_prob'].mean(dim=1),
                torch.full((features['pause_prob'].size(0),), pause_target, device=self.device)
            )
            losses['dynamics'] = pause_loss
        else:
            losses['dynamics'] = torch.tensor(0.0, device=self.device)
        
        # 7. Rare codon protection
        rare_codon_loss = self._compute_rare_codon_loss(
            pred_codon_ids, target_codon_ids, species_ids, mask
        )
        losses['rare_codon'] = rare_codon_loss
        
        # 8. Manufacturability
        manufact_loss = self._compute_manufacturability_loss(pred_codon_ids, mask)
        losses['manufacturability'] = manufact_loss
        
        # Total loss
        total_loss = sum(self.weights.get(k, 0.0) * v for k, v in losses.items())
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def _compute_rare_codon_loss(
        self,
        pred_codon_ids: torch.Tensor,
        target_codon_ids: torch.Tensor,
        species_ids: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Rare codon protection loss"""
        from MPCG_BaseCodonFormer import ID2CODON
        
        B, L = pred_codon_ids.shape
        losses = []
        
        for b in range(B):
            sp_idx = species_ids[b].item()
            if sp_idx >= len(self.rscu_calculator.species_list):
                continue
            
            species = self.rscu_calculator.species_list[sp_idx]
            rscu_dict = self.codon_data.get_rscu(species)
            
            # Identify rare codon clusters
            rare_clusters = []
            cluster_start = None
            
            for l in range(L):
                if mask is not None and not mask[b, l]:
                    continue
                
                target_codon_idx = target_codon_ids[b, l].item()
                if target_codon_idx == 0:
                    continue
                
                target_codon = ID2CODON.get(target_codon_idx)
                if not target_codon:
                    continue
                
                rscu_value = rscu_dict.get(target_codon, 1.0)
                
                if rscu_value < self.rare_threshold:
                    if cluster_start is None:
                        cluster_start = l
                else:
                    if cluster_start is not None and (l - cluster_start) >= 3:
                        rare_clusters.append((cluster_start, l))
                    cluster_start = None
            
            if cluster_start is not None and (L - cluster_start) >= 3:
                rare_clusters.append((cluster_start, L))
            
            # Intra-cluster loss
            cluster_loss = 0.0
            for start, end in rare_clusters:
                cluster_pred = pred_codon_ids[b, start:end]
                cluster_target = target_codon_ids[b, start:end]
                mismatch = (cluster_pred != cluster_target).float().sum()
                cluster_loss += mismatch / max(1, end - start)
            
            if len(rare_clusters) > 0:
                cluster_loss /= len(rare_clusters)
            
            losses.append(cluster_loss)
        
        if losses:
            return torch.stack([torch.tensor(l, device=self.device) for l in losses]).mean()
        else:
            return torch.tensor(0.0, device=self.device)
    
    def _compute_manufacturability_loss(
        self,
        codon_ids: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Manufacturability loss"""
        B, L = codon_ids.shape
        losses = []
        
        for b in range(B):
            seq = codon_ids[b]
            if mask is not None:
                seq = seq[mask[b]]
            
            if len(seq) < 6:
                losses.append(torch.tensor(0.0, device=self.device))
                continue
            
            # Triplet repeats
            repeat_penalty = 0
            for i in range(len(seq) - 5):
                if torch.equal(seq[i:i+3], seq[i+3:i+6]):
                    repeat_penalty += 1
            repeat_penalty = repeat_penalty / max(1, len(seq) - 5)
            
            # Homopolymers
            homopolymer_penalty = 0
            current_run = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    current_run += 1
                    if current_run >= 4:
                        homopolymer_penalty += 1
                else:
                    current_run = 1
            homopolymer_penalty = homopolymer_penalty / max(1, len(seq))
            
            total_loss = repeat_penalty + homopolymer_penalty
            losses.append(torch.tensor(total_loss, device=self.device))
        
        return torch.stack(losses).mean()

