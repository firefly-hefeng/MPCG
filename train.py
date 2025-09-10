#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPCG-Codon Training Script
Supports codon optimization training for five species
"""

import os
import time
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

try:
    from torch.amp import autocast, GradScaler
    USE_NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    USE_NEW_AMP = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Import model components
from MPCG_BaseCodonFormer import (
    AA_TOKENS, CODON_TOKENS, SYN_CODON, ID2AA, CODON2ID, ID2CODON,
    aa_to_ids, codon_to_ids, ids_to_aa, ids_to_codons,
    BiologicalFeatureExtractor
)

from MPCG_CoreModel import (
    MPCGConfig,
    MPCGCodon,
    FiveSpeciesCodonData
)

from MPCG_BioPriorLoss import BiologicallyInformedLoss


# ==================== Logging Configuration ====================
def setup_logging(log_dir: str):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


# ==================== Utility Functions ====================
def set_seed(seed: int):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Average value recorder"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


# ==================== Dataset ====================
class MPCGCodonDataset(Dataset):
    """MPCG-Codon Dataset (Corrected Version)"""
    
    def __init__(
        self,
        aa_sequences: List[str],
        nn_sequences: List[str],
        organisms: List[str],
        codon_data: FiveSpeciesCodonData,
        max_length: int = 2000,
        augment: bool = True
    ):
        self.data = []
        self.codon_data = codon_data
        self.feature_extractor = BiologicalFeatureExtractor()
        self.max_length = max_length
        self.augment = augment
        
        # Species to ID mapping
        self.sp2id = {
            species: idx for idx, species in enumerate(codon_data.species_list)
        }
        self.sp2id['<pad>'] = len(self.sp2id)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Building dataset from {len(aa_sequences)} sequences...")
        
        valid_count = 0
        skipped_length = 0
        skipped_mismatch = 0
        
        for aa_seq, nn_seq, organism in zip(aa_sequences, nn_sequences, organisms):
            # Ensure strings
            aa_seq_str = ''.join(aa_seq) if isinstance(aa_seq, list) else str(aa_seq)
            nn_seq_str = str(nn_seq).upper()
            
            # Remove whitespace
            aa_seq_str = aa_seq_str.strip().upper()
            nn_seq_str = nn_seq_str.strip()
            
            # Ensure stop codon
            if not aa_seq_str.endswith('*'):
                aa_seq_str += '*'
            
            # Filter out sequences that are too long
            if len(aa_seq_str) > max_length:
                skipped_length += 1
                continue
            
            # Extract codons
            codons = [nn_seq_str[i:i+3] for i in range(0, len(nn_seq_str), 3)]
            
            # Filter incomplete last codon
            if len(codons[-1]) != 3:
                codons = codons[:-1]
            
            # Ensure length match: number of codons should equal number of amino acids
            if len(codons) != len(aa_seq_str):
                # Try to fix
                if len(codons) == len(aa_seq_str) - 1:
                    # Missing stop codon, add one
                    # Choose stop codon based on the last amino acid
                    if aa_seq_str[-1] == '*':
                        # Choose the most common stop codon
                        stop_codons = ['TAA', 'TAG', 'TGA']
                        codons.append(stop_codons[0])
                elif len(codons) == len(aa_seq_str) + 1:
                    # One extra codon, remove stop codon
                    aa_seq_str = aa_seq_str[:-1]
                else:
                    skipped_mismatch += 1
                    continue
            
            # Check length again
            if len(codons) != len(aa_seq_str):
                skipped_mismatch += 1
                continue
            
            # Validate codon validity
            valid_codons = []
            valid_aa = []
            for i, (aa, codon) in enumerate(zip(aa_seq_str, codons)):
                # Check codon length
                if len(codon) != 3:
                    break
                
                # Check if codon only contains ATCG
                if not all(nt in 'ATCGU' for nt in codon):
                    break
                
                valid_codons.append(codon)
                valid_aa.append(aa)
            
            # Skip if there are invalid codons
            if len(valid_codons) != len(codons):
                skipped_mismatch += 1
                continue
            
            # Convert to IDs (do not add BOS/EOS as the model will add them)
            aa_ids = []
            for aa in aa_seq_str:
                if aa in AA_TOKENS:
                    aa_idx = AA_TOKENS.index(aa)
                    aa_ids.append(aa_idx)
                else:
                    # Replace unknown amino acids with 'X'
                    aa_ids.append(AA_TOKENS.index('X'))
            
            codon_ids = []
            for codon in codons:
                if codon in CODON_TOKENS:
                    codon_idx = CODON_TOKENS.index(codon)
                    codon_ids.append(codon_idx)
                else:
                    # Replace unknown codons with the first synonymous codon
                    aa = aa_seq_str[len(codon_ids)] if len(codon_ids) < len(aa_seq_str) else '*'
                    if aa in SYN_CODON and SYN_CODON[aa]:
                        alt_codon = SYN_CODON[aa][0]
                        if alt_codon in CODON_TOKENS:
                            codon_idx = CODON_TOKENS.index(alt_codon)
                            codon_ids.append(codon_idx)
                        else:
                            codon_ids.append(0)  # padding
                    else:
                        codon_ids.append(0)
            
            # Final length check
            if len(aa_ids) != len(codon_ids):
                skipped_mismatch += 1
                continue
            
            # Species ID
            sp_id = self.sp2id.get(organism, self.sp2id.get('<pad>'))
            
            # Extract biological features
            aux_features = self._extract_features(aa_seq_str, codons, organism)
            
            # Nucleotide sequence
            nucleotide_seq = self._codons_to_nucleotides(codons)
            
            # tRNA ID
            trna_ids = codon_ids.copy()
            
            self.data.append({
                'aa_ids': aa_ids,
                'codon_ids': codon_ids,
                'sp_id': sp_id,
                'aux_features': aux_features,
                'nucleotide_seq': nucleotide_seq,
                'trna_ids': trna_ids,
                'organism': organism,
                'aa_seq': aa_seq_str,
                'codons': codons
            })
            
            valid_count += 1
        
        logger.info(f"Dataset built with {valid_count} valid sequences")
        logger.info(f"Skipped: {skipped_length} too long, {skipped_mismatch} length mismatch")
    
    def _extract_features(
        self,
        aa_seq: str,
        codons: List[str],
        organism: str
    ) -> torch.Tensor:
        """Extract 64-dimensional biological features"""
        features = []
        
        # 1. Sequence length (normalized)
        features.append(len(aa_seq) / 1000.0)
        
        # 2. Amino acid composition (20-dim)
        aa_list = 'ARNDCQEGHILKMFPSTWYV'
        for aa in aa_list:
            features.append(aa_seq.count(aa) / max(1, len(aa_seq)))
        
        # 3. DNA sequence features
        dna_seq = ''.join(codons)
        
        # GC content
        gc_content = self.feature_extractor.calculate_gc_content(dna_seq)
        features.append(gc_content)
        
        # GC variance
        gc_variance = self.feature_extractor.calculate_gc_variance(dna_seq)
        features.append(gc_variance)
        
        # 4. Codon entropy
        from scipy.stats import entropy
        codon_counts = {}
        for codon in codons:
            codon_counts[codon] = codon_counts.get(codon, 0) + 1
        
        if len(codon_counts) > 1:
            probs = np.array(list(codon_counts.values())) / len(codons)
            codon_entropy = entropy(probs)
        else:
            codon_entropy = 0.0
        features.append(codon_entropy)
        
        # 5. Rare codon ratio
        rscu_dict = self.codon_data.get_rscu(organism)
        rare_count = 0
        for codon in codons:
            rscu_value = rscu_dict.get(codon, 1.0)
            if rscu_value < 0.6:
                rare_count += 1
        features.append(rare_count / max(1, len(codons)))
        
        # 6. Dinucleotide frequency (16-dim)
        dinucleotides = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
                         'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
        for di in dinucleotides:
            count = dna_seq.count(di)
            features.append(count / max(1, len(dna_seq) - 1))
        
        # 7. CpG count
        cpg_count = dna_seq.count('CG')
        features.append(cpg_count / max(1, len(dna_seq)))
        
        # 8. Protein physicochemical properties (5-dim)
        hydrophobic = sum(1 for aa in aa_seq if aa in 'AILMFVP') / max(1, len(aa_seq))
        charged = sum(1 for aa in aa_seq if aa in 'DEKR') / max(1, len(aa_seq))
        polar = sum(1 for aa in aa_seq if aa in 'STNQCYWH') / max(1, len(aa_seq))
        aromatic = sum(1 for aa in aa_seq if aa in 'FYW') / max(1, len(aa_seq))
        small = sum(1 for aa in aa_seq if aa in 'AGSVT') / max(1, len(aa_seq))
        features.extend([hydrophobic, charged, polar, aromatic, small])
        
        # 9. CAI and tRNA adaptation index
        cai_weights = self.codon_data.get_cai_weights(organism)
        cai_values = [cai_weights.get(c, 0.5) for c in codons]
        features.append(np.mean(cai_values) if cai_values else 0.5)
        
        trna_abundance = self.codon_data.get_trna_abundance(organism)
        trna_values = [trna_abundance.get(c, 0.5) for c in codons]
        features.append(np.mean(trna_values) if trna_values else 0.5)
        
        # Pad to 64 dimensions
        while len(features) < 64:
            features.append(0.0)
        
        return torch.tensor(features[:64], dtype=torch.float32)
    
    def _codons_to_nucleotides(self, codons: List[str]) -> List[int]:
        """Convert codons to nucleotide IDs"""
        nucleotide_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'U': 4, 'N': 0}
        nucleotides = []
        for codon in codons:
            for nt in codon:
                nucleotides.append(nucleotide_map.get(nt, 0))
        return nucleotides
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Data augmentation
        if self.augment and random.random() < 0.3:
            item = self._augment(item)
        
        return (
            item['aa_ids'],
            item['codon_ids'],
            item['sp_id'],
            item['aux_features'],
            item['nucleotide_seq'],
            item['trna_ids']
        )
    
    def _augment(self, item):
        """Synonymous codon replacement data augmentation"""
        import copy
        new_item = copy.deepcopy(item)
        
        aa_seq = new_item['aa_seq']
        codon_ids = new_item['codon_ids'].copy()
        
        # Randomly replace 10% of codons
        for i in range(len(codon_ids)):
            if random.random() < 0.1 and i < len(aa_seq):
                aa = aa_seq[i]
                if aa in SYN_CODON:
                    synonyms = SYN_CODON[aa]
                    if len(synonyms) > 1:
                        current_codon = ID2CODON.get(codon_ids[i])
                        alternatives = [c for c in synonyms if c != current_codon]
                        if alternatives:
                            new_codon = random.choice(alternatives)
                            if new_codon in CODON2ID:
                                codon_ids[i] = CODON2ID[new_codon]
        
        new_item['codon_ids'] = codon_ids
        
        # Update nucleotide sequence
        new_codons = [ID2CODON.get(cid, 'NNN') for cid in codon_ids]
        new_item['nucleotide_seq'] = self._codons_to_nucleotides(new_codons)
        new_item['trna_ids'] = codon_ids.copy()
        
        return new_item


# ==================== Collate Function (Enhanced) ====================
def collate_fn(batch):
    """Batch collation function (with length check)"""
    aa_ids, codon_ids, sp_ids, aux_features, nucleotide_seqs, trna_ids = zip(*batch)
    
    # Check length consistency for each sample
    valid_samples = []
    for i in range(len(aa_ids)):
        if len(aa_ids[i]) == len(codon_ids[i]) == len(trna_ids[i]):
            valid_samples.append(i)
        else:
            logging.getLogger(__name__).warning(
                f"Length mismatch in batch sample {i}: "
                f"aa={len(aa_ids[i])}, codon={len(codon_ids[i])}, trna={len(trna_ids[i])}"
            )
    
    # If no valid samples, return empty batch
    if not valid_samples:
        raise ValueError("No valid samples in batch")
    
    # Keep only valid samples
    aa_ids = [aa_ids[i] for i in valid_samples]
    codon_ids = [codon_ids[i] for i in valid_samples]
    sp_ids = [sp_ids[i] for i in valid_samples]
    aux_features = [aux_features[i] for i in valid_samples]
    nucleotide_seqs = [nucleotide_seqs[i] for i in valid_samples]
    trna_ids = [trna_ids[i] for i in valid_samples]
    
    # Pad sequences
    aa_padded = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in aa_ids],
        batch_first=True,
        padding_value=0
    )
    
    codon_padded = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in codon_ids],
        batch_first=True,
        padding_value=0
    )
    
    nucleotide_padded = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in nucleotide_seqs],
        batch_first=True,
        padding_value=0
    )
    
    trna_padded = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in trna_ids],
        batch_first=True,
        padding_value=0
    )
    
    # Convert to tensors
    sp_tensor = torch.tensor(sp_ids, dtype=torch.long)
    aux_tensor = torch.stack(aux_features)
    
    return aa_padded, codon_padded, sp_tensor, aux_tensor, nucleotide_padded, trna_padded

def collate_fn(batch):
    """Batch collation function"""
    aa_ids, codon_ids, sp_ids, aux_features, nucleotide_seqs, trna_ids = zip(*batch)
    
    # Pad sequences
    aa_padded = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in aa_ids],
        batch_first=True,
        padding_value=0
    )
    
    codon_padded = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in codon_ids],
        batch_first=True,
        padding_value=0
    )
    
    nucleotide_padded = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in nucleotide_seqs],
        batch_first=True,
        padding_value=0
    )
    
    trna_padded = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in trna_ids],
        batch_first=True,
        padding_value=0
    )
    
    # Convert to tensors
    sp_tensor = torch.tensor(sp_ids, dtype=torch.long)
    aux_tensor = torch.stack(aux_features)
    
    return aa_padded, codon_padded, sp_tensor, aux_tensor, nucleotide_padded, trna_padded



# ==================== Synonymous Codon Mask (Fixed) ====================
def apply_synonym_mask(logits: torch.Tensor, aa_ids: torch.Tensor) -> torch.Tensor:
    """
    Apply synonymous codon mask (fixed version, handles length mismatch)
    
    Args:
        logits: [B, L, V_codon] Model output
        aa_ids: [B, L] or [B, L-2] Amino acid ID sequence
    
    Returns:
        masked_logits: [B, L, V_codon]
    """
    B, L, V = logits.shape
    B_aa, L_aa = aa_ids.shape
    
    # ✅ Debug info (optional)
    logger = logging.getLogger(__name__)
    
    # ✅ Length alignment strategy
    if L_aa == L - 2:
        # aa_ids does not contain BOS/EOS, logits does
        # Add placeholders
        bos = torch.full((B, 1), 1, dtype=aa_ids.dtype, device=aa_ids.device)
        eos = torch.full((B, 1), 2, dtype=aa_ids.dtype, device=aa_ids.device)
        aa_ids_aligned = torch.cat([bos, aa_ids, eos], dim=1)
    elif L_aa == L:
        aa_ids_aligned = aa_ids
    else:
        # Other cases: truncate or pad
        logger.warning(f"Unexpected length: logits={L}, aa_ids={L_aa}, adjusting...")
        if L_aa > L:
            aa_ids_aligned = aa_ids[:, :L]
        else:
            pad_len = L - L_aa
            aa_ids_aligned = torch.cat([
                aa_ids,
                torch.zeros(B, pad_len, dtype=aa_ids.dtype, device=aa_ids.device)
            ], dim=1)
    
    # ✅ Initialize mask
    mask_value = get_mask_value(logits.dtype)
    mask = torch.ones_like(logits) * mask_value
    
    # ✅ Safe indexing - ensure no out of bounds
    for b in range(B):
        for i in range(min(L, aa_ids_aligned.size(1))):  # Key fix: use min()
            aa_id = aa_ids_aligned[b, i].item()
            
            if aa_id <= 2:  # Skip <pad>, <bos>, <eos>
                continue
            
            aa = ID2AA.get(aa_id)
            if aa and aa in SYN_CODON:
                valid_codons = SYN_CODON[aa]
                for codon in valid_codons:
                    if codon in CODON2ID:
                        codon_id = CODON2ID[codon]
                        if codon_id < V:  # Boundary check
                            mask[b, i, codon_id] = 0
    
    return logits + mask


def get_mask_value(dtype: torch.dtype) -> float:
    """Get dtype-safe mask value"""
    if dtype in [torch.float16, torch.bfloat16]:
        return -65000.0
    else:
        return -1e9

