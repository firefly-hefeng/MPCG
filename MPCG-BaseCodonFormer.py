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

# ==================== Enhanced Model Architecture ====================
class SparseTransformerLayer(nn.Module):
    """Sparse Transformer layer"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.sparse_attn = SparseAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # attention sub-layer
        attn_out = self.sparse_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # feed-forward sub-layer
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


class EnhancedCodonFormerE(nn.Module):
    """Enhanced codon optimization Transformer"""
    
    def __init__(self, v_aa, v_cd, v_sp, aux_dim=16, d=512, depth=6, heads=8, drop=0.1):
        super().__init__()
        self.d_model = d
        
        # embedding layers
        self.e_aa = nn.Embedding(v_aa, d, padding_idx=0)
        self.e_sp = nn.Embedding(v_sp, d)
        self.e_aux = nn.Linear(aux_dim, d)
        
        # positional encoding
        self.pos_enc = PosEnc(d)
        
        # multi-scale feature extraction
        self.local_conv = nn.Conv1d(d, d, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # sparse attention encoder layers
        self.layers = nn.ModuleList([
            SparseTransformerLayer(d, heads, drop) for _ in range(depth)
        ])
        
        # biological feature predictors
        self.gc_predictor = nn.Linear(d, 1)
        self.structure_predictor = nn.Linear(d, 1)
        
        # output projection
        self.proj = nn.Linear(d, v_cd)
        
        # feature extractor
        self.feature_extractor = BiologicalFeatureExtractor()
        
    def forward(self, aa, mask, sp, aux, return_features=False):
        B, L = aa.shape
        
        # base embedding
        h = self.e_aa(aa) + self.pos_enc(aa)
        
        # species and auxiliary feature embeddings
        h = h + self.e_sp(sp).unsqueeze(1).expand(-1, L, -1)
        h = h + self.e_aux(aux).unsqueeze(1).expand(-1, L, -1)
        
        # multi-scale features
        h_conv = self.local_conv(h.transpose(1, 2)).transpose(1, 2)
        h_global = self.global_pool(h.transpose(1, 2)).transpose(1, 2).expand(-1, L, -1)
        h = h + 0.1 * h_conv + 0.05 * h_global
        
        # sparse attention encoder
        for layer in self.layers:
            h = layer(h, mask)
        
        # main output: codon prediction
        logits = self.proj(h)
        
        if return_features:
            # biological feature predictions
            gc_pred = torch.sigmoid(self.gc_predictor(h)).squeeze(-1)
            structure_pred = self.structure_predictor(h).squeeze(-1)
            
            return logits, {
                'gc_content': gc_pred,
                'structure_energy': structure_pred,
                'hidden_states': h
            }
        
        return logits


# ==================== Multi-Objective Loss Function ====================
class MultiObjectiveLoss(nn.Module):
    """Multi-objective loss function"""
    
    def __init__(self, weights=None, device='cpu'):
        super().__init__()
        self.weights = weights or {
            'ce': 1.0,
            'cai': 0.3,
            'rscu': 0.2,
            'gc': 0.1,
            'structure': 0.05,
            'manufacturability': 0.05
        }
        self.device = device
        self.feature_extractor = BiologicalFeatureExtractor()
        
    def forward(self, logits, targets, aa_seq, sp_id, aux_features=None, predictions=None):
        """
        Compute multi-objective loss
        
        Args:
            logits: model predicted logits [B, L, V]
            targets: true codon sequence [B, L]
            aa_seq: amino acid sequence [B, L]
            sp_id: species ID [B]
            aux_features: auxiliary features [B, aux_dim]
            predictions: additional predicted feature dictionary
        """
        losses = {}
        
        # 1. cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=0
        )
        losses['ce'] = ce_loss
        
        # 2. CAI loss
        pred_codons = logits.argmax(dim=-1)
        cai_loss = self._calculate_cai_loss(pred_codons, targets, sp_id)
        losses['cai'] = cai_loss
        
        # 3. RSCU loss
        rscu_loss = self._calculate_rscu_loss(pred_codons, targets)
        losses['rscu'] = rscu_loss
        
        # 4. GC content loss
        if predictions and 'gc_content' in predictions:
            gc_loss = self._calculate_gc_loss(pred_codons, predictions['gc_content'])
            losses['gc'] = gc_loss
        else:
            losses['gc'] = torch.tensor(0.0, device=self.device)
        
        # 5. structure loss
        if predictions and 'structure_energy' in predictions:
            structure_loss = self._calculate_structure_loss(predictions['structure_energy'])
            losses['structure'] = structure_loss
        else:
            losses['structure'] = torch.tensor(0.0, device=self.device)
        
        # 6. manufacturability loss
        manufact_loss = self._calculate_manufacturability_loss(pred_codons)
        losses['manufacturability'] = manufact_loss
        
        # total loss
        total_loss = sum(self.weights[k] * v for k, v in losses.items())
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def _calculate_cai_loss(self, pred_codons, target_codons, sp_id):
        """CAI loss calculation"""
        batch_size = pred_codons.size(0)
        cai_losses = []
        
        for i in range(batch_size):
            pred_seq = pred_codons[i][pred_codons[i] != 0]  # remove padding
            target_seq = target_codons[i][target_codons[i] != 0]
            
            # simplified CAI calculation (in practice, species-specific weights should be used)
            pred_cai = self._simple_cai(pred_seq)
            target_cai = self._simple_cai(target_seq)
            
            cai_losses.append(F.mse_loss(pred_cai, target_cai))
        
        return torch.stack(cai_losses).mean()
    
    def _simple_cai(self, codon_ids):
        """Simplified CAI calculation"""
        # using a simplified version here; in practice, real CAI weights should be loaded per species
        weights = torch.ones_like(codon_ids, dtype=torch.float) * 0.5
        weights[codon_ids > 30] = 0.8  # assume high-ID codons are optimized
        return weights.mean()
    
    def _calculate_rscu_loss(self, pred_codons, target_codons):
        """RSCU divergence loss"""
        # compute RSCU distribution difference between predicted and target sequences
        pred_dist = self._get_codon_distribution(pred_codons)
        target_dist = self._get_codon_distribution(target_codons)
        
        # KL divergence
        kl_div = F.kl_div(
            F.log_softmax(pred_dist, dim=-1),
            F.softmax(target_dist, dim=-1),
            reduction='batchmean'
        )
        return kl_div
    
    def _get_codon_distribution(self, codon_ids):
        """Get codon distribution"""
        batch_size, seq_len = codon_ids.shape
        vocab_size = 61  # codon vocabulary size
        
        distributions = []
        for i in range(batch_size):
            valid_codons = codon_ids[i][codon_ids[i] != 0]
            hist = torch.histc(valid_codons.float(), bins=vocab_size, min=1, max=vocab_size)
            distributions.append(hist)
        
        return torch.stack(distributions)
    
    def _calculate_gc_loss(self, pred_codons, gc_pred):
        """GC content loss"""
        target_gc = 0.5  # target GC content
        gc_loss = F.mse_loss(gc_pred.mean(dim=1), 
                           torch.full((gc_pred.size(0),), target_gc, device=self.device))
        return gc_loss
    
    def _calculate_structure_loss(self, structure_pred):
        """RNA structure loss"""
        # encourage lower folding energy (more stable structure)
        target_energy = -10.0  # target folding energy
        structure_loss = F.relu(structure_pred.mean(dim=1) - target_energy).mean()
        return structure_loss
    
    def _calculate_manufacturability_loss(self, pred_codons):
        """Manufacturability loss - avoid repeated sequences and extreme GC"""
        batch_size = pred_codons.size(0)
        manufact_losses = []
        
        for i in range(batch_size):
            seq = pred_codons[i][pred_codons[i] != 0]
            
            # 1. repeat sequence penalty
            repeat_penalty = self._calculate_repeat_penalty(seq)
            
            # 2. GC content variance penalty (overly localized GC distribution)
            gc_var_penalty = self._calculate_gc_variance_penalty(seq)
            
            manufact_losses.append(repeat_penalty + gc_var_penalty)
        
        return torch.stack(manufact_losses).mean()
    
    def _calculate_repeat_penalty(self, sequence):
        """Calculate repeat sequence penalty"""
        if len(sequence) < 6:
            return torch.tensor(0.0, device=self.device)
        
        # check triplet repeats
        repeats = 0
        for i in range(len(sequence) - 5):
            if torch.equal(sequence[i:i+3], sequence[i+3:i+6]):
                repeats += 1
        
        return torch.tensor(repeats / max(1, len(sequence) - 5), device=self.device)
    
    def _calculate_gc_variance_penalty(self, sequence):
        """GC content variance penalty"""
        if len(sequence) < 10:
            return torch.tensor(0.0, device=self.device)
        
        # simplified GC variance calculation
        # in practice, codon IDs need to be converted back to nucleotide sequences
        gc_variance = torch.var(sequence.float()) / len(sequence)
        return torch.relu(gc_variance - 1.0)  # penalize excessive variance

