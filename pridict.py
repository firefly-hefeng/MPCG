#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPCG-Codon One-Click Prediction Script
Supports single sequence and batch prediction
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from MPCG_BaseCodonFormer import (
    AA_TOKENS, CODON_TOKENS, aa_to_ids, ids_to_codons,
    BiologicalFeatureExtractor
)

from MPCG_CoreModel import (
    MPCGCodon, MPCGConfig, FiveSpeciesCodonData
)


class CodonPredictor:
    """Codon Predictor"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        temperature: float = 1.0
    ):
        """
        Initialize predictor
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device ('cuda' or 'cpu')
            temperature: Sampling temperature
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        
        print(f"Loading model from {checkpoint_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Initialize codon data
        self.codon_data = FiveSpeciesCodonData()
        
        # Create model
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Try to reconstruct config from args
            args = checkpoint.get('args', {})
            config = MPCGConfig(
                d_model=args.get('d_model', 512),
                n_layers=args.get('n_layers', 12),
                n_heads=args.get('n_heads', 8),
                d_ff=args.get('d_ff', 2048),
                dropout=args.get('dropout', 0.1),
                max_seq_len=args.get('max_seq_len', 2048)
            )
        
        self.model = MPCGCodon(
            config=config,
            v_aa=len(AA_TOKENS),
            v_codon=len(CODON_TOKENS),
            v_species=len(self.codon_data.species_list) + 1,
            codon_data=self.codon_data
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.feature_extractor = BiologicalFeatureExtractor()
        
        print(f"Model loaded successfully!")
        print(f"Available species: {self.codon_data.species_list}")
        print(f"Using device: {self.device}")
    
    def predict_single(
        self,
        protein_sequence: str,
        target_species: str,
        return_probs: bool = False
    ) -> Dict:
        """
        Predict a single protein sequence
        
        Args:
            protein_sequence: Amino acid sequence
            target_species: Target species
            return_probs: Whether to return probability distribution
        
        Returns:
            Prediction result dictionary
        """
        # Ensure terminator is present
        if not protein_sequence.endswith('*'):
            protein_sequence += '*'
        
        # Convert to IDs
        aa_ids = aa_to_ids(protein_sequence)
        aa_tensor = torch.tensor([aa_ids], dtype=torch.long).to(self.device)
        
        # Species ID
        try:
            sp_idx = self.codon_data.species_list.index(target_species)
        except ValueError:
            raise ValueError(
                f"Unknown species: {target_species}. "
                f"Available: {self.codon_data.species_list}"
            )
        
        sp_tensor = torch.tensor([sp_idx], dtype=torch.long).to(self.device)
        
        # Auxiliary features (simplified)
        aux_features = torch.zeros(1, 64).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits, features = self.model(
                aa_ids=aa_tensor,
                mask=(aa_tensor == 0),
                species_ids=sp_tensor,
                aux_features=aux_features,
                return_features=True
            )
        
        # Sample codons
        probs = torch.softmax(logits / self.temperature, dim=-1)
        codon_ids = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(1, -1)
        
        # Convert to codon sequence
        codon_ids_list = codon_ids[0].cpu().tolist()
        valid_ids = [i for i in codon_ids_list if i not in (0, 1, 2)]
        optimized_codons = ids_to_codons(valid_ids)
        
        # Build DNA sequence
        optimized_dna = ''.join(optimized_codons)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            optimized_codons, target_species, optimized_dna
        )
        
        result = {
            'protein_sequence': protein_sequence,
            'optimized_dna': optimized_dna,
            'optimized_codons': optimized_codons,
            'target_species': target_species,
            'metrics': metrics
        }
        
        if return_probs:
            result['probabilities'] = probs[0].cpu().numpy()
        
        return result
