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
