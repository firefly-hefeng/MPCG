#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEA fine-tuning training script
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import argparse
from pathlib import Path

# Import necessary modules
from MPCG_SPEA_Modules import (
    SPEAConfig, SPEAFineTuner, SignalPeptideDB
)
from MPCG_CoreModel import MPCGCodon, MPCGConfig, FiveSpeciesCodonData
from MPCG_BaseCodonFormer import AA2ID, CODON2ID, aa_to_ids, codon_to_ids
from MPCG_SPEA_DataPrep import SecretionProteinDataPreparator

class SecretionProteinDataset(Dataset):
    """Secretion protein dataset"""
    
    def __init__(self, csv_file: str, max_length: int = 500):
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Convert to IDs
        aa_ids = aa_to_ids(row['aa_sequence'])
        
        # Process codons
        if isinstance(row['initial_codons'], str):
            initial_codons = json.loads(row['initial_codons'])
        else:
            initial_codons = row['initial_codons']
        codon_ids = codon_to_ids(initial_codons)
        
        # Position labels
        if isinstance(row['position_labels'], str):
            position_labels = json.loads(row['position_labels'])
        else:
            position_labels = row['position_labels']
        
        # Features
        features = json.loads(row['features'])
        
        # Convert to tensors
        aa_tensor = torch.tensor(aa_ids[:self.max_length], dtype=torch.long)
        codon_tensor = torch.tensor(codon_ids[:self.max_length], dtype=torch.long)
        position_tensor = torch.tensor(position_labels[:self.max_length], dtype=torch.long)
        
        # Species ID (E. coli = 2)
        species_id = torch.tensor(2, dtype=torch.long)
        
        # Auxiliary features
        aux_features = self._extract_aux_features(features)
        
        return {
            'aa_ids': aa_tensor,
            'codon_ids': codon_tensor,
            'position_labels': position_tensor,
            'species_id': species_id,
            'aux_features': aux_features,
            'protein_id': row['protein_id']
        }
    
    def _extract_aux_features(self, features: Dict) -> torch.Tensor:
        """Extract auxiliary features"""
        aux = torch.zeros(64)
        
        # Basic features
        aux[0] = features.get('size', 20000) / 100000  # Normalization
        aux[1] = float(features.get('has_disulfide', False))
        aux[2] = features.get('n_cys', 0) / 10
        aux[3] = features.get('glycosylation_sites', 0) / 5
        
        return aux


def collate_fn(batch):
    """Batch processing function"""
    # Find maximum length
    max_len = max(item['aa_ids'].size(0) for item in batch)
    
    # Padding
    aa_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    codon_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    position_labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    species_ids = torch.zeros(len(batch), dtype=torch.long)
    aux_features = torch.zeros(len(batch), 64)
    
    protein_ids = []
    
    for i, item in enumerate(batch):
        length = item['aa_ids'].size(0)
        aa_ids[i, :length] = item['aa_ids']
        codon_ids[i, :length] = item['codon_ids']
        position_labels[i, :length] = item['position_labels']
        species_ids[i] = item['species_id']
        aux_features[i] = item['aux_features']
        protein_ids.append(item['protein_id'])
    
    return {
        'aa_ids': aa_ids,
        'codon_ids': codon_ids,
        'position_labels': position_labels,
        'species_ids': species_ids,
        'aux_features': aux_features,
        'protein_ids': protein_ids
    }


def train_spea(args):
    """Train SPEA model"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained model
    print("Loading pretrained model...")
    checkpoint = torch.load(args.pretrained_model, map_location=device)
    
    # Create base model
    base_config = checkpoint.get('config', MPCGConfig())
    codon_data = FiveSpeciesCodonData()
    
    base_model = MPCGCodon(
        config=base_config,
        v_aa=len(AA2ID),
        v_codon=len(CODON2ID),
        v_species=6,
        codon_data=codon_data
    ).to(device)
    
    # Load pretrained weights
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create SPEA config
    spea_config = SPEAConfig(
        d_model=base_config.d_model,
        n_heads=base_config.n_heads,
        dropout=0.1
    )
    
    # Create SPEA fine-tuner
    print("Creating SPEA fine-tuner...")
    model = SPEAFineTuner(
        base_model=base_model,
        config=spea_config,
        codon_data=codon_data,
        freeze_base=args.freeze_base
    ).to(device)
    
    # Prepare data
    print("Preparing data...")
    if not os.path.exists(args.data_file):
        print("Generating training data...")
        preparator = SecretionProteinDataPreparator()
        base_data = preparator.prepare_training_data()
        augmented_data = preparator.create_augmented_dataset(base_data, n_augment=args.n_augment)
        augmented_data.to_csv(args.data_file, index=False)
    
    # Create dataset
    dataset = SecretionProteinDataset(args.data_file)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Training loop
