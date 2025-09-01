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
