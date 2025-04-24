#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPCG-Codon: Multi-Modal Physics-Constrained Guided Codon Optimization
Full model implementation (with five-species prior data)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
from Bio.Data import CodonTable

def get_mask_value(dtype: torch.dtype) -> float:
    """
    Get a mask value safe for the current dtype.
    
    Args:
        dtype: Tensor data type.
    
    Returns:
        Safe mask value.
    """
    if dtype in [torch.float16, torch.bfloat16]:
        # float16/bfloat16 range is approximately ±65504
        return -65000.0
    else:
        # float32/float64 can use a larger value
        return -1e9

# ==================== Config Class ====================
@dataclass
class MPCGConfig:
    """MPCG-Codon model configuration."""
    # Basic parameters
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 2048
    
    # Sparse attention parameters
    local_window: int = 64
    global_ratio: float = 0.1
    
    # Physics constraint parameters
    lambda_energy: float = 0.1
    lambda_pause: float = 0.05
    lambda_gc: float = 0.01
    
    # Multi-modal parameters
    aux_dim: int = 64
    structure_dim: int = 128
    dynamics_dim: int = 64
    
    # Other parameters
    use_neural_folder: bool = True


# ==================== Five-Species Codon Usage Data ====================
