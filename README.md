<div align="center">

# 🧬 MPCG-Codon

**Multi-Modal Physics-Constrained Guided Codon Optimization**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Bioinformatics](https://img.shields.io/badge/Bioinformatics-Biopython-green.svg)](https://biopython.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

</div>

<p align="center">
  <img src="docs/structure.png" alt="MPCG Architecture" width="95%">
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Inference](#-inference)
- [SPEA Fine-Tuning](#-spea-fine-tuning)
- [Project Structure](#-project-structure)
- [Citation](#-citation)

---

## 🔬 Overview

**MPCG-Codon** is a deep-learning framework for intelligent codon optimization that goes beyond traditional frequency-based methods. By integrating **multi-modal biological priors**, **physics-constrained attention mechanisms**, and **neural RNA structure prediction**, MPCG-Codon generates codon sequences optimized for heterologous protein expression across multiple host organisms.

The framework currently supports **five model organisms**:
- 🧑 *Homo sapiens* (Human)
- 🐭 *Mus musculus* (Mouse)
- 🦠 *Escherichia coli* (E. coli)
- 🍞 *Saccharomyces cerevisiae* (Yeast)
- 🧫 *Pichia angusta* (Methylotrophic yeast)

---

## ✨ Key Features

### 1. Physics-Constrained Attention
MPCG-Codon introduces a novel **Physics-Constrained Attention** mechanism that injects biological structure directly into the Transformer architecture:
- **RNA secondary-structure pairing matrices** guide local attention patterns
- **Ribosomal pause probabilities** modulate positional importance
- **Translation-time-aware positional encodings** replace naive sequence-position embeddings

### 2. Neural RNA Folder
An embedded **NeuralRNAFolder** predicts minimum free energy (MFE) and base-pairing probabilities from nucleotide sequences, enabling structure-aware loss terms during training.

### 3. Translation Dynamics Model
A dedicated submodule estimates **tRNA abundance–mediated elongation rates** and **ribosomal pause probabilities**, allowing the model to optimize for smooth translation kinetics.

### 4. Biologically Informed Multi-Objective Loss
The training objective combines eight complementary loss terms:
| Loss Component | Description |
|---------------|-------------|
| `CE` | Cross-entropy against natural coding sequences |
| `CAI` | Codon Adaptation Index alignment |
| `RSCU` | Relative Synonymous Codon Usage divergence |
| `GC` | GC-content regularization |
| `Structure` | RNA folding energy guidance |
| `Dynamics` | Ribosomal pause penalty |
| `Rare Codon` | Conservation of rare-codon clusters |
| `Manufacturability` | Repeat & homopolymer avoidance |

### 5. Secretion Protein Expression Adapter (SPEA)
For **E. coli secretion proteins**, MPCG provides a specialized fine-tuning stack:
- **Signal-peptide-aware adapter** with cleavage-site prediction
- **Disulfide-bond optimization** module with redox-environment modeling
- **Solubility optimization** with aggregation-prediction and tag recommendations

---

## 🏗 Architecture

<p align="center">
  <img src="docs/structure.png" alt="MPCG Architecture" width="90%">
</p>

The architecture consists of three hierarchical levels:

1. **Base Encoder** (`MPCG-BaseCodonFormer.py`) — Sparse-attention Transformer with multi-scale convolutions and biological feature extraction.
2. **Core Model** (`MPCG-CoreModel.py`) — Physics-constrained layers, neural RNA folder, translation dynamics, and five-species codon priors.
3. **SPEA Adapter** (`MPCG-SPEA-Modules.py`) — Task-specific modules for signal peptides, disulfide bonds, and solubility.

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/firefly-hefeng/MPCG.git
cd MPCG

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Optional: ViennaRNA
For exact RNA MFE calculations, install ViennaRNA via Conda:
```bash
conda install -c bioconda viennarna
```
If ViennaRNA is unavailable, the model falls back to the internal **NeuralRNAFolder**.

---

## ⚡ Quick Start

### Single-sequence prediction
```bash
python pridict.py \
  --checkpoint checkpoints/best.pt \
  --species "Escherichia coli" \
  --protein "MKTLLIL*" \
  --output optimized.fasta
```

### Batch prediction from FASTA
```bash
python pridict.py \
  --checkpoint checkpoints/best.pt \
  --species "Saccharomyces cerevisiae" \
  --fasta input_sequences.fasta \
  --output_format fasta \
  --output predictions.fasta
```

### Batch prediction from CSV
```bash
python pridict.py \
  --checkpoint checkpoints/best.pt \
  --species "Pichia angusta" \
  --csv sequences.csv \
  --output predictions.csv \
  --output_format csv
```

---

## 🏋️ Training

### Prepare your data
Your CSV should contain at least three columns:
- `RefSeq_aa` — Amino-acid sequences (include `*` for stop codons)
- `RefSeq_nn` — Native nucleotide sequences (coding DNA)
- `Organism` — Host organism name (must match one of the five supported species)

### Run training
```bash
python train.py \
  --data_csv your_data.csv \
  --d_model 512 \
  --n_layers 12 \
  --n_heads 8 \
  --batch_size 8 \
  --lr 1e-4 \
  --epochs 50 \
  --save_dir ./checkpoints \
  --wandb
```

### Training options
| Argument | Default | Description |
|----------|---------|-------------|
| `--d_model` | 512 | Transformer embedding dimension |
| `--n_layers` | 12 | Number of Transformer layers |
| `--n_heads` | 8 | Attention heads |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-4 | Peak learning rate |
| `--warmup_steps` | 4000 | Linear warmup steps |
| `--weight_ce` | 1.0 | Cross-entropy weight |
| `--weight_cai` | 0.4 | CAI loss weight |
| `--weight_rscu` | 0.3 | RSCU loss weight |
| `--wandb` | — | Enable Weights & Biases logging |

---

## 🔮 Inference

Low-level inference via `mpcg_inference.py`:
```bash
python mpcg_inference.py \
  --checkpoint checkpoints/best.pt \
  --protein "MKTLLIL*" \
  --species "Escherichia coli" \
  --temperature 1.0 \
  --output result.fasta
```

For programmatic usage:
```python
from pridict import CodonPredictor

predictor = CodonPredictor("checkpoints/best.pt")
result = predictor.predict_single("MKTLLIL*", "Escherichia coli")
print(result['optimized_dna'])
```

---

## 🧪 SPEA Fine-Tuning

The **Secretion Protein Expression Adapter (SPEA)** is designed for *E. coli* periplasmic expression of complex proteins (e.g., antibodies, disulfide-rich scaffolds).

```bash
python MPCG-SPEA-Finetune.py \
  --pretrained_model checkpoints/best.pt \
  --data_file secretion_data.csv \
  --output_dir ./spea_checkpoints \
  --epochs 20 \
  --batch_size 4 \
  --freeze_base \
  --n_augment 50
```

SPEA modules include:
- `SecretionSignalAdapter` — Optimizes signal-peptide coding regions and predicts cleavage sites
- `DisulfideBondAwareModule` — Enhances cysteine-adjacent codons for correct oxidative folding
- `SolubilityOptimizationModule` — Predicts aggregation risk and recommends solubility tags (MBP, SUMO, TRX, GST)

---

## 📁 Project Structure

```
MPCG/
├── docs/
│   └── structure.png              # Architecture diagram
├── MPCG-BaseCodonFormer.py      # Sparse-attention base encoder
├── MPCG-BioPriorLoss.py         # Biologically informed loss functions
├── MPCG-CoreModel.py            # Core MPCG model & 5-species priors
├── MPCG-SPEA-Modules.py         # SPEA adapter modules
├── MPCG-SPEA-Finetune.py        # SPEA fine-tuning script
├── MPCG-SPEA-DataPrep.py        # SPEA data preparation (placeholder)
├── mpcg_inference.py            # Standalone inference script
├── pridict.py                   # One-click prediction CLI
├── train.py                     # Main training pipeline
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 📚 Citation

If you use MPCG-Codon in your research, please cite:

```bibtex
@software{mpcg_codon,
  author = {firefly-hefeng},
  title = {MPCG-Codon: Multi-Modal Physics-Constrained Guided Codon Optimization},
  url = {https://github.com/firefly-hefeng/MPCG},
  year = {2025}
}
```

---

<div align="center">

**Made with ❤️ for synthetic biology and protein engineering.**

</div>
