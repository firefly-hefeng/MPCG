#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPCG-Codon inference script
Used for codon optimization of new sequences
"""

import argparse
import torch
import numpy as np
from typing import List, Dict

from CDT924 import (
    AA_TOKENS, CODON_TOKENS, aa_to_ids, ids_to_codons,
    BiologicalFeatureExtractor
)
from mpcg_model import MPCGCodon, FiveSpeciesCodonData, MPCGConfig


class CodonOptimizer:
    """Codon optimizer"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Initialize codon data
        self.codon_data = FiveSpeciesCodonData()
        
        # Create model
        config = checkpoint.get('config', MPCGConfig())
        
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
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Available species: {self.codon_data.species_list}")
    
    def optimize(
        self,
        protein_sequence: str,
        target_species: str,
        temperature: float = 1.0
    ) -> Dict:
        """
        Optimize codon usage for a protein sequence
        
        Args:
            protein_sequence: Amino acid sequence
            target_species: Target species
            temperature: Sampling temperature
        
        Returns:
            dict: Dictionary containing optimization results
        """
        # Ensure stop codon is present
        if not protein_sequence.endswith('*'):
            protein_sequence += '*'
        
        # Convert to IDs
        aa_ids = aa_to_ids(protein_sequence)
        aa_tensor = torch.tensor([aa_ids], dtype=torch.long).to(self.device)
        
        # Species ID
        try:
            sp_idx = self.codon_data.species_list.index(target_species)
        except ValueError:
            raise ValueError(f"Unknown species: {target_species}")
        
        sp_tensor = torch.tensor([sp_idx], dtype=torch.long).to(self.device)
        
        # Auxiliary features (simplified version)
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
        probs = torch.softmax(logits / temperature, dim=-1)
        codon_ids = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(1, -1)
        
        # Convert to codon sequence
        codon_ids_list = codon_ids[0].cpu().tolist()
        # Remove all special tokens: 0=<pad> 1=<bos> 2=<eos>; specific IDs depend on your vocabulary
        valid_ids = [i for i in codon_ids_list if i not in (0, 1, 2)]
        optimized_codons = ids_to_codons(valid_ids)  # Remove <bos> and <eos>
        
        # Build DNA sequence
        optimized_dna = ''.join(optimized_codons)
        
        # Calculate metrics
        cai_weights = self.codon_data.get_cai_weights(target_species)
        cai_values = [cai_weights.get(c, 0.5) for c in optimized_codons]
        cai = np.exp(np.mean([np.log(max(v, 1e-8)) for v in cai_values]))
        
        gc_content = self.feature_extractor.calculate_gc_content(optimized_dna)
        
        return {
            'protein_sequence': protein_sequence,
            'optimized_dna': optimized_dna,
            'optimized_codons': optimized_codons,
            'target_species': target_species,
            'cai': cai,
            'gc_content': gc_content,
            'length': len(optimized_codons)
        }


def main():
    parser = argparse.ArgumentParser(description="MPCG-Codon Inference")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--protein', type=str, required=True,
                       help='Protein sequence (amino acids)')
    parser.add_argument('--species', type=str, required=True,
                       choices=['Homo sapiens', 'Mus musculus',
                               'Escherichia coli', 'Saccharomyces cerevisiae',
                               'Pichia angusta'],
                       help='Target species')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--output', type=str, default='optimized.fasta',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = CodonOptimizer(args.checkpoint)
    
    # Optimize sequence
    result = optimizer.optimize(
        args.protein,
        args.species,
        args.temperature
    )
    
    # Print results
    print("\n" + "="*80)
    print("Optimization Results")
    print("="*80)
    print(f"Target Species: {result['target_species']}")
    print(f"Protein Length: {result['length']} aa")
    print(f"CAI: {result['cai']:.4f}")
    print(f"GC Content: {result['gc_content']*100:.2f}%")
    print(f"\nOptimized DNA Sequence:")
    print(result['optimized_dna'])
    
    # Save to file
    with open(args.output, 'w') as f:
        f.write(f">Optimized for {result['target_species']} | CAI={result['cai']:.4f}\n")
        # 60 characters per line
        for i in range(0, len(result['optimized_dna']), 60):
            f.write(result['optimized_dna'][i:i+60] + '\n')
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
