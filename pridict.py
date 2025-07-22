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
    
    def predict_batch(
        self,
        sequences: List[str],
        target_species: str,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Batch prediction
        
        Args:
            sequences: List of amino acid sequences
            target_species: Target species
            show_progress: Whether to show progress bar
        
        Returns:
            List of prediction results
        """
        results = []
        
        iterator = tqdm(sequences) if show_progress else sequences
        
        for seq in iterator:
            try:
                result = self.predict_single(seq, target_species)
                results.append(result)
            except Exception as e:
                print(f"Error processing sequence: {e}")
                results.append({
                    'protein_sequence': seq,
                    'error': str(e)
                })
        
        return results
    
    def _calculate_metrics(
        self,
        codons: List[str],
        species: str,
        dna_seq: str
    ) -> Dict:
        """Calculate optimization metrics"""
        metrics = {}
        
        # CAI
        cai_weights = self.codon_data.get_cai_weights(species)
        cai_values = [cai_weights.get(c, 0.5) for c in codons]
        metrics['cai'] = float(np.exp(np.mean([np.log(max(v, 1e-8)) for v in cai_values])))
        
        # GC content
        metrics['gc_content'] = float(self.feature_extractor.calculate_gc_content(dna_seq))
        
        # RSCU
        rscu_dict = self.feature_extractor.calculate_rscu(codons)
        metrics['mean_rscu'] = float(np.mean(list(rscu_dict.values())))
        
        # Sequence length
        metrics['length'] = len(codons)
        
        # Rare codon ratio
        rscu_ref = self.codon_data.get_rscu(species)
        rare_count = sum(1 for c in codons if rscu_ref.get(c, 1.0) < 0.6)
        metrics['rare_codon_ratio'] = float(rare_count / len(codons))
        
        return metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MPCG-Codon Prediction Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--species', type=str, required=True,
        choices=[
            'Homo sapiens',
            'Mus musculus',
            'Escherichia coli',
            'Saccharomyces cerevisiae',
            'Pichia angusta'
        ],
        help='Target species'
    )
    
    # Input methods (choose one)'
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--protein', type=str,
        help='Single protein sequence (amino acids)'
    )
    input_group.add_argument(
        '--fasta', type=str,
        help='Path to FASTA format file'
    )
    input_group.add_argument(
        '--csv', type=str,
        help='Path to CSV file (must contain protein_sequence column)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output', type=str, default='predictions.fasta',
        help='Output file path'
    )
    parser.add_argument(
        '--output_format', type=str, default='fasta',
        choices=['fasta', 'csv', 'json'],
        help='Output format'
    )
    
    # Prediction arguments
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Sampling temperature (higher values increase diversity)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='Computing device'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size (currently only 1 is supported)'
    )
    
    # Other arguments
    parser.add_argument(
        '--verbose', action='store_true',
        help='Show verbose information'
    )
    
    return parser.parse_args()


def load_sequences_from_fasta(fasta_path: str) -> List[Dict]:
    """Load sequences from FASTA file"""
    sequences = []
    current_header = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append({
                        'header': current_header,
                        'sequence': ''.join(current_seq)
                    })
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        
        # Last sequence
        if current_header is not None:
            sequences.append({
                'header': current_header,
                'sequence': ''.join(current_seq)
            })
    
    return sequences


def save_results_fasta(results: List[Dict], output_path: str):
    """Save results in FASTA format"""
    with open(output_path, 'w') as f:
        for i, result in enumerate(results):
            if 'error' in result:
                continue
            
            header = (
                f">Optimized_{i+1} | Species={result['target_species']} | "
                f"CAI={result['metrics']['cai']:.4f} | "
                f"GC={result['metrics']['gc_content']*100:.2f}%"
            )
            f.write(header + '\n')
            
            # 60 characters per line
            dna = result['optimized_dna']
            for j in range(0, len(dna), 60):
                f.write(dna[j:j+60] + '\n')


def save_results_csv(results: List[Dict], output_path: str):
    """Save results in CSV format"""
    data = []
    for i, result in enumerate(results):
        if 'error' in result:
            data.append({
                'id': i + 1,
                'protein_sequence': result.get('protein_sequence', ''),
                'error': result['error']
            })
        else:
            row = {
                'id': i + 1,
                'protein_sequence': result['protein_sequence'],
                'optimized_dna': result['optimized_dna'],
                'target_species': result['target_species'],
                **{f"metric_{k}": v for k, v in result['metrics'].items()}
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


def save_results_json(results: List[Dict], output_path: str):
    """Save results in JSON format"""
    # Convert numpy types to Python native types
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    clean_results = []
    for result in results:
        clean_result = {}
        for k, v in result.items():
            if isinstance(v, dict):
                clean_result[k] = {kk: convert_types(vv) for kk, vv in v.items()}
            else:
                clean_result[k] = convert_types(v)
        clean_results.append(clean_result)
    
    with open(output_path, 'w') as f:
        json.dump(clean_results, f, indent=2)


def main():
    """Main prediction workflow"""
    args = parse_args()
    
    print("="*80)
    print("MPCG-Codon Prediction Script")
    print("="*80)
    
    # Create predictor
    predictor = CodonPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        temperature=args.temperature
    )
    
    # Load input sequences
    sequences = []
    
    if args.protein:
        sequences = [{'header': 'Input', 'sequence': args.protein}]
    
    elif args.fasta:
        print(f"Loading sequences from {args.fasta}...")
        sequences = load_sequences_from_fasta(args.fasta)
        print(f"Loaded {len(sequences)} sequences")
    
    elif args.csv:
        print(f"Loading sequences from {args.csv}...")
        df = pd.read_csv(args.csv)
        if 'protein_sequence' not in df.columns:
            print("Error: CSV must contain 'protein_sequence' column")
            sys.exit(1)
        sequences = [
            {'header': f"Seq_{i+1}", 'sequence': seq}
            for i, seq in enumerate(df['protein_sequence'])
        ]
        print(f"Loaded {len(sequences)} sequences")
    
    # Run prediction
    print(f"\nPredicting for species: {args.species}")
    print("-"*80)
    
    seq_list = [s['sequence'] for s in sequences]
    results = predictor.predict_batch(
        seq_list,
        args.species,
        show_progress=True
    )
    
    # Print statistics
    print("\n" + "="*80)
    print("Prediction Summary")
    print("="*80)
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"Total sequences: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_cai = np.mean([r['metrics']['cai'] for r in successful])
        avg_gc = np.mean([r['metrics']['gc_content'] for r in successful])
        print(f"\nAverage CAI: {avg_cai:.4f}")
        print(f"Average GC content: {avg_gc*100:.2f}%")
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    
    if args.output_format == 'fasta':
        save_results_fasta(results, args.output)
    elif args.output_format == 'csv':
        save_results_csv(results, args.output)
    elif args.output_format == 'json':
        save_results_json(results, args.output)
    
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    main()
