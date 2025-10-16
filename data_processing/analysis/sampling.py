#!/usr/bin/env python3
"""
JSONL Sampling Module

This module provides utilities for sampling and inspecting JSONL files,
useful for data exploration and quality assessment.

Author: CURV Team
"""

import json
import os
import sys
import random
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_io import load_jsonl, save_jsonl, validate_jsonl_format

class JSONLSampler:
    """Utility class for sampling JSONL files with various strategies."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the sampler.
        
        Args:
            random_seed: Seed for random sampling reproducibility
        """
        if random_seed is not None:
            random.seed(random_seed)
    
    def sample_first_n(self, jsonl_path: str, num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Sample the first n entries from a JSONL file.
        
        Args:
            jsonl_path: Path to the JSONL file
            num_samples: Number of samples to extract
            
        Returns:
            List of sample JSON objects
        """
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"File not found: {jsonl_path}")
        
        samples = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                    
                try:
                    json_obj = json.loads(line.strip())
                    samples.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON at line {i+1}: {e}")
        
        return samples
    
    def sample_random(self, jsonl_path: str, num_samples: int = 5, 
                     total_lines: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Sample random entries from a JSONL file.
        
        Args:
            jsonl_path: Path to the JSONL file
            num_samples: Number of samples to extract
            total_lines: Total number of lines (if known, for efficiency)
            
        Returns:
            List of sample JSON objects
        """
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"File not found: {jsonl_path}")
        
        # Count total lines if not provided
        if total_lines is None:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
        
        if num_samples >= total_lines:
            print(f"Warning: Requested {num_samples} samples but file only has {total_lines} lines")
            return self.sample_first_n(jsonl_path, total_lines)
        
        # Generate random line numbers
        sample_lines = sorted(random.sample(range(total_lines), num_samples))
        
        samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i in sample_lines:
                    try:
                        json_obj = json.loads(line.strip())
                        samples.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON at line {i+1}: {e}")
                
                # Early exit if we've collected all samples
                if len(samples) == num_samples:
                    break
        
        return samples
    
    def sample_stratified(self, jsonl_path: str, num_samples: int = 5, 
                         stratify_key: str = 'patient_id') -> List[Dict[str, Any]]:
        """
        Sample entries stratified by a specific key.
        
        Args:
            jsonl_path: Path to the JSONL file
            num_samples: Number of samples to extract
            stratify_key: Key to stratify by
            
        Returns:
            List of sample JSON objects
        """
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"File not found: {jsonl_path}")
        
        # Group entries by stratify key
        groups = {}
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    json_obj = json.loads(line.strip())
                    key_value = json_obj.get(stratify_key)
                    
                    if key_value is not None:
                        if key_value not in groups:
                            groups[key_value] = []
                        groups[key_value].append(json_obj)
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON at line {i+1}: {e}")
        
        if not groups:
            print(f"Warning: No valid entries found with key '{stratify_key}'")
            return []
        
        # Sample from each group
        samples = []
        samples_per_group = max(1, num_samples // len(groups))
        
        for group_key, group_items in groups.items():
            group_samples = random.sample(group_items, min(samples_per_group, len(group_items)))
            samples.extend(group_samples)
            
            if len(samples) >= num_samples:
                break
        
        # If we need more samples, randomly select from remaining
        if len(samples) < num_samples:
            all_items = [item for group in groups.values() for item in group]
            remaining_items = [item for item in all_items if item not in samples]
            additional_samples = random.sample(
                remaining_items, 
                min(num_samples - len(samples), len(remaining_items))
            )
            samples.extend(additional_samples)
        
        return samples[:num_samples]
    
    def sample_by_condition(self, jsonl_path: str, condition_func, 
                           num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Sample entries that meet a specific condition.
        
        Args:
            jsonl_path: Path to the JSONL file
            condition_func: Function that takes a JSON object and returns bool
            num_samples: Number of samples to extract
            
        Returns:
            List of sample JSON objects
        """
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"File not found: {jsonl_path}")
        
        matching_items = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    json_obj = json.loads(line.strip())
                    if condition_func(json_obj):
                        matching_items.append(json_obj)
                        
                        if len(matching_items) >= num_samples * 10:  # Collect more for random selection
                            break
                            
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON at line {i+1}: {e}")
        
        if not matching_items:
            print("Warning: No items found matching the condition")
            return []
        
        # Randomly sample from matching items
        return random.sample(matching_items, min(num_samples, len(matching_items)))

def print_json_structure(obj: Any, indent: int = 0, max_depth: int = 5) -> None:
    """
    Print the structure of a JSON object with types and sample values.
    
    Args:
        obj: The JSON object to inspect
        indent: Current indentation level
        max_depth: Maximum depth to traverse
    """
    if indent > max_depth:
        print("  " * indent + "... (max depth reached)")
        return
    
    prefix = "  " * indent
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            value_type = type(value).__name__
            
            if isinstance(value, dict):
                print(f"{prefix}{key} ({value_type}, {len(value)} keys):")
                print_json_structure(value, indent + 1, max_depth)
            elif isinstance(value, list):
                print(f"{prefix}{key} ({value_type}, length {len(value)}):")
                if value:
                    print(f"{prefix}  Sample element:")
                    print_json_structure(value[0], indent + 2, max_depth)
            else:
                # Show sample value for primitive types
                sample_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                print(f"{prefix}{key} ({value_type}): {sample_value}")
                
    elif isinstance(obj, list):
        print(f"{prefix}List with {len(obj)} elements")
        if obj:
            print(f"{prefix}Sample element:")
            print_json_structure(obj[0], indent + 1, max_depth)
    else:
        sample_value = str(obj)[:50] + "..." if len(str(obj)) > 50 else str(obj)
        print(f"{prefix}{type(obj).__name__}: {sample_value}")

def analyze_sample_diversity(samples: List[Dict[str, Any]], 
                           key: str = 'patient_id') -> Dict[str, Any]:
    """
    Analyze the diversity of samples based on a specific key.
    
    Args:
        samples: List of sample objects
        key: Key to analyze diversity for
        
    Returns:
        Diversity analysis results
    """
    if not samples:
        return {'error': 'No samples provided'}
    
    values = [sample.get(key) for sample in samples if key in sample]
    unique_values = set(values)
    
    return {
        'total_samples': len(samples),
        'samples_with_key': len(values),
        'unique_values': len(unique_values),
        'diversity_ratio': len(unique_values) / len(values) if values else 0,
        'sample_values': list(unique_values)[:10]  # Show first 10 unique values
    }

def sample_jsonl(jsonl_path: str, num_samples: int = 5, 
                method: str = 'first', **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function to sample JSONL files with different methods.
    
    Args:
        jsonl_path: Path to the JSONL file
        num_samples: Number of samples to extract
        method: Sampling method ('first', 'random', 'stratified', 'condition')
        **kwargs: Additional arguments for specific sampling methods
        
    Returns:
        List of sample JSON objects
    """
    sampler = JSONLSampler(kwargs.get('random_seed'))
    
    if method == 'first':
        return sampler.sample_first_n(jsonl_path, num_samples)
    elif method == 'random':
        return sampler.sample_random(jsonl_path, num_samples, kwargs.get('total_lines'))
    elif method == 'stratified':
        return sampler.sample_stratified(jsonl_path, num_samples, kwargs.get('stratify_key', 'patient_id'))
    elif method == 'condition':
        condition_func = kwargs.get('condition_func')
        if condition_func is None:
            raise ValueError("condition_func must be provided for condition sampling")
        return sampler.sample_by_condition(jsonl_path, condition_func, num_samples)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

def save_samples_with_analysis(samples: List[Dict[str, Any]], output_path: str,
                              include_structure: bool = True) -> None:
    """
    Save samples with optional structure analysis.
    
    Args:
        samples: List of sample objects
        output_path: Path to save the samples
        include_structure: Whether to include structure analysis
    """
    output_data = {
        'samples': samples,
        'metadata': {
            'total_samples': len(samples),
            'sampling_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else None
        }
    }
    
    if include_structure and samples:
        # Analyze structure of first sample
        output_data['structure_analysis'] = {
            'sample_keys': list(samples[0].keys()) if samples else [],
            'diversity_analysis': analyze_sample_diversity(samples)
        }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Samples saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample JSONL files")
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--num-samples", "-n", type=int, default=5,
                       help="Number of samples to extract")
    parser.add_argument("--method", choices=['first', 'random', 'stratified'], 
                       default='first', help="Sampling method")
    parser.add_argument("--stratify-key", default='patient_id',
                       help="Key for stratified sampling")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--print-structure", action='store_true',
                       help="Print structure of samples")
    parser.add_argument("--max-depth", type=int, default=5,
                       help="Maximum depth for structure printing")
    
    args = parser.parse_args()
    
    # Sample the file
    kwargs = {}
    if args.method == 'stratified':
        kwargs['stratify_key'] = args.stratify_key
    if args.random_seed:
        kwargs['random_seed'] = args.random_seed
    
    samples = sample_jsonl(args.input_file, args.num_samples, args.method, **kwargs)
    
    print(f"Extracted {len(samples)} samples from {args.input_file}")
    
    # Print structure if requested
    if args.print_structure:
        for i, sample in enumerate(samples):
            print(f"\n{'='*20} Sample {i+1} {'='*20}")
            print_json_structure(sample, max_depth=args.max_depth)
    
    # Analyze diversity
    diversity = analyze_sample_diversity(samples)
    print(f"\nDiversity analysis: {diversity}")
    
    # Save samples
    if args.output:
        save_samples_with_analysis(samples, args.output)
    else:
        # Print samples as JSON
        print(f"\nSample data:")
        print(json.dumps(samples, indent=2, ensure_ascii=False))