#!/usr/bin/env python3
"""
JSONL Key Analysis Module

This module provides functions to analyze the structure and keys of JSONL files,
particularly useful for understanding dataset schemas and data quality.

Author: CURV Team
"""

import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, Set, List, Any, Optional, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_io import load_jsonl, validate_jsonl_format

class JSONLKeyAnalyzer:
    """Analyzer for JSONL file key structure and statistics."""
    
    def __init__(self, max_depth: int = 5, sample_nested: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            max_depth: Maximum depth for nested key analysis
            sample_nested: Whether to sample nested structures for analysis
        """
        self.max_depth = max_depth
        self.sample_nested = sample_nested
        self.reset()
    
    def reset(self):
        """Reset analyzer state."""
        self.all_keys = set()
        self.key_counts = Counter()
        self.nested_keys = defaultdict(set)
        self.key_types = defaultdict(Counter)
        self.total_objects = 0
        self.null_counts = Counter()
        self.list_lengths = defaultdict(list)
    
    def _process_value(self, key: str, value: Any, depth: int = 0) -> None:
        """Process a single value and track its type and characteristics."""
        # Track value type
        value_type = type(value).__name__
        self.key_types[key][value_type] += 1
        
        # Track null values
        if value is None:
            self.null_counts[key] += 1
        
        # Handle different value types
        if isinstance(value, dict) and depth < self.max_depth:
            self._process_dict(value, key, depth + 1)
        elif isinstance(value, list):
            self.list_lengths[key].append(len(value))
            if value and depth < self.max_depth:
                # Sample first item for structure analysis
                if isinstance(value[0], dict):
                    list_key = f"{key}[]"
                    self._process_dict(value[0], list_key, depth + 1)
                elif isinstance(value[0], (str, int, float, bool)):
                    # Track list element types
                    element_types = [type(item).__name__ for item in value[:10]]  # Sample first 10
                    for elem_type in set(element_types):
                        self.key_types[f"{key}[{elem_type}]"][elem_type] += element_types.count(elem_type)
    
    def _process_dict(self, data: Dict[str, Any], prefix: str = "", depth: int = 0) -> None:
        """Process a dictionary recursively."""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            # Count the key
            self.key_counts[full_key] += 1
            self.all_keys.add(full_key)
            
            # Track nested relationship
            if prefix:
                self.nested_keys[prefix].add(key)
            
            # Process the value
            self._process_value(full_key, value, depth)
    
    def analyze_file(self, jsonl_path: str, progress_interval: int = 10000) -> Dict[str, Any]:
        """
        Analyze a JSONL file for key structure and statistics.
        
        Args:
            jsonl_path: Path to the JSONL file
            progress_interval: Interval for progress reporting
            
        Returns:
            Dictionary containing analysis results
        """
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        if not validate_jsonl_format(jsonl_path):
            raise ValueError(f"Invalid JSONL format: {jsonl_path}")
        
        print(f"Analyzing JSONL file: {jsonl_path}")
        print("This may take some time for large files...")
        
        self.reset()
        
        # Process the file line by line
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if progress_interval and i % progress_interval == 0 and i > 0:
                    print(f"Processed {i} objects...")
                
                try:
                    json_obj = json.loads(line.strip())
                    self.total_objects += 1
                    self._process_dict(json_obj)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {i+1}: {e}")
                    continue
        
        return self._compile_results()
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile analysis results into a structured format."""
        # Calculate statistics
        key_stats = {}
        for key in self.all_keys:
            count = self.key_counts[key]
            coverage = (count / self.total_objects) * 100 if self.total_objects > 0 else 0
            null_count = self.null_counts.get(key, 0)
            null_rate = (null_count / count) * 100 if count > 0 else 0
            
            key_stats[key] = {
                'count': count,
                'coverage_percent': round(coverage, 2),
                'null_count': null_count,
                'null_rate_percent': round(null_rate, 2),
                'types': dict(self.key_types[key])
            }
            
            # Add list statistics if applicable
            if key in self.list_lengths:
                lengths = self.list_lengths[key]
                key_stats[key]['list_stats'] = {
                    'avg_length': round(sum(lengths) / len(lengths), 2) if lengths else 0,
                    'min_length': min(lengths) if lengths else 0,
                    'max_length': max(lengths) if lengths else 0
                }
        
        # Sort keys by frequency
        top_keys = self.key_counts.most_common()
        
        results = {
            'file_info': {
                'total_objects': self.total_objects,
                'unique_keys': len(self.all_keys)
            },
            'key_frequencies': top_keys,
            'key_statistics': key_stats,
            'nested_structure': dict(self.nested_keys),
            'summary': self._generate_summary()
        }
        
        return results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the analysis."""
        if self.total_objects == 0:
            return {'error': 'No valid objects found'}
        
        # Find most common keys
        universal_keys = [key for key, count in self.key_counts.items() 
                         if count == self.total_objects]
        
        # Find keys with high null rates
        high_null_keys = []
        for key in self.all_keys:
            null_count = self.null_counts.get(key, 0)
            key_count = self.key_counts[key]
            if key_count > 0 and (null_count / key_count) > 0.5:
                high_null_keys.append(key)
        
        # Find nested structures
        nested_count = len([key for key in self.all_keys if '.' in key])
        
        return {
            'universal_keys': universal_keys,
            'high_null_rate_keys': high_null_keys,
            'nested_keys_count': nested_count,
            'avg_keys_per_object': round(len(self.all_keys) / max(1, self.total_objects), 2)
        }

def analyze_jsonl_keys(jsonl_path: str, output_path: Optional[str] = None, 
                      max_depth: int = 5) -> Dict[str, Any]:
    """
    Convenience function to analyze JSONL keys.
    
    Args:
        jsonl_path: Path to the JSONL file
        output_path: Optional path to save results
        max_depth: Maximum depth for nested analysis
        
    Returns:
        Analysis results dictionary
    """
    analyzer = JSONLKeyAnalyzer(max_depth=max_depth)
    results = analyzer.analyze_file(jsonl_path)
    
    if output_path:
        save_analysis_results(results, output_path)
    
    return results

def print_analysis_results(results: Dict[str, Any], top_n: int = 20) -> None:
    """Print analysis results in a readable format."""
    print("\n" + "="*60)
    print("JSONL KEY ANALYSIS RESULTS")
    print("="*60)
    
    # File info
    file_info = results['file_info']
    print(f"Total JSON objects: {file_info['total_objects']:,}")
    print(f"Total unique keys: {file_info['unique_keys']:,}")
    
    # Summary
    summary = results['summary']
    if 'error' not in summary:
        print(f"Average keys per object: {summary['avg_keys_per_object']}")
        print(f"Nested keys count: {summary['nested_keys_count']}")
        
        print(f"\nUniversal keys (present in all objects): {len(summary['universal_keys'])}")
        for key in summary['universal_keys'][:10]:  # Show first 10
            print(f"  • {key}")
        
        if summary['high_null_rate_keys']:
            print(f"\nKeys with high null rates: {len(summary['high_null_rate_keys'])}")
            for key in summary['high_null_rate_keys'][:5]:  # Show first 5
                null_rate = results['key_statistics'][key]['null_rate_percent']
                print(f"  • {key} ({null_rate}% null)")
    
    # Top keys by frequency
    print(f"\nTop {top_n} keys by frequency:")
    for i, (key, count) in enumerate(results['key_frequencies'][:top_n]):
        coverage = results['key_statistics'][key]['coverage_percent']
        types = list(results['key_statistics'][key]['types'].keys())
        print(f"  {i+1:2d}. {key}: {count:,} ({coverage}%) - {', '.join(types)}")
    
    # Nested structures
    nested = results['nested_structure']
    if nested:
        print(f"\nNested structures detected:")
        for parent, children in list(nested.items())[:10]:  # Show first 10
            if children:
                print(f"  {parent} → {', '.join(sorted(children))}")

def save_analysis_results(results: Dict[str, Any], output_path: str) -> None:
    """Save analysis results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")

def compare_jsonl_schemas(file1: str, file2: str) -> Dict[str, Any]:
    """
    Compare the schemas of two JSONL files.
    
    Args:
        file1: Path to first JSONL file
        file2: Path to second JSONL file
        
    Returns:
        Comparison results
    """
    print("Analyzing first file...")
    results1 = analyze_jsonl_keys(file1)
    
    print("Analyzing second file...")
    results2 = analyze_jsonl_keys(file2)
    
    keys1 = set(results1['key_statistics'].keys())
    keys2 = set(results2['key_statistics'].keys())
    
    comparison = {
        'file1_path': file1,
        'file2_path': file2,
        'file1_objects': results1['file_info']['total_objects'],
        'file2_objects': results2['file_info']['total_objects'],
        'common_keys': sorted(keys1 & keys2),
        'file1_only_keys': sorted(keys1 - keys2),
        'file2_only_keys': sorted(keys2 - keys1),
        'key_coverage_differences': {}
    }
    
    # Compare coverage for common keys
    for key in comparison['common_keys']:
        cov1 = results1['key_statistics'][key]['coverage_percent']
        cov2 = results2['key_statistics'][key]['coverage_percent']
        diff = abs(cov1 - cov2)
        if diff > 5:  # Only report significant differences
            comparison['key_coverage_differences'][key] = {
                'file1_coverage': cov1,
                'file2_coverage': cov2,
                'difference': round(diff, 2)
            }
    
    return comparison

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze JSONL file key structure")
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--max-depth", type=int, default=5, 
                       help="Maximum depth for nested analysis")
    parser.add_argument("--top-n", type=int, default=20,
                       help="Number of top keys to display")
    parser.add_argument("--compare", help="Second JSONL file for schema comparison")
    
    args = parser.parse_args()
    
    if args.compare:
        # Schema comparison mode
        comparison = compare_jsonl_schemas(args.input_file, args.compare)
        print("\n" + "="*60)
        print("SCHEMA COMPARISON RESULTS")
        print("="*60)
        print(f"File 1: {comparison['file1_path']} ({comparison['file1_objects']:,} objects)")
        print(f"File 2: {comparison['file2_path']} ({comparison['file2_objects']:,} objects)")
        print(f"Common keys: {len(comparison['common_keys'])}")
        print(f"File 1 only: {len(comparison['file1_only_keys'])}")
        print(f"File 2 only: {len(comparison['file2_only_keys'])}")
        
        if args.output:
            save_analysis_results(comparison, args.output)
    else:
        # Single file analysis mode
        results = analyze_jsonl_keys(args.input_file, args.output, args.max_depth)
        print_analysis_results(results, args.top_n)