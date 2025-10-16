#!/usr/bin/env python3
"""
Data transformation utilities for CURV data processing.

This module provides functions for transforming data formats,
merging datasets, and other data manipulation tasks.

Author: CURV Team
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def transform_data_format(
    data: List[Dict[str, Any]], 
    source_format: str, 
    target_format: str,
    mapping_config: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Transform data from one format to another.
    
    Args:
        data: List of data items to transform
        source_format: Source data format identifier
        target_format: Target data format identifier
        mapping_config: Optional field mapping configuration
        
    Returns:
        List of transformed data items
    """
    if source_format == target_format:
        return data
    
    transformed_data = []
    default_mapping = {
        'mimic_to_standard': {
            'study_id': 'study_id',
            'subject_id': 'patient_id',
            'findings': 'findings',
            'impression': 'impression',
            'indication': 'indication'
        },
        'iu_xray_to_standard': {
            'id': 'study_id',
            'report': 'findings',
            'tags': 'labels',
            'filename': 'image_path'
        }
    }
    
    # Use provided mapping or default
    field_mapping = mapping_config or default_mapping.get(f"{source_format}_to_{target_format}", {})
    
    for item in data:
        transformed_item = {}
        
        # Apply field mapping
        for source_field, target_field in field_mapping.items():
            if source_field in item:
                transformed_item[target_field] = item[source_field]
        
        # Copy unmapped fields
        for key, value in item.items():
            if key not in field_mapping and key not in transformed_item:
                transformed_item[key] = value
        
        transformed_data.append(transformed_item)
    
    logger.info(f"Transformed {len(data)} items from {source_format} to {target_format}")
    return transformed_data

def merge_datasets(
    datasets: List[List[Dict[str, Any]]], 
    dataset_names: Optional[List[str]] = None,
    add_source_info: bool = True
) -> List[Dict[str, Any]]:
    """
    Merge multiple datasets into a single dataset.
    
    Args:
        datasets: List of datasets to merge
        dataset_names: Optional names for each dataset
        add_source_info: Whether to add source dataset information
        
    Returns:
        Merged dataset
    """
    merged_data = []
    
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]
    
    for i, (dataset, name) in enumerate(zip(datasets, dataset_names)):
        for item in dataset:
            merged_item = item.copy()
            
            if add_source_info:
                merged_item['source_dataset'] = name
                merged_item['source_index'] = i
            
            merged_data.append(merged_item)
    
    logger.info(f"Merged {len(datasets)} datasets into {len(merged_data)} total items")
    return merged_data

def normalize_text_fields(
    data: List[Dict[str, Any]], 
    text_fields: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Normalize text fields in the dataset.
    
    Args:
        data: List of data items
        text_fields: List of text field names to normalize
        
    Returns:
        Data with normalized text fields
    """
    if text_fields is None:
        text_fields = ['findings', 'impression', 'indication', 'report']
    
    normalized_data = []
    
    for item in data:
        normalized_item = item.copy()
        
        for field in text_fields:
            if field in item and isinstance(item[field], str):
                # Basic text normalization
                text = item[field]
                text = text.strip()
                text = ' '.join(text.split())  # Normalize whitespace
                normalized_item[field] = text
        
        normalized_data.append(normalized_item)
    
    return normalized_data

def split_dataset(
    data: List[Dict[str, Any]], 
    split_ratios: Dict[str, float] = None,
    stratify_field: Optional[str] = None,
    random_seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        data: Dataset to split
        split_ratios: Dictionary with split ratios (e.g., {'train': 0.7, 'val': 0.15, 'test': 0.15})
        stratify_field: Optional field to stratify splits on
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with split datasets
    """
    import random
    random.seed(random_seed)
    
    if split_ratios is None:
        split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    # Validate split ratios
    total_ratio = sum(split_ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split indices
    n_total = len(shuffled_data)
    splits = {}
    start_idx = 0
    
    for split_name, ratio in split_ratios.items():
        if split_name == list(split_ratios.keys())[-1]:  # Last split gets remaining data
            end_idx = n_total
        else:
            end_idx = start_idx + int(n_total * ratio)
        
        splits[split_name] = shuffled_data[start_idx:end_idx]
        start_idx = end_idx
    
    logger.info(f"Split dataset into: {', '.join([f'{k}: {len(v)}' for k, v in splits.items()])}")
    return splits

def filter_data(
    data: List[Dict[str, Any]], 
    filter_criteria: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Filter data based on specified criteria.
    
    Args:
        data: Data to filter
        filter_criteria: Dictionary with filter criteria
        
    Returns:
        Filtered data
    """
    filtered_data = []
    
    for item in data:
        include_item = True
        
        for field, criteria in filter_criteria.items():
            if field not in item:
                include_item = False
                break
            
            value = item[field]
            
            if isinstance(criteria, dict):
                # Range or condition-based filtering
                if 'min' in criteria and value < criteria['min']:
                    include_item = False
                    break
                if 'max' in criteria and value > criteria['max']:
                    include_item = False
                    break
                if 'equals' in criteria and value != criteria['equals']:
                    include_item = False
                    break
                if 'contains' in criteria and criteria['contains'] not in str(value):
                    include_item = False
                    break
            else:
                # Direct value comparison
                if value != criteria:
                    include_item = False
                    break
        
        if include_item:
            filtered_data.append(item)
    
    logger.info(f"Filtered data from {len(data)} to {len(filtered_data)} items")
    return filtered_data

def deduplicate_data(
    data: List[Dict[str, Any]], 
    key_fields: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Remove duplicate entries from the dataset.
    
    Args:
        data: Data to deduplicate
        key_fields: Fields to use for identifying duplicates
        
    Returns:
        Deduplicated data
    """
    if key_fields is None:
        key_fields = ['study_id', 'patient_id']
    
    seen_keys = set()
    deduplicated_data = []
    
    for item in data:
        # Create key from specified fields
        key_values = []
        for field in key_fields:
            if field in item:
                key_values.append(str(item[field]))
            else:
                key_values.append('None')
        
        key = tuple(key_values)
        
        if key not in seen_keys:
            seen_keys.add(key)
            deduplicated_data.append(item)
    
    logger.info(f"Deduplicated data from {len(data)} to {len(deduplicated_data)} items")
    return deduplicated_data