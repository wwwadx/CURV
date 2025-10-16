#!/usr/bin/env python3
"""
Data I/O utilities for CURV data processing.

This module provides common functions for loading and saving data in various formats
used throughout the CURV data processing pipeline.

Author: CURV Team
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries loaded from the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If a line contains invalid JSON
    """
    data = []
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                logger.debug(f"Invalid line content: {line[:100]}...")
    
    logger.info(f"Loaded {len(data)} items from {file_path}")
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to the output JSONL file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(data)} items to {file_path}")

def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary loaded from the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded JSON data from {file_path}")
    return data

def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to the output JSON file
        indent: Number of spaces for indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    
    logger.info(f"Saved JSON data to {file_path}")

def load_text_file(file_path: Union[str, Path]) -> str:
    """
    Load text from a file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Text content of the file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    logger.info(f"Loaded text file {file_path} ({len(content)} characters)")
    return content

def save_text_file(content: str, file_path: Union[str, Path]) -> None:
    """
    Save text to a file.
    
    Args:
        content: Text content to save
        file_path: Path to the output text file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Saved text file {file_path} ({len(content)} characters)")

def sample_jsonl(
    file_path: Union[str, Path], 
    n_samples: int = 10, 
    output_file: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Sample entries from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        n_samples: Number of samples to extract
        output_file: Optional path to save samples
        
    Returns:
        List of sampled dictionaries
    """
    import random
    
    # Load all data
    data = load_jsonl(file_path)
    
    # Sample data
    if len(data) <= n_samples:
        samples = data
        logger.warning(f"File has only {len(data)} items, returning all")
    else:
        samples = random.sample(data, n_samples)
        logger.info(f"Sampled {n_samples} items from {len(data)} total")
    
    # Save samples if requested
    if output_file:
        save_jsonl(samples, output_file)
    
    return samples

def count_lines(file_path: Union[str, Path]) -> int:
    """
    Count the number of lines in a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Number of lines in the file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        count = sum(1 for _ in f)
    
    logger.info(f"File {file_path} has {count} lines")
    return count

def validate_jsonl_format(file_path: Union[str, Path]) -> bool:
    """
    Validate that a file is in proper JSONL format.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        True if valid JSONL format, False otherwise
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    try:
        data = load_jsonl(file_path)
        logger.info(f"JSONL file {file_path} is valid with {len(data)} entries")
        return True
    except Exception as e:
        logger.error(f"JSONL file {file_path} is invalid: {e}")
        return False