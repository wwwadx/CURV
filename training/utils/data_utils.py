"""
Data Utilities for Training

Utilities for loading, processing, and preparing datasets for training.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    PreTrainedTokenizer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling
)

from ..prompts.cxr_prompts import create_cxr_conversation, load_cxr_prompt


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    dataset_path: str
    dataset_type: str = "json"  # json, jsonl, csv, custom
    image_column: str = "image"
    text_column: str = "text"
    target_column: str = "target"
    max_length: int = 2048
    image_size: tuple = (224, 224)
    preprocessing_fn: Optional[Callable] = None


class CXRDataset(Dataset):
    """
    Dataset class for chest X-ray report generation.
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        image_processor: Optional[Callable] = None,
        prompt_template: Optional[Any] = None
    ):
        """
        Initialize CXR dataset.
        
        Args:
            data: List of data samples
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            image_processor: Image preprocessing function
            prompt_template: Prompt template for formatting
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_processor = image_processor
        self.prompt_template = prompt_template or load_cxr_prompt()
        self.logger = logging.getLogger(__name__)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Processed data sample
        """
        sample = self.data[idx]
        
        try:
            # Extract fields
            image_path = sample.get('image', '')
            findings = sample.get('findings', '')
            thinking = sample.get('thinking', '')
            impression = sample.get('impression', '')
            
            # Create conversation format
            conversation = create_cxr_conversation(
                image_path=image_path,
                findings=findings,
                thinking=thinking,
                impression=impression,
                prompt_template=self.prompt_template
            )
            
            # Process text
            processed_sample = self._process_text(conversation)
            
            # Process image if available
            if image_path and os.path.exists(image_path):
                processed_sample['image'] = self._process_image(image_path)
            
            return processed_sample
            
        except Exception as e:
            self.logger.warning(f"Error processing sample {idx}: {e}")
            # Return a dummy sample to avoid breaking training
            return self._create_dummy_sample()
    
    def _process_text(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text data for training.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            Processed text data
        """
        # Combine system prompt and user message
        system_prompt = conversation.get('system', '')
        messages = conversation.get('messages', [])
        
        # Format input text
        input_text = system_prompt
        if messages:
            user_message = messages[0].get('content', '')
            input_text += f"\n\nUser: {user_message}\n\nAssistant: "
        
        # Get target text (assistant response)
        target_text = ""
        if len(messages) > 1:
            target_text = messages[1].get('content', '')
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length - 256,  # Leave space for target
            padding=False,
            return_tensors="pt"
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            max_length=256,
            padding=False,
            return_tensors="pt"
        )
        
        # Combine input and target
        input_ids = torch.cat([
            input_encoding['input_ids'].squeeze(0),
            target_encoding['input_ids'].squeeze(0)
        ])
        
        attention_mask = torch.cat([
            input_encoding['attention_mask'].squeeze(0),
            target_encoding['attention_mask'].squeeze(0)
        ])
        
        # Create labels (only target tokens are used for loss)
        labels = torch.full_like(input_ids, -100)
        target_start = len(input_encoding['input_ids'].squeeze(0))
        labels[target_start:] = target_encoding['input_ids'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': input_text + target_text,
            'target': target_text
        }
    
    def _process_image(self, image_path: str) -> torch.Tensor:
        """
        Process image data.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Processed image tensor
        """
        if self.image_processor:
            return self.image_processor(image_path)
        else:
            # Return dummy image tensor if no processor available
            return torch.zeros(3, 224, 224)
    
    def _create_dummy_sample(self) -> Dict[str, Any]:
        """Create a dummy sample for error cases."""
        dummy_text = "Error processing sample"
        encoding = self.tokenizer(
            dummy_text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0),
            'text': dummy_text,
            'target': dummy_text
        }


class DatasetProcessor:
    """
    Processor for handling different dataset formats and types.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize dataset processor.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from the configured source.
        
        Returns:
            List of data samples
        """
        dataset_path = Path(self.config.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        if self.config.dataset_type == "json":
            return self._load_json(dataset_path)
        elif self.config.dataset_type == "jsonl":
            return self._load_jsonl(dataset_path)
        elif self.config.dataset_type == "csv":
            return self._load_csv(dataset_path)
        else:
            raise ValueError(f"Unsupported dataset type: {self.config.dataset_type}")
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Assume it's a single sample
            return [data]
        else:
            raise ValueError("JSON file must contain a list or dictionary")
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        import pandas as pd
        
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def preprocess_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess loaded data.
        
        Args:
            data: Raw data samples
            
        Returns:
            Preprocessed data samples
        """
        if self.config.preprocessing_fn:
            return [self.config.preprocessing_fn(sample) for sample in data]
        
        # Default preprocessing
        processed_data = []
        for sample in data:
            processed_sample = self._standardize_sample(sample)
            if processed_sample:
                processed_data.append(processed_sample)
        
        return processed_data
    
    def _standardize_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Standardize a single sample to expected format.
        
        Args:
            sample: Raw sample
            
        Returns:
            Standardized sample or None if invalid
        """
        try:
            # Map columns to standard names
            standardized = {}
            
            # Image path
            if self.config.image_column in sample:
                standardized['image'] = sample[self.config.image_column]
            
            # Text content (could be findings, thinking, impression)
            if self.config.text_column in sample:
                text_content = sample[self.config.text_column]
                
                # Try to parse structured content
                if isinstance(text_content, str):
                    # Check if it contains tagged sections
                    from ..prompts.prompt_utils import extract_cxr_sections
                    sections = extract_cxr_sections(text_content)
                    
                    if any(sections.values()):
                        standardized.update(sections)
                    else:
                        # Treat as findings if no structure found
                        standardized['findings'] = text_content
            
            # Target/label
            if self.config.target_column in sample:
                standardized['target'] = sample[self.config.target_column]
            
            # Copy other fields
            for key, value in sample.items():
                if key not in standardized:
                    standardized[key] = value
            
            return standardized
            
        except Exception as e:
            self.logger.warning(f"Failed to standardize sample: {e}")
            return None


def load_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizer,
    dataset_type: str = "json",
    max_length: int = 2048,
    image_processor: Optional[Callable] = None,
    preprocessing_fn: Optional[Callable] = None
) -> CXRDataset:
    """
    Load and create a CXR dataset.
    
    Args:
        dataset_path: Path to dataset file
        tokenizer: Tokenizer for text processing
        dataset_type: Type of dataset file
        max_length: Maximum sequence length
        image_processor: Image preprocessing function
        preprocessing_fn: Custom preprocessing function
        
    Returns:
        CXRDataset instance
    """
    config = DatasetConfig(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        max_length=max_length,
        preprocessing_fn=preprocessing_fn
    )
    
    processor = DatasetProcessor(config)
    data = processor.load_data()
    data = processor.preprocess_data(data)
    
    return CXRDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
        image_processor=image_processor
    )


def prepare_training_data(
    train_dataset_path: str,
    eval_dataset_path: Optional[str],
    tokenizer: PreTrainedTokenizer,
    **kwargs
) -> tuple:
    """
    Prepare training and evaluation datasets.
    
    Args:
        train_dataset_path: Path to training dataset
        eval_dataset_path: Path to evaluation dataset (optional)
        tokenizer: Tokenizer
        **kwargs: Additional arguments for dataset creation
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    train_dataset = load_dataset(train_dataset_path, tokenizer, **kwargs)
    
    eval_dataset = None
    if eval_dataset_path:
        eval_dataset = load_dataset(eval_dataset_path, tokenizer, **kwargs)
    
    return train_dataset, eval_dataset


def create_data_collator(
    tokenizer: PreTrainedTokenizer,
    padding: Union[bool, str] = True,
    max_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: str = "pt"
) -> DataCollatorWithPadding:
    """
    Create a data collator for batching.
    
    Args:
        tokenizer: Tokenizer
        padding: Padding strategy
        max_length: Maximum length for padding
        pad_to_multiple_of: Pad to multiple of this value
        return_tensors: Return tensor type
        
    Returns:
        Data collator
    """
    return DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=padding,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors
    )


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        collate_fn: Custom collate function
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )