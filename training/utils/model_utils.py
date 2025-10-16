"""
Model Utilities for Training

Utilities for loading models, setting up LoRA configurations,
and managing model states during training.
"""

import os
import logging
import torch
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)


class ModelManager:
    """
    Manager class for handling model loading, configuration, and state management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model manager.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.ref_model = None
    
    def load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model and tokenizer based on configuration.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_path = self.config.get('model_id_or_path')
        if not model_path:
            raise ValueError("model_id_or_path must be specified in config")
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer(model_path)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Setup LoRA if configured
        if self.config.get('use_lora', False):
            self.model = self._setup_lora(self.model)
        
        # Load reference model if needed
        if self.config.get('load_ref_model', False):
            self.ref_model = self._load_reference_model(model_path)
        
        return self.model, self.tokenizer
    
    def _load_tokenizer(self, model_path: str) -> PreTrainedTokenizer:
        """Load tokenizer from model path."""
        self.logger.info(f"Loading tokenizer from {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Set special tokens if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
        
        return tokenizer
    
    def _load_model(self, model_path: str) -> PreTrainedModel:
        """Load model from model path."""
        self.logger.info(f"Loading model from {model_path}")
        
        # Setup quantization if configured
        quantization_config = None
        if self.config.get('use_quantization', False):
            quantization_config = self._create_quantization_config()
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self._get_torch_dtype(),
            device_map=self.config.get('device_map', 'auto'),
            trust_remote_code=True,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True
        )
        
        # Prepare for training
        model = self._prepare_model_for_training(model)
        
        return model
    
    def _load_reference_model(self, model_path: str) -> PreTrainedModel:
        """Load reference model for KL divergence computation."""
        self.logger.info(f"Loading reference model from {model_path}")
        
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self._get_torch_dtype(),
            device_map=self.config.get('device_map', 'auto'),
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Reference model should be in eval mode and frozen
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        
        return ref_model
    
    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """Create quantization configuration."""
        return BitsAndBytesConfig(
            load_in_4bit=self.config.get('load_in_4bit', True),
            load_in_8bit=self.config.get('load_in_8bit', False),
            bnb_4bit_compute_dtype=self._get_torch_dtype(),
            bnb_4bit_use_double_quant=self.config.get('bnb_4bit_use_double_quant', True),
            bnb_4bit_quant_type=self.config.get('bnb_4bit_quant_type', 'nf4')
        )
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype from configuration."""
        dtype_str = self.config.get('torch_dtype', 'bfloat16')
        
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
            'auto': 'auto'
        }
        
        return dtype_map.get(dtype_str, torch.bfloat16)
    
    def _prepare_model_for_training(self, model: PreTrainedModel) -> PreTrainedModel:
        """Prepare model for training."""
        # Enable gradient checkpointing if configured
        if self.config.get('gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
        
        # Prepare for quantized training if using quantization
        if self.config.get('use_quantization', False):
            model = prepare_model_for_kbit_training(model)
        
        return model
    
    def _setup_lora(self, model: PreTrainedModel) -> PeftModel:
        """Setup LoRA configuration and apply to model."""
        lora_config = self._create_lora_config()
        
        self.logger.info("Applying LoRA configuration to model")
        peft_model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        peft_model.print_trainable_parameters()
        
        return peft_model
    
    def _create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration."""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get('lora_r', 64),
            lora_alpha=self.config.get('lora_alpha', 16),
            lora_dropout=self.config.get('lora_dropout', 0.1),
            target_modules=self.config.get('lora_target_modules', ['q_proj', 'v_proj']),
            bias=self.config.get('lora_bias', 'none'),
            fan_in_fan_out=self.config.get('lora_fan_in_fan_out', False),
            init_lora_weights=self.config.get('init_lora_weights', True)
        )
    
    def save_model(self, output_dir: str):
        """Save model and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        if isinstance(self.model, PeftModel):
            # Save LoRA adapters
            self.model.save_pretrained(output_dir)
            # Save base model config
            self.model.base_model.config.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"Model saved to {output_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load LoRA adapters if present
        if os.path.exists(os.path.join(checkpoint_path, 'adapter_config.json')):
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)


def load_model_and_tokenizer(
    model_id_or_path: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    use_quantization: bool = False,
    trust_remote_code: bool = True,
    **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer with common configurations.
    
    Args:
        model_id_or_path: Model identifier or path
        torch_dtype: Torch data type
        device_map: Device mapping strategy
        use_quantization: Whether to use quantization
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (model, tokenizer)
    """
    config = {
        'model_id_or_path': model_id_or_path,
        'torch_dtype': torch_dtype,
        'device_map': device_map,
        'use_quantization': use_quantization,
        'trust_remote_code': trust_remote_code,
        **kwargs
    }
    
    manager = ModelManager(config)
    return manager.load_model_and_tokenizer()


def setup_lora_config(
    r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
    bias: str = "none",
    task_type: TaskType = TaskType.CAUSAL_LM,
    **kwargs
) -> LoraConfig:
    """
    Setup LoRA configuration.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        target_modules: Target modules for LoRA
        bias: Bias configuration
        task_type: Task type
        **kwargs: Additional LoRA parameters
        
    Returns:
        LoraConfig instance
    """
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
    
    return LoraConfig(
        task_type=task_type,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        **kwargs
    )


def prepare_model_for_training(
    model: PreTrainedModel,
    use_lora: bool = False,
    lora_config: Optional[LoraConfig] = None,
    gradient_checkpointing: bool = False,
    use_quantization: bool = False
) -> PreTrainedModel:
    """
    Prepare model for training with various optimizations.
    
    Args:
        model: Pre-trained model
        use_lora: Whether to use LoRA
        lora_config: LoRA configuration
        gradient_checkpointing: Whether to enable gradient checkpointing
        use_quantization: Whether model uses quantization
        
    Returns:
        Prepared model
    """
    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Prepare for quantized training
    if use_quantization:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    if use_lora:
        if lora_config is None:
            lora_config = setup_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model


def get_model_memory_usage(model: PreTrainedModel) -> Dict[str, float]:
    """
    Get model memory usage statistics.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with memory usage statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (rough approximation)
    param_memory = total_params * 4 / (1024**3)  # Assuming float32, convert to GB
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'trainable_percentage': (trainable_params / total_params) * 100,
        'estimated_memory_gb': param_memory
    }


def save_model_config(model: PreTrainedModel, output_dir: str):
    """
    Save model configuration and metadata.
    
    Args:
        model: Model to save config for
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model config
    model.config.save_pretrained(output_dir)
    
    # Save memory usage stats
    memory_stats = get_model_memory_usage(model)
    
    import json
    with open(os.path.join(output_dir, 'model_stats.json'), 'w') as f:
        json.dump(memory_stats, f, indent=2)


def load_model_from_checkpoint(
    checkpoint_path: str,
    base_model_path: Optional[str] = None,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model from checkpoint (supports both full model and LoRA checkpoints).
    
    Args:
        checkpoint_path: Path to checkpoint
        base_model_path: Path to base model (for LoRA checkpoints)
        torch_dtype: Torch data type
        device_map: Device mapping
        
    Returns:
        Tuple of (model, tokenizer)
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Check if it's a LoRA checkpoint
    if (checkpoint_path / 'adapter_config.json').exists():
        if base_model_path is None:
            raise ValueError("base_model_path required for LoRA checkpoints")
        
        # Load base model first
        base_model, tokenizer = load_model_and_tokenizer(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=getattr(torch, torch_dtype),
            device_map=device_map,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )
    
    return model, tokenizer