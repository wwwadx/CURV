"""
GRPO (Generalized Reward-based Policy Optimization) Configuration

Configuration settings for training medical vision-language models
using GRPO with multiple reward functions.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class GRPOConfig:
    """
    Configuration for GRPO training of medical vision-language models.
    """
    
    # Model Configuration
    model_type: str = "qwen-vl-chat"
    model_id_or_path: str = "qwen/Qwen-VL-Chat"
    sft_type: str = "lora"
    template_type: str = "qwen-vl"
    
    # Dataset Configuration
    dataset: List[str] = field(default_factory=lambda: ["medical-cxr"])
    dataset_test_ratio: float = 0.01
    dataset_seed: int = 42
    
    # Training Configuration
    num_train_epochs: int = 1
    max_length: int = 2048
    batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # GRPO Specific Configuration
    ref_model: Optional[str] = None
    ref_model_type: Optional[str] = None
    beta: float = 0.1  # KL divergence coefficient
    label_smoothing: float = 0.0
    
    # Reward Function Configuration
    reward_funcs: List[str] = field(default_factory=lambda: [
        "format_cxr",
        "accuracy_cxr", 
        "coherence_cxr"
    ])
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "format_cxr": 0.33,
        "accuracy_cxr": 0.33,
        "coherence_cxr": 0.34
    })
    
    # VLLM Configuration for Inference
    vllm_enable: bool = True
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_max_model_len: int = 2048
    
    # LoRA Configuration
    lora_target_modules: List[str] = field(default_factory=lambda: ["ALL"])
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Logging and Checkpointing
    logging_steps: int = 5
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = True
    
    # Output Configuration
    output_dir: str = "./output"
    logging_dir: Optional[str] = None
    
    # Environment Configuration
    seed: int = 42
    dataloader_num_workers: int = 1
    
    # Advanced Configuration
    gradient_checkpointing: bool = True
    deepspeed: Optional[str] = None
    local_rank: int = -1
    ddp_backend: str = "nccl"
    
    # Prompt Configuration
    system_prompt_path: str = "prompts/system_prompt_cxr.txt"
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set logging directory if not specified
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate reward function weights
        if set(self.reward_funcs) != set(self.reward_weights.keys()):
            missing_weights = set(self.reward_funcs) - set(self.reward_weights.keys())
            extra_weights = set(self.reward_weights.keys()) - set(self.reward_funcs)
            
            if missing_weights:
                # Add default weights for missing functions
                for func in missing_weights:
                    self.reward_weights[func] = 1.0 / len(self.reward_funcs)
            
            if extra_weights:
                # Remove weights for unused functions
                for func in extra_weights:
                    del self.reward_weights[func]
        
        # Normalize reward weights to sum to 1.0
        total_weight = sum(self.reward_weights.values())
        if total_weight > 0:
            self.reward_weights = {
                func: weight / total_weight 
                for func, weight in self.reward_weights.items()
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GRPOConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'GRPOConfig':
        """Load configuration from file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class DatasetConfig:
    """Configuration for medical dataset processing."""
    
    # Dataset paths
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    test_data_path: str = "data/test"
    
    # Image processing
    image_size: int = 224
    image_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Text processing
    max_text_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Caching
    cache_dir: Optional[str] = None
    use_cache: bool = True


@dataclass
class RewardConfig:
    """Reward function configuration."""
    format_cxr: float = 1.0
    accuracy_cxr: float = 1.0
    coherence_cxr: float = 1.0
    # Removed f1_chexbert_cxr and radgraph_f1_cxr as requested
    # These are now used as dependencies for coherence calculation


def create_default_config() -> GRPOConfig:
    """Create a default GRPO configuration for medical training."""
    return GRPOConfig()


def create_config_from_args(args) -> GRPOConfig:
    """Create GRPO configuration from command line arguments."""
    config = GRPOConfig()
    
    # Update config with provided arguments
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    
    return config