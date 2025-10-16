"""
CURV Training Configuration Package

This package contains configuration classes and utilities for training medical vision-language models.

Classes:
    - GRPOConfig: Main configuration for GRPO training
    - DatasetConfig: Dataset configuration
    - RewardConfig: Reward function configuration

Functions:
    - create_default_config: Create default GRPO configuration
    - create_config_from_args: Create configuration from command line arguments
"""

from .grpo_config import (
    GRPOConfig,
    DatasetConfig,
    RewardConfig,
    create_default_config,
    create_config_from_args
)

__version__ = "0.1.0"
__all__ = [
    "GRPOConfig",
    "DatasetConfig", 
    "RewardConfig",
    "create_default_config",
    "create_config_from_args"
]