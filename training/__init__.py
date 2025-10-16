"""
CURV Training Module

This module provides comprehensive training functionality for medical vision-language models,
including GRPO (Group Relative Policy Optimization) training with medical domain-specific
reward functions.

Main Components:
- reward_functions: Medical domain-specific reward functions for RLHF
- configs: Training configuration management
- scripts: Training execution scripts
- prompts: System prompts and templates
- utils: Training utilities and helpers
"""

from .reward_functions import *
from .configs import TrainingConfig
from .utils import setup_training_environment

__version__ = "1.0.0"
__all__ = [
    "TrainingConfig",
    "setup_training_environment",
]