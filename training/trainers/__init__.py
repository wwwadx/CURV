"""
Trainers Package for CURV Training

This package contains training implementations for medical vision-language models
using various optimization techniques including GRPO.
"""

from .grpo_trainer import GRPOTrainer

__all__ = ['GRPOTrainer']