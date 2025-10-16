"""Reward Functions Package for CURV Training

This package contains various reward functions for evaluating
the quality of generated medical reports during training.
"""

from .base_reward import BaseRewardFunction
from .format_reward import FormatRewardCXR
from .accuracy_reward import AccuracyRewardCXR
from .coherence_reward import CoherenceRewardCXR
# Keep f1_chexbert and radgraph imports for dependency setup
from .f1_chexbert_reward import F1ChexbertRewardCXR
from .radgraph_f1_reward import RadGraphF1RewardCXR

# Registry of available reward functions (only three main rewards as requested)
REWARD_FUNCTIONS = {
    "format_cxr": FormatRewardCXR,
    "accuracy_cxr": AccuracyRewardCXR,
    "coherence_cxr": CoherenceRewardCXR,
}

def get_reward_function(name: str, **kwargs):
    """
    Get a reward function by name.
    
    Args:
        name: Name of the reward function
        **kwargs: Arguments to pass to the reward function constructor
        
    Returns:
        Instantiated reward function
        
    Raises:
        ValueError: If reward function name is not found
    """
    if name not in REWARD_FUNCTIONS:
        available = ', '.join(REWARD_FUNCTIONS.keys())
        raise ValueError(f"Unknown reward function '{name}'. Available: {available}")
    
    return REWARD_FUNCTIONS[name](**kwargs)

def list_reward_functions():
    """
    List all available reward functions.
    
    Returns:
        List of reward function names
    """
    return list(REWARD_FUNCTIONS.keys())

__all__ = [
    'BaseRewardFunction',
    'FormatRewardCXR',
    'AccuracyRewardCXR', 
    'CoherenceRewardCXR',
    'REWARD_FUNCTIONS',
    'get_reward_function',
    'list_reward_functions'
]