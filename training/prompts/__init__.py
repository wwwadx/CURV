"""
Prompts and Templates Package

This package contains prompt templates and formatting utilities
for medical vision-language model training and inference.
"""

from .cxr_prompts import CXRPromptTemplate, load_cxr_prompt
from .prompt_utils import PromptFormatter, validate_prompt_format

__all__ = [
    'CXRPromptTemplate',
    'load_cxr_prompt', 
    'PromptFormatter',
    'validate_prompt_format'
]

__version__ = "1.0.0"