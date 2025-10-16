"""
Uncertainty processing module for CURV.

This module contains tools for extracting and processing uncertainty expressions
from radiology reports, which is a core component of the CURV framework.
"""

from .extract_uncertainty import (
    UncertaintyExtractor,
    UncertaintyConfig,
    extract_uncertainty_expressions,
    validate_uncertainty_output,
    analyze_uncertainty_patterns
)

__all__ = [
    'UncertaintyExtractor',
    'UncertaintyConfig',
    'extract_uncertainty_expressions',
    'validate_uncertainty_output',
    'analyze_uncertainty_patterns'
]