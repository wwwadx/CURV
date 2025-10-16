"""
CURV Data Processing Pipeline

This package contains the complete data processing pipeline for the CURV
(Coherent Uncertainty-Aware Reasoning in Vision-Language Models for X-Ray Report Generation) framework.

The pipeline includes modules for:
- Uncertainty expression extraction and processing
- Previous study identification and integration  
- Visual grounding data preparation
- Data analysis and exploration
- Visualization tools
- Common utilities

Example usage:
    from curv.data_processing import uncertainty, previous_studies, grounding
    
    # Extract uncertainty expressions
    uncertainty_data = uncertainty.extract_uncertainty_expressions(reports)
    
    # Find previous studies
    studies_with_previous = previous_studies.find_previous_studies(data)
    
    # Prepare grounding data
    grounding_data = grounding.prepare_grounding_data_for_sft(data)
"""

__version__ = "1.0.0"

# Import main modules
from . import uncertainty
from . import previous_studies  
from . import grounding
from . import analysis
from . import visualization
from . import utils

__all__ = [
    'uncertainty',
    'previous_studies',
    'grounding', 
    'analysis',
    'visualization',
    'utils'
]