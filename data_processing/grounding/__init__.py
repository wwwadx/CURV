"""
Visual grounding data preparation module for CURV.

This module handles the preparation of visual grounding data with bounding boxes
and phrase-region relationships for training vision-language models.
"""

from .bbox_utils import (
    scale_bbox, 
    validate_bbox, 
    compute_overlap,
    calculate_patch_bbox_overlap,
    transform_bbox_for_model,
    transform_bbox_for_standard_resize,
    transform_bbox_for_rad_dino,
    get_uncertainty_phrase_mapping
)

__all__ = [
    'scale_bbox',
    'validate_bbox', 
    'compute_overlap',
    'calculate_patch_bbox_overlap',
    'transform_bbox_for_model',
    'transform_bbox_for_standard_resize',
    'transform_bbox_for_rad_dino',
    'get_uncertainty_phrase_mapping'
]