#!/usr/bin/env python3
"""
Bounding Box Utilities for CURV

This module provides utilities for working with bounding boxes in the context
of visual grounding and uncertainty-aware training.

Author: CURV Team
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    """Represents a bounding box with associated metadata."""
    x1: float
    y1: float
    x2: float
    y2: float
    name: str = ""
    confidence: float = 1.0
    original_width: Optional[int] = None
    original_height: Optional[int] = None

def scale_bbox(
    bbox: Union[Dict[str, Any], List[float], BoundingBox], 
    original_size: Tuple[int, int], 
    target_size: Tuple[int, int]
) -> List[float]:
    """
    Scale bounding box coordinates from original image dimensions to target size.
    
    Args:
        bbox: Bounding box dictionary, list [x1, y1, x2, y2], or BoundingBox object
        original_size: Original image size (width, height)
        target_size: Target image size (width, height)
        
    Returns:
        Scaled bounding box coordinates [x1, y1, x2, y2]
    """
    if isinstance(bbox, dict):
        x1 = bbox.get('x1', bbox.get('original_x1', 0))
        y1 = bbox.get('y1', bbox.get('original_y1', 0))
        x2 = bbox.get('x2', bbox.get('original_x2', 0))
        y2 = bbox.get('y2', bbox.get('original_y2', 0))
    elif isinstance(bbox, list):
        x1, y1, x2, y2 = bbox
    else:
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
    
    orig_width, orig_height = original_size
    target_width, target_height = target_size
    
    # Calculate scaling factors
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height
    
    # Apply scaling
    scaled_x1 = x1 * scale_x
    scaled_y1 = y1 * scale_y
    scaled_x2 = x2 * scale_x
    scaled_y2 = y2 * scale_y
    
    return [scaled_x1, scaled_y1, scaled_x2, scaled_y2]

def validate_bbox(bbox: List[float], image_size: Tuple[int, int]) -> bool:
    """
    Validate that a bounding box is within image bounds and properly formatted.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        image_size: Image size (width, height)
        
    Returns:
        True if valid, False otherwise
    """
    if len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    width, height = image_size
    
    # Check bounds
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False
    
    # Check that x2 > x1 and y2 > y1
    if x2 <= x1 or y2 <= y1:
        return False
    
    return True

def compute_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_patch_bbox_overlap(
    image_size: Tuple[int, int],
    num_patches_side: int,
    bboxes: List[Dict[str, Any]],
    uncertainty_phrases: Optional[List[Dict[str, Any]]] = None,
    overlap_threshold: float = 0.05
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """
    Calculate overlap between image patches and bounding boxes for uncertainty training.
    
    Args:
        image_size: Image size (height, width)
        num_patches_side: Number of patches per side (creates NxN grid)
        bboxes: List of bounding box dictionaries
        uncertainty_phrases: List of uncertainty phrase annotations
        overlap_threshold: Minimum overlap threshold to consider
        
    Returns:
        Tuple of (uncertainty_mask, phrase_patch_pairs)
        - uncertainty_mask: Binary mask indicating uncertain patches
        - phrase_patch_pairs: List of (phrase, patch_index) pairs
    """
    height, width = image_size
    
    # Calculate patch dimensions
    patch_height = height / num_patches_side
    patch_width = width / num_patches_side
    
    # Initialize uncertainty mask
    uncertainty_mask = np.zeros((num_patches_side, num_patches_side), dtype=np.float32)
    phrase_patch_pairs = []
    
    # Create mapping from region names to uncertainty phrases
    region_to_phrases = {}
    if uncertainty_phrases:
        for item in uncertainty_phrases:
            region = item.get('region', '')
            phrases = item.get('uncertainty_phrases', [])
            if region and phrases:
                region_to_phrases[region] = phrases
    
    # Process each bounding box
    for bbox in bboxes:
        bbox_name = bbox.get('name', '')
        if not bbox_name:
            continue
        
        # Get bounding box coordinates
        if 'original_x1' in bbox:
            # Scale from original coordinates
            orig_width = bbox.get('original_width', width)
            orig_height = bbox.get('original_height', height)
            
            x1 = bbox['original_x1'] * (width / orig_width)
            y1 = bbox['original_y1'] * (height / orig_height)
            x2 = bbox['original_x2'] * (width / orig_width)
            y2 = bbox['original_y2'] * (height / orig_height)
        else:
            # Use provided coordinates
            x1, y1 = bbox.get('x1', 0), bbox.get('y1', 0)
            x2, y2 = bbox.get('x2', 0), bbox.get('y2', 0)
        
        # Validate bounding box
        bbox_coords = [x1, y1, x2, y2]
        if not validate_bbox(bbox_coords, (width, height)):
            logger.warning(f"Invalid bounding box for {bbox_name}: {bbox_coords}")
            continue
        
        # Get uncertainty phrases for this region
        phrases = region_to_phrases.get(bbox_name, [])
        if not phrases:
            continue
        
        # Calculate overlap with each patch
        for i in range(num_patches_side):
            for j in range(num_patches_side):
                # Calculate patch coordinates
                patch_x1 = j * patch_width
                patch_y1 = i * patch_height
                patch_x2 = (j + 1) * patch_width
                patch_y2 = (i + 1) * patch_height
                
                patch_coords = [patch_x1, patch_y1, patch_x2, patch_y2]
                
                # Calculate overlap
                overlap = compute_overlap(bbox_coords, patch_coords)
                
                if overlap > overlap_threshold:
                    uncertainty_mask[i, j] = 1.0
                    
                    # Add phrase-patch pairs
                    patch_idx = i * num_patches_side + j
                    for phrase in phrases:
                        phrase_patch_pairs.append((phrase, patch_idx))
    
    return uncertainty_mask, phrase_patch_pairs

def transform_bbox_for_model(
    bbox: Dict[str, Any],
    original_size: Tuple[int, int],
    model_input_size: Tuple[int, int],
    model_type: str = "standard"
) -> Dict[str, Any]:
    """
    Transform bounding box coordinates for different model input requirements.
    
    Args:
        bbox: Original bounding box dictionary
        original_size: Original image size (width, height)
        model_input_size: Model input size (width, height)
        model_type: Type of model transformation ("standard", "rad_dino", etc.)
        
    Returns:
        Transformed bounding box dictionary
    """
    if model_type == "rad_dino":
        return transform_bbox_for_rad_dino(bbox, original_size, model_input_size)
    else:
        return transform_bbox_for_standard_resize(bbox, original_size, model_input_size)

def transform_bbox_for_standard_resize(
    bbox: Dict[str, Any],
    original_size: Tuple[int, int],
    target_size: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Transform bounding box for standard resize operation.
    
    Args:
        bbox: Original bounding box dictionary
        original_size: Original image size (width, height)
        target_size: Target image size (width, height)
        
    Returns:
        Transformed bounding box dictionary
    """
    scaled_coords = scale_bbox(bbox, original_size, target_size)
    
    transformed_bbox = bbox.copy()
    transformed_bbox.update({
        'x1': scaled_coords[0],
        'y1': scaled_coords[1],
        'x2': scaled_coords[2],
        'y2': scaled_coords[3],
        'transformed_for': 'standard_resize',
        'target_size': target_size
    })
    
    return transformed_bbox

def transform_bbox_for_rad_dino(
    bbox: Dict[str, Any],
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
    shorter_side: int = 518,
    center_crop: bool = False
) -> Dict[str, Any]:
    """
    Transform bounding box for RAD-DINO model preprocessing.
    
    Args:
        bbox: Original bounding box dictionary
        original_size: Original image size (width, height)
        target_size: Target image size (width, height)
        shorter_side: Target shorter side length for RAD-DINO
        center_crop: Whether to apply center cropping
        
    Returns:
        Transformed bounding box dictionary
    """
    orig_width, orig_height = original_size
    
    # Calculate scaling factor based on shorter side
    scale = shorter_side / min(orig_width, orig_height)
    
    # Calculate new dimensions after scaling
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    
    # Scale bounding box coordinates
    scaled_coords = scale_bbox(bbox, original_size, (new_width, new_height))
    
    # Apply center crop if needed
    if center_crop:
        target_width, target_height = target_size
        
        # Calculate crop offsets
        crop_x = max(0, (new_width - target_width) // 2)
        crop_y = max(0, (new_height - target_height) // 2)
        
        # Adjust coordinates for crop
        scaled_coords[0] -= crop_x
        scaled_coords[1] -= crop_y
        scaled_coords[2] -= crop_x
        scaled_coords[3] -= crop_y
        
        # Clamp to target image bounds
        scaled_coords[0] = max(0, min(scaled_coords[0], target_width))
        scaled_coords[1] = max(0, min(scaled_coords[1], target_height))
        scaled_coords[2] = max(0, min(scaled_coords[2], target_width))
        scaled_coords[3] = max(0, min(scaled_coords[3], target_height))
    
    transformed_bbox = bbox.copy()
    transformed_bbox.update({
        'x1': scaled_coords[0],
        'y1': scaled_coords[1],
        'x2': scaled_coords[2],
        'y2': scaled_coords[3],
        'transformed_for': 'rad_dino',
        'scale_factor': scale,
        'intermediate_size': (new_width, new_height),
        'target_size': target_size,
        'center_crop': center_crop
    })
    
    return transformed_bbox

def get_uncertainty_phrase_mapping() -> Dict[str, int]:
    """
    Get standard mapping of uncertainty phrases to indices.
    
    Returns:
        Dictionary mapping phrase strings to integer indices
    """
    return {
        "likely": 0,
        "possibly": 1,
        "probable": 2,
        "may": 3,
        "appears": 4,
        "suggestive": 5,
        "suspicious": 6,
        "cannot exclude": 7,
        "probably": 8,
        "questionable": 9,
        "suspected": 10,
        "unclear": 11,
        "consistent with": 12,
        "concerning for": 13,
        "potential": 14,
        "versus": 15
    }