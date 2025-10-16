#!/usr/bin/env python3
"""
Unit tests for data_processing.grounding module

Tests cover:
- Bounding box operations (scaling, validation, IoU calculation)
- Patch-bbox overlap calculations
- Coordinate transformations for different models
"""

import unittest
import numpy as np
from dataclasses import asdict

from data_processing.grounding.bbox_utils import (
    BoundingBox,
    scale_bbox,
    validate_bbox,
    compute_overlap,
    calculate_patch_bbox_overlap,
    transform_bbox_for_standard_resize,
    transform_bbox_for_rad_dino
)

class TestBoundingBox(unittest.TestCase):
    """Test BoundingBox dataclass and basic operations."""
    
    def test_bounding_box_creation(self):
        """Test BoundingBox creation and properties."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        
        self.assertEqual(bbox.x1, 10)
        self.assertEqual(bbox.y1, 20)
        self.assertEqual(bbox.x2, 100)
        self.assertEqual(bbox.y2, 150)
        
        # Test computed properties
        self.assertEqual(bbox.width, 90)
        self.assertEqual(bbox.height, 130)
        self.assertEqual(bbox.area, 11700)
        self.assertEqual(bbox.center, (55.0, 85.0))
    
    def test_bounding_box_to_list(self):
        """Test conversion to list format."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        bbox_list = bbox.to_list()
        
        self.assertEqual(bbox_list, [10, 20, 100, 150])
    
    def test_bounding_box_from_list(self):
        """Test creation from list format."""
        bbox_list = [10, 20, 100, 150]
        bbox = BoundingBox.from_list(bbox_list)
        
        self.assertEqual(bbox.x1, 10)
        self.assertEqual(bbox.y1, 20)
        self.assertEqual(bbox.x2, 100)
        self.assertEqual(bbox.y2, 150)


class TestBoundingBoxOperations(unittest.TestCase):
    """Test bounding box utility functions."""
    
    def test_scale_bbox(self):
        """Test bounding box scaling."""
        original_bbox = [100, 150, 300, 400]
        original_size = (512, 512)
        target_size = (224, 224)
        
        scaled_bbox = scale_bbox(original_bbox, original_size, target_size)
        
        # Expected scaling factor: 224/512 = 0.4375
        expected = [43.75, 65.625, 131.25, 175.0]
        
        for i in range(4):
            self.assertAlmostEqual(scaled_bbox[i], expected[i], places=2)
    
    def test_scale_bbox_different_ratios(self):
        """Test scaling with different aspect ratios."""
        original_bbox = [0, 0, 100, 200]
        original_size = (200, 400)  # 1:2 ratio
        target_size = (100, 100)    # 1:1 ratio
        
        scaled_bbox = scale_bbox(original_bbox, original_size, target_size)
        
        # X scaling: 100/200 = 0.5
        # Y scaling: 100/400 = 0.25
        expected = [0, 0, 50, 50]
        
        for i in range(4):
            self.assertAlmostEqual(scaled_bbox[i], expected[i], places=2)
    
    def test_validate_bbox_valid(self):
        """Test validation of valid bounding boxes."""
        valid_bboxes = [
            [0, 0, 100, 100],
            [10, 20, 50, 80],
            [0.5, 0.5, 0.8, 0.9]  # Normalized coordinates
        ]
        
        for bbox in valid_bboxes:
            self.assertTrue(validate_bbox(bbox))
    
    def test_validate_bbox_invalid(self):
        """Test validation of invalid bounding boxes."""
        invalid_bboxes = [
            [100, 100, 50, 50],    # x2 < x1, y2 < y1
            [50, 100, 100, 50],    # y2 < y1
            [100, 50, 50, 100],    # x2 < x1
            [-10, 0, 100, 100],    # Negative coordinates
            [0, 0, 0, 100],        # Zero width
            [0, 0, 100, 0],        # Zero height
        ]
        
        for bbox in invalid_bboxes:
            self.assertFalse(validate_bbox(bbox))
    
    def test_compute_overlap_iou(self):
        """Test IoU computation between bounding boxes."""
        # Test overlapping boxes
        bbox1 = [0, 0, 100, 100]    # Area: 10000
        bbox2 = [50, 50, 150, 150]  # Area: 10000
        
        iou = compute_overlap(bbox1, bbox2)
        
        # Intersection: [50, 50, 100, 100] = 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500 / 17500 = 0.142857...
        expected_iou = 2500 / 17500
        self.assertAlmostEqual(iou, expected_iou, places=5)
    
    def test_compute_overlap_no_overlap(self):
        """Test IoU computation for non-overlapping boxes."""
        bbox1 = [0, 0, 50, 50]
        bbox2 = [100, 100, 150, 150]
        
        iou = compute_overlap(bbox1, bbox2)
        self.assertEqual(iou, 0.0)
    
    def test_compute_overlap_identical(self):
        """Test IoU computation for identical boxes."""
        bbox1 = [10, 20, 100, 150]
        bbox2 = [10, 20, 100, 150]
        
        iou = compute_overlap(bbox1, bbox2)
        self.assertEqual(iou, 1.0)
    
    def test_compute_overlap_contained(self):
        """Test IoU computation for contained boxes."""
        bbox1 = [0, 0, 100, 100]    # Area: 10000
        bbox2 = [25, 25, 75, 75]    # Area: 2500, fully contained
        
        iou = compute_overlap(bbox1, bbox2)
        
        # Intersection: 2500 (smaller box)
        # Union: 10000 (larger box)
        # IoU: 2500 / 10000 = 0.25
        expected_iou = 0.25
        self.assertAlmostEqual(iou, expected_iou, places=5)


class TestPatchBboxOverlap(unittest.TestCase):
    """Test patch-bbox overlap calculations."""
    
    def test_calculate_patch_bbox_overlap_basic(self):
        """Test basic patch-bbox overlap calculation."""
        image_size = (224, 224)
        num_patches_side = 14  # 14x14 = 196 patches
        bboxes = [[0, 0, 32, 32]]  # Should overlap with top-left patches
        
        uncertainty_phrases = [
            {"phrase": "test phrase", "bbox": [0, 0, 32, 32]}
        ]
        
        uncertainty_mask, phrase_patch_pairs = calculate_patch_bbox_overlap(
            image_size=image_size,
            num_patches_side=num_patches_side,
            bboxes=bboxes,
            uncertainty_phrases=uncertainty_phrases,
            overlap_threshold=0.1
        )
        
        # Check mask shape
        self.assertEqual(uncertainty_mask.shape, (196,))
        
        # Check that some patches are marked as uncertain
        self.assertGreater(np.sum(uncertainty_mask), 0)
        
        # Check phrase-patch pairs
        self.assertGreater(len(phrase_patch_pairs), 0)
        
        # Verify structure of phrase-patch pairs
        for pair in phrase_patch_pairs:
            self.assertIn('phrase', pair)
            self.assertIn('patch_idx', pair)
            self.assertIn('overlap', pair)
            self.assertGreaterEqual(pair['overlap'], 0.1)  # Above threshold
    
    def test_calculate_patch_bbox_overlap_no_overlap(self):
        """Test patch-bbox overlap with no significant overlap."""
        image_size = (224, 224)
        num_patches_side = 14
        
        # Small bbox that shouldn't have significant overlap
        bboxes = [[220, 220, 224, 224]]
        uncertainty_phrases = [
            {"phrase": "small phrase", "bbox": [220, 220, 224, 224]}
        ]
        
        uncertainty_mask, phrase_patch_pairs = calculate_patch_bbox_overlap(
            image_size=image_size,
            num_patches_side=num_patches_side,
            bboxes=bboxes,
            uncertainty_phrases=uncertainty_phrases,
            overlap_threshold=0.5  # High threshold
        )
        
        # Should have minimal or no overlap above threshold
        self.assertLessEqual(len(phrase_patch_pairs), 1)
    
    def test_calculate_patch_bbox_overlap_multiple_bboxes(self):
        """Test patch-bbox overlap with multiple bounding boxes."""
        image_size = (224, 224)
        num_patches_side = 14
        
        bboxes = [
            [0, 0, 50, 50],      # Top-left
            [174, 174, 224, 224] # Bottom-right
        ]
        
        uncertainty_phrases = [
            {"phrase": "phrase 1", "bbox": [0, 0, 50, 50]},
            {"phrase": "phrase 2", "bbox": [174, 174, 224, 224]}
        ]
        
        uncertainty_mask, phrase_patch_pairs = calculate_patch_bbox_overlap(
            image_size=image_size,
            num_patches_side=num_patches_side,
            bboxes=bboxes,
            uncertainty_phrases=uncertainty_phrases,
            overlap_threshold=0.1
        )
        
        # Should have overlaps from both bboxes
        self.assertGreater(len(phrase_patch_pairs), 1)
        
        # Check that we have phrases from both bboxes
        phrases = [pair['phrase'] for pair in phrase_patch_pairs]
        self.assertIn("phrase 1", phrases)
        self.assertIn("phrase 2", phrases)


class TestCoordinateTransformations(unittest.TestCase):
    """Test coordinate transformations for different models."""
    
    def test_transform_bbox_for_standard_resize(self):
        """Test bbox transformation for standard resize."""
        bbox = [100, 150, 300, 400]
        original_size = (512, 512)
        target_size = (224, 224)
        
        transformed = transform_bbox_for_standard_resize(
            bbox, original_size, target_size
        )
        
        # Should be same as scale_bbox for square images
        expected = scale_bbox(bbox, original_size, target_size)
        
        for i in range(4):
            self.assertAlmostEqual(transformed[i], expected[i], places=2)
    
    def test_transform_bbox_for_rad_dino(self):
        """Test bbox transformation for RadDINO model."""
        bbox = [100, 150, 300, 400]
        original_size = (512, 512)
        
        transformed = transform_bbox_for_rad_dino(bbox, original_size)
        
        # RadDINO uses 224x224 by default
        expected = scale_bbox(bbox, original_size, (224, 224))
        
        for i in range(4):
            self.assertAlmostEqual(transformed[i], expected[i], places=2)
    
    def test_transform_bbox_for_rad_dino_custom_size(self):
        """Test bbox transformation for RadDINO with custom size."""
        bbox = [0, 0, 256, 256]
        original_size = (512, 512)
        target_size = (384, 384)
        
        transformed = transform_bbox_for_rad_dino(
            bbox, original_size, target_size
        )
        
        # Should scale to 384x384
        expected = scale_bbox(bbox, original_size, target_size)
        
        for i in range(4):
            self.assertAlmostEqual(transformed[i], expected[i], places=2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_empty_bboxes(self):
        """Test handling of empty bbox lists."""
        image_size = (224, 224)
        num_patches_side = 14
        
        uncertainty_mask, phrase_patch_pairs = calculate_patch_bbox_overlap(
            image_size=image_size,
            num_patches_side=num_patches_side,
            bboxes=[],
            uncertainty_phrases=[],
            overlap_threshold=0.1
        )
        
        # Should return empty results
        self.assertEqual(uncertainty_mask.shape, (196,))
        self.assertEqual(np.sum(uncertainty_mask), 0)
        self.assertEqual(len(phrase_patch_pairs), 0)
    
    def test_invalid_image_size(self):
        """Test handling of invalid image sizes."""
        with self.assertRaises((ValueError, TypeError)):
            calculate_patch_bbox_overlap(
                image_size=(0, 224),  # Invalid size
                num_patches_side=14,
                bboxes=[[0, 0, 100, 100]],
                uncertainty_phrases=[],
                overlap_threshold=0.1
            )
    
    def test_mismatched_phrases_bboxes(self):
        """Test handling of mismatched phrases and bboxes."""
        image_size = (224, 224)
        num_patches_side = 14
        
        # More bboxes than phrases
        bboxes = [[0, 0, 50, 50], [100, 100, 150, 150]]
        uncertainty_phrases = [{"phrase": "only one", "bbox": [0, 0, 50, 50]}]
        
        # Should handle gracefully (only process matching pairs)
        uncertainty_mask, phrase_patch_pairs = calculate_patch_bbox_overlap(
            image_size=image_size,
            num_patches_side=num_patches_side,
            bboxes=bboxes,
            uncertainty_phrases=uncertainty_phrases,
            overlap_threshold=0.1
        )
        
        # Should only process the matching phrase
        phrases = [pair['phrase'] for pair in phrase_patch_pairs]
        self.assertEqual(len(set(phrases)), 1)
        self.assertIn("only one", phrases)


if __name__ == '__main__':
    unittest.main()