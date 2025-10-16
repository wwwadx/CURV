#!/usr/bin/env python3
"""
Integration tests for the CURV data processing pipeline

Tests cover:
- End-to-end pipeline execution
- Module integration and data flow
- Configuration loading and validation
- Error handling across modules
"""

import unittest
import tempfile
import os
import json
import yaml
from unittest.mock import patch, Mock

from data_processing.utils.data_io import save_jsonl, load_jsonl
from data_processing.utils.config import load_config, validate_config
from data_processing.analysis.key_analysis import JSONLKeyAnalyzer
from data_processing.analysis.sampling import JSONLSampler
from data_processing.previous_studies.find_previous import PreviousStudyFinder
from data_processing.grounding.bbox_operations import BoundingBoxProcessor
from data_processing.uncertainty.extract_uncertainty import UncertaintyExtractor


class TestPipelineIntegration(unittest.TestCase):
    """Test integration of the complete data processing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create comprehensive test dataset
        self.test_data = [
            {
                "study_id": "s001",
                "patient_id": "p001",
                "study_date": "2023-01-15",
                "study_time": "14:30:00",
                "findings": "The chest X-ray shows possible pneumonia in the right lower lobe. There may be a small pleural effusion.",
                "impression": "Possible pneumonia, recommend follow-up.",
                "image_path": "/path/to/image1.jpg",
                "bounding_boxes": [
                    {
                        "label": "pneumonia",
                        "coordinates": [100, 150, 200, 250],
                        "format": "xyxy",
                        "confidence": 0.85
                    }
                ]
            },
            {
                "study_id": "s002",
                "patient_id": "p001",
                "study_date": "2023-06-20",
                "study_time": "09:15:00",
                "findings": "Mild cardiomegaly compared to prior study. No acute pulmonary findings.",
                "impression": "Stable cardiomegaly.",
                "image_path": "/path/to/image2.jpg",
                "bounding_boxes": [
                    {
                        "label": "heart",
                        "coordinates": [150, 100, 300, 280],
                        "format": "xyxy",
                        "confidence": 0.92
                    }
                ]
            },
            {
                "study_id": "s003",
                "patient_id": "p002",
                "study_date": "2023-03-10",
                "study_time": "11:00:00",
                "findings": "Clear lungs bilaterally. Heart size is normal.",
                "impression": "Normal chest X-ray.",
                "image_path": "/path/to/image3.jpg",
                "bounding_boxes": []
            },
            {
                "study_id": "s004",
                "patient_id": "p001",
                "study_date": "2023-12-01",
                "study_time": "16:45:00",
                "findings": "Suspicious opacity in the left upper lobe. Could represent infection or malignancy.",
                "impression": "Suspicious finding, further evaluation needed.",
                "image_path": "/path/to/image4.jpg",
                "bounding_boxes": [
                    {
                        "label": "opacity",
                        "coordinates": [80, 120, 180, 220],
                        "format": "xyxy",
                        "confidence": 0.78
                    }
                ]
            }
        ]
        
        # Create test configuration
        self.test_config = {
            "data": {
                "input_file": os.path.join(self.temp_dir, "input.jsonl"),
                "output_dir": os.path.join(self.temp_dir, "output"),
                "intermediate_dir": os.path.join(self.temp_dir, "intermediate")
            },
            "uncertainty": {
                "enabled": True,
                "api_config": {
                    "api_keys": ["test_key"],
                    "model": "gpt-3.5-turbo",
                    "max_tokens": 1000,
                    "temperature": 0.1
                },
                "quality_control": {
                    "min_confidence": 0.5,
                    "validate_output": True
                }
            },
            "previous_studies": {
                "enabled": True,
                "max_lookback_days": 365,
                "min_time_gap_hours": 1,
                "include_reports": True,
                "include_images": False
            },
            "grounding": {
                "enabled": True,
                "image_size": [512, 512],
                "bbox_format": "xyxy",
                "models": {
                    "rad_dino": {
                        "patch_size": 16,
                        "overlap_threshold": 0.1
                    }
                }
            },
            "processing": {
                "batch_size": 2,
                "num_workers": 1,
                "save_intermediate": True
            },
            "logging": {
                "level": "INFO",
                "file": os.path.join(self.temp_dir, "pipeline.log")
            }
        }
        
        # Save test data and config
        save_jsonl(self.test_data, self.test_config["data"]["input_file"])
        
        config_file = os.path.join(self.temp_dir, "config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(self.test_config, f)
        
        self.config_file = config_file
        
        # Create output directories
        os.makedirs(self.test_config["data"]["output_dir"], exist_ok=True)
        os.makedirs(self.test_config["data"]["intermediate_dir"], exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_config_loading_and_validation(self):
        """Test configuration loading and validation."""
        # Test loading
        config = load_config(self.config_file)
        self.assertIsInstance(config, dict)
        self.assertIn("data", config)
        self.assertIn("uncertainty", config)
        self.assertIn("previous_studies", config)
        self.assertIn("grounding", config)
        
        # Test validation
        is_valid, errors = validate_config(config)
        self.assertTrue(is_valid, f"Config validation failed: {errors}")
    
    def test_data_analysis_pipeline(self):
        """Test the data analysis pipeline components."""
        input_file = self.test_config["data"]["input_file"]
        
        # Test key analysis
        analyzer = JSONLKeyAnalyzer()
        key_results = analyzer.analyze_file(input_file)
        
        self.assertIn("total_objects", key_results)
        self.assertIn("unique_keys", key_results)
        self.assertIn("key_frequencies", key_results)
        
        # Verify expected keys are found
        expected_keys = ["study_id", "patient_id", "findings", "impression", "bounding_boxes"]
        for key in expected_keys:
            self.assertIn(key, key_results["unique_keys"])
        
        # Test sampling
        sampler = JSONLSampler()
        sample = sampler.sample_first_n(input_file, n=2)
        
        self.assertEqual(len(sample), 2)
        self.assertEqual(sample[0]["study_id"], "s001")
        self.assertEqual(sample[1]["study_id"], "s002")
        
        # Test random sampling
        random_sample = sampler.sample_random(input_file, n=3, seed=42)
        self.assertEqual(len(random_sample), 3)
    
    def test_previous_studies_integration(self):
        """Test previous studies finding and integration."""
        input_file = self.test_config["data"]["input_file"]
        output_file = os.path.join(self.temp_dir, "with_previous.jsonl")
        
        finder = PreviousStudyFinder()
        
        results = finder.process_studies(
            input_file=input_file,
            output_file=output_file,
            max_lookback_days=365,
            min_time_gap_hours=1,
            include_reports=True,
            include_images=False
        )
        
        self.assertEqual(len(results), 4)
        
        # Check that previous study information was added
        for result in results:
            self.assertIn("previous_study_id", result)
            self.assertIn("days_since_previous", result)
            self.assertIn("hours_since_previous", result)
        
        # Verify specific relationships
        # s002 should have s001 as previous study
        s002_result = next(r for r in results if r["study_id"] == "s002")
        self.assertEqual(s002_result["previous_study_id"], "s001")
        self.assertIsNotNone(s002_result["days_since_previous"])
        
        # s004 should have s002 as previous study (most recent)
        s004_result = next(r for r in results if r["study_id"] == "s004")
        self.assertEqual(s004_result["previous_study_id"], "s002")
    
    def test_bounding_box_processing_integration(self):
        """Test bounding box processing and grounding."""
        processor = BoundingBoxProcessor()
        
        # Test processing all bounding boxes in the dataset
        for study in self.test_data:
            if study["bounding_boxes"]:
                for bbox in study["bounding_boxes"]:
                    # Test coordinate validation
                    coords = bbox["coordinates"]
                    is_valid = processor.validate_bbox(coords, format="xyxy")
                    self.assertTrue(is_valid)
                    
                    # Test scaling
                    scaled = processor.scale_bbox(
                        coords, 
                        original_size=(1024, 1024),
                        target_size=(512, 512),
                        format="xyxy"
                    )
                    self.assertEqual(len(scaled), 4)
                    
                    # Scaled coordinates should be half the original
                    for i, coord in enumerate(scaled):
                        self.assertAlmostEqual(coord, coords[i] / 2, places=1)
        
        # Test patch-bbox overlap calculation
        patch_coords = [0, 0, 100, 100]
        bbox_coords = [50, 50, 150, 150]
        
        overlap = processor.calculate_patch_bbox_overlap(
            patch_coords, bbox_coords, format="xyxy"
        )
        
        self.assertGreater(overlap, 0)
        self.assertLessEqual(overlap, 1.0)
    
    @patch('data_processing.uncertainty.extract_uncertainty.openai.ChatCompletion.create')
    def test_uncertainty_extraction_integration(self, mock_openai):
        """Test uncertainty extraction integration."""
        # Mock API responses
        mock_responses = {
            "s001": {
                "uncertainty_expressions": [
                    {
                        "text": "possible pneumonia",
                        "type": "diagnostic_uncertainty",
                        "confidence": 0.8,
                        "context": "The chest X-ray shows possible pneumonia"
                    }
                ],
                "summary": {
                    "total_expressions": 1,
                    "uncertainty_level": "moderate"
                }
            },
            "s002": {
                "uncertainty_expressions": [],
                "summary": {
                    "total_expressions": 0,
                    "uncertainty_level": "low"
                }
            }
        }
        
        # Mock API to return different responses based on input
        def mock_api_call(*args, **kwargs):
            # Simple mock that returns different responses
            return Mock(
                choices=[Mock(message=Mock(content=json.dumps(mock_responses["s001"])))]
            )
        
        mock_openai.side_effect = mock_api_call
        
        extractor = UncertaintyExtractor(self.test_config["uncertainty"]["api_config"])
        
        input_file = self.test_config["data"]["input_file"]
        output_file = os.path.join(self.temp_dir, "with_uncertainty.jsonl")
        
        results = extractor.process_reports(
            input_file=input_file,
            output_file=output_file,
            batch_size=2
        )
        
        self.assertEqual(len(results), 4)
        
        # Check that uncertainty analysis was added
        for result in results:
            self.assertIn("uncertainty_analysis", result)
            analysis = result["uncertainty_analysis"]
            self.assertIn("uncertainty_expressions", analysis)
            self.assertIn("summary", analysis)
    
    def test_full_pipeline_workflow(self):
        """Test the complete pipeline workflow."""
        input_file = self.test_config["data"]["input_file"]
        intermediate_dir = self.test_config["data"]["intermediate_dir"]
        output_dir = self.test_config["data"]["output_dir"]
        
        # Step 1: Data analysis and validation
        analyzer = JSONLKeyAnalyzer()
        analysis_results = analyzer.analyze_file(input_file)
        
        analysis_file = os.path.join(intermediate_dir, "key_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        self.assertTrue(os.path.exists(analysis_file))
        
        # Step 2: Previous studies processing
        finder = PreviousStudyFinder()
        previous_studies_file = os.path.join(intermediate_dir, "with_previous_studies.jsonl")
        
        previous_results = finder.process_studies(
            input_file=input_file,
            output_file=previous_studies_file,
            **self.test_config["previous_studies"]
        )
        
        self.assertTrue(os.path.exists(previous_studies_file))
        self.assertEqual(len(previous_results), 4)
        
        # Step 3: Bounding box processing
        processor = BoundingBoxProcessor()
        bbox_processed_file = os.path.join(intermediate_dir, "bbox_processed.jsonl")
        
        # Process bounding boxes for each study
        bbox_results = []
        for study in previous_results:
            processed_study = study.copy()
            
            if study.get("bounding_boxes"):
                processed_bboxes = []
                for bbox in study["bounding_boxes"]:
                    # Scale bounding boxes to target size
                    scaled_coords = processor.scale_bbox(
                        bbox["coordinates"],
                        original_size=(1024, 1024),
                        target_size=tuple(self.test_config["grounding"]["image_size"]),
                        format=self.test_config["grounding"]["bbox_format"]
                    )
                    
                    processed_bbox = bbox.copy()
                    processed_bbox["scaled_coordinates"] = scaled_coords
                    processed_bboxes.append(processed_bbox)
                
                processed_study["processed_bounding_boxes"] = processed_bboxes
            
            bbox_results.append(processed_study)
        
        save_jsonl(bbox_results, bbox_processed_file)
        self.assertTrue(os.path.exists(bbox_processed_file))
        
        # Step 4: Final output generation
        final_output_file = os.path.join(output_dir, "processed_data.jsonl")
        save_jsonl(bbox_results, final_output_file)
        
        # Verify final output
        final_data = load_jsonl(final_output_file)
        self.assertEqual(len(final_data), 4)
        
        # Check that all processing steps are reflected in the data
        for study in final_data:
            # Should have original data
            self.assertIn("study_id", study)
            self.assertIn("patient_id", study)
            self.assertIn("findings", study)
            
            # Should have previous study information
            self.assertIn("previous_study_id", study)
            self.assertIn("days_since_previous", study)
            
            # Should have processed bounding boxes if original had them
            if study.get("bounding_boxes"):
                self.assertIn("processed_bounding_boxes", study)
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test with invalid input data
        invalid_data = [
            {"study_id": "s001"},  # Missing required fields
            {"patient_id": "p001"},  # Missing study_id
            {}  # Empty object
        ]
        
        invalid_file = os.path.join(self.temp_dir, "invalid.jsonl")
        save_jsonl(invalid_data, invalid_file)
        
        # Test that modules handle invalid data gracefully
        analyzer = JSONLKeyAnalyzer()
        try:
            results = analyzer.analyze_file(invalid_file)
            # Should complete without crashing
            self.assertIn("total_objects", results)
        except Exception as e:
            self.fail(f"Key analysis should handle invalid data gracefully: {e}")
        
        # Test previous studies finder with invalid data
        finder = PreviousStudyFinder()
        try:
            output_file = os.path.join(self.temp_dir, "invalid_output.jsonl")
            results = finder.process_studies(
                input_file=invalid_file,
                output_file=output_file,
                max_lookback_days=365,
                min_time_gap_hours=1
            )
            # Should handle gracefully, possibly with warnings
            self.assertIsInstance(results, list)
        except Exception as e:
            # Should raise appropriate exceptions for invalid data
            self.assertIsInstance(e, (ValueError, KeyError))
    
    def test_configuration_edge_cases(self):
        """Test configuration edge cases and validation."""
        # Test with missing required configuration sections
        incomplete_configs = [
            {},  # Empty config
            {"data": {}},  # Missing other sections
            {"data": {"input_file": "test.jsonl"}},  # Minimal config
        ]
        
        for config in incomplete_configs:
            config_file = os.path.join(self.temp_dir, "incomplete_config.yaml")
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            
            try:
                loaded_config = load_config(config_file)
                is_valid, errors = validate_config(loaded_config)
                
                if not is_valid:
                    # Should identify missing required sections
                    self.assertGreater(len(errors), 0)
                
            except Exception as e:
                # Should handle invalid configs appropriately
                self.assertIsInstance(e, (KeyError, ValueError, yaml.YAMLError))
    
    def test_data_flow_consistency(self):
        """Test that data flows consistently through the pipeline."""
        input_file = self.test_config["data"]["input_file"]
        
        # Load original data
        original_data = load_jsonl(input_file)
        original_study_ids = {study["study_id"] for study in original_data}
        
        # Process through previous studies
        finder = PreviousStudyFinder()
        temp_file = os.path.join(self.temp_dir, "temp_previous.jsonl")
        
        previous_results = finder.process_studies(
            input_file=input_file,
            output_file=temp_file
        )
        
        # Check that all original studies are preserved
        processed_study_ids = {study["study_id"] for study in previous_results}
        self.assertEqual(original_study_ids, processed_study_ids)
        
        # Check that no data is lost
        self.assertEqual(len(original_data), len(previous_results))
        
        # Check that original fields are preserved
        for original, processed in zip(original_data, previous_results):
            for key in ["study_id", "patient_id", "findings", "impression"]:
                self.assertEqual(original[key], processed[key])


class TestModuleInteroperability(unittest.TestCase):
    """Test interoperability between different modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_analysis_to_sampling_workflow(self):
        """Test workflow from analysis to sampling."""
        # Create test data with varying characteristics
        test_data = [
            {"study_id": f"s{i:03d}", "patient_id": f"p{i//3:03d}", "type": "normal"}
            for i in range(10)
        ] + [
            {"study_id": f"s{i:03d}", "patient_id": f"p{i//3:03d}", "type": "abnormal"}
            for i in range(10, 15)
        ]
        
        input_file = os.path.join(self.temp_dir, "test_data.jsonl")
        save_jsonl(test_data, input_file)
        
        # Analyze data structure
        analyzer = JSONLKeyAnalyzer()
        analysis = analyzer.analyze_file(input_file)
        
        # Use analysis results to inform sampling strategy
        total_objects = analysis["total_objects"]
        
        # Sample based on analysis
        sampler = JSONLSampler()
        
        # Sample 20% of data
        sample_size = max(1, int(total_objects * 0.2))
        sample = sampler.sample_random(input_file, n=sample_size, seed=42)
        
        self.assertEqual(len(sample), sample_size)
        self.assertLessEqual(len(sample), total_objects)
    
    def test_previous_studies_to_uncertainty_workflow(self):
        """Test workflow from previous studies to uncertainty extraction."""
        # Create data with previous study relationships
        test_data = [
            {
                "study_id": "s001",
                "patient_id": "p001",
                "study_date": "2023-01-15",
                "study_time": "14:30:00",
                "findings": "Normal chest X-ray.",
                "impression": "No acute findings."
            },
            {
                "study_id": "s002",
                "patient_id": "p001",
                "study_date": "2023-06-20",
                "study_time": "09:15:00",
                "findings": "Possible pneumonia compared to prior normal study.",
                "impression": "Possible pneumonia, recommend follow-up."
            }
        ]
        
        input_file = os.path.join(self.temp_dir, "studies.jsonl")
        previous_file = os.path.join(self.temp_dir, "with_previous.jsonl")
        
        save_jsonl(test_data, input_file)
        
        # Process previous studies
        finder = PreviousStudyFinder()
        previous_results = finder.process_studies(
            input_file=input_file,
            output_file=previous_file,
            include_reports=True
        )
        
        # Verify that previous study information is available for uncertainty analysis
        s002_result = next(r for r in previous_results if r["study_id"] == "s002")
        
        self.assertIsNotNone(s002_result["previous_study_id"])
        self.assertIn("previous_study_report", s002_result)
        
        # The uncertainty extraction could use both current and previous reports
        current_report = s002_result["findings"] + " " + s002_result["impression"]
        previous_report = s002_result.get("previous_study_report", "")
        
        combined_context = f"Current: {current_report}\nPrevious: {previous_report}"
        
        self.assertIn("Normal chest X-ray", combined_context)
        self.assertIn("Possible pneumonia", combined_context)
    
    def test_grounding_with_uncertainty_workflow(self):
        """Test workflow combining grounding and uncertainty information."""
        # Create data with both bounding boxes and uncertainty-prone text
        test_data = [
            {
                "study_id": "s001",
                "findings": "Possible opacity in the right upper lobe",
                "bounding_boxes": [
                    {
                        "label": "opacity",
                        "coordinates": [100, 150, 200, 250],
                        "format": "xyxy",
                        "confidence": 0.75  # Lower confidence suggests uncertainty
                    }
                ]
            }
        ]
        
        # Process bounding boxes
        processor = BoundingBoxProcessor()
        
        for study in test_data:
            for bbox in study["bounding_boxes"]:
                # Low confidence bounding boxes might correlate with uncertain language
                bbox_confidence = bbox["confidence"]
                
                # Validate bounding box
                is_valid = processor.validate_bbox(bbox["coordinates"], format="xyxy")
                self.assertTrue(is_valid)
                
                # Check if low confidence bbox correlates with uncertain language
                if bbox_confidence < 0.8:
                    # This study might have uncertainty expressions
                    findings_text = study["findings"].lower()
                    uncertainty_indicators = ["possible", "may", "could", "suspicious"]
                    
                    has_uncertainty = any(indicator in findings_text for indicator in uncertainty_indicators)
                    self.assertTrue(has_uncertainty)


if __name__ == '__main__':
    unittest.main()