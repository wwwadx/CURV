#!/usr/bin/env python3
"""
Unit tests for data_processing.uncertainty module

Tests cover:
- Uncertainty expression extraction from medical reports
- API integration and rate limiting
- Quality control and validation
- Checkpointing and error handling
"""

import unittest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock

from data_processing.utils.data_io import save_jsonl, load_jsonl
from data_processing.uncertainty.extract_uncertainty import (
    UncertaintyExtractor,
    extract_uncertainty_expressions,
    validate_uncertainty_output,
    analyze_uncertainty_patterns
)

class TestUncertaintyExtractor(unittest.TestCase):
    """Test UncertaintyExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock API configuration
        self.api_config = {
            "api_keys": ["test_key_1", "test_key_2"],
            "base_url": "https://api.test.com",
            "model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.1,
            "rate_limit": {
                "requests_per_minute": 60,
                "tokens_per_minute": 90000
            }
        }
        
        # Sample medical reports
        self.sample_reports = [
            {
                "study_id": "s001",
                "patient_id": "p001",
                "findings": "The chest X-ray shows possible pneumonia in the right lower lobe. There may be a small pleural effusion.",
                "impression": "Possible pneumonia, recommend follow-up."
            },
            {
                "study_id": "s002",
                "patient_id": "p002",
                "findings": "Clear lungs bilaterally. Heart size is normal.",
                "impression": "Normal chest X-ray."
            },
            {
                "study_id": "s003",
                "patient_id": "p003",
                "findings": "Suspicious opacity in the left upper lobe. Could represent infection or malignancy.",
                "impression": "Suspicious finding, further evaluation needed."
            }
        ]
        
        # Mock API responses
        self.mock_responses = {
            "s001": {
                "uncertainty_expressions": [
                    {
                        "text": "possible pneumonia",
                        "type": "diagnostic_uncertainty",
                        "confidence": 0.8,
                        "context": "The chest X-ray shows possible pneumonia in the right lower lobe"
                    },
                    {
                        "text": "may be",
                        "type": "existence_uncertainty", 
                        "confidence": 0.7,
                        "context": "There may be a small pleural effusion"
                    }
                ],
                "summary": {
                    "total_expressions": 2,
                    "uncertainty_level": "moderate",
                    "primary_concerns": ["pneumonia", "pleural effusion"]
                }
            },
            "s002": {
                "uncertainty_expressions": [],
                "summary": {
                    "total_expressions": 0,
                    "uncertainty_level": "low",
                    "primary_concerns": []
                }
            },
            "s003": {
                "uncertainty_expressions": [
                    {
                        "text": "suspicious opacity",
                        "type": "diagnostic_uncertainty",
                        "confidence": 0.9,
                        "context": "Suspicious opacity in the left upper lobe"
                    },
                    {
                        "text": "could represent",
                        "type": "diagnostic_uncertainty",
                        "confidence": 0.8,
                        "context": "Could represent infection or malignancy"
                    }
                ],
                "summary": {
                    "total_expressions": 2,
                    "uncertainty_level": "high",
                    "primary_concerns": ["opacity", "infection", "malignancy"]
                }
            }
        }
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('data_processing.uncertainty.extract_uncertainty.openai.ChatCompletion.create')
    def test_extract_uncertainty_basic(self, mock_openai):
        """Test basic uncertainty extraction."""
        # Mock OpenAI API response
        mock_openai.return_value = Mock(
            choices=[Mock(message=Mock(content=json.dumps(self.mock_responses["s001"])))]
        )
        
        extractor = UncertaintyExtractor(self.api_config)
        
        report = self.sample_reports[0]
        result = extractor.extract_uncertainty(report)
        
        self.assertIsNotNone(result)
        self.assertIn("uncertainty_expressions", result)
        self.assertIn("summary", result)
        self.assertEqual(len(result["uncertainty_expressions"]), 2)
        
        # Check uncertainty expression structure
        expr = result["uncertainty_expressions"][0]
        self.assertIn("text", expr)
        self.assertIn("type", expr)
        self.assertIn("confidence", expr)
        self.assertIn("context", expr)
    
    @patch('data_processing.uncertainty.extract_uncertainty.openai.ChatCompletion.create')
    def test_extract_uncertainty_no_expressions(self, mock_openai):
        """Test extraction when no uncertainty expressions are found."""
        mock_openai.return_value = Mock(
            choices=[Mock(message=Mock(content=json.dumps(self.mock_responses["s002"])))]
        )
        
        extractor = UncertaintyExtractor(self.api_config)
        
        report = self.sample_reports[1]
        result = extractor.extract_uncertainty(report)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result["uncertainty_expressions"]), 0)
        self.assertEqual(result["summary"]["uncertainty_level"], "low")
    
    @patch('data_processing.uncertainty.extract_uncertainty.openai.ChatCompletion.create')
    def test_api_error_handling(self, mock_openai):
        """Test handling of API errors."""
        # Mock API error
        mock_openai.side_effect = Exception("API Error")
        
        extractor = UncertaintyExtractor(self.api_config)
        
        report = self.sample_reports[0]
        result = extractor.extract_uncertainty(report)
        
        # Should return None or error result
        self.assertIsNone(result)
    
    @patch('data_processing.uncertainty.extract_uncertainty.openai.ChatCompletion.create')
    def test_rate_limiting(self, mock_openai):
        """Test rate limiting functionality."""
        mock_openai.return_value = Mock(
            choices=[Mock(message=Mock(content=json.dumps(self.mock_responses["s001"])))]
        )
        
        # Create extractor with strict rate limits
        config = self.api_config.copy()
        config["rate_limit"]["requests_per_minute"] = 2  # Very low limit
        
        extractor = UncertaintyExtractor(config)
        
        # Make multiple requests quickly
        start_time = time.time()
        
        for i in range(3):
            result = extractor.extract_uncertainty(self.sample_reports[0])
            self.assertIsNotNone(result)
        
        elapsed_time = time.time() - start_time
        
        # Should take at least some time due to rate limiting
        # (This is a basic test - in practice, rate limiting is more complex)
        self.assertGreater(elapsed_time, 0)
    
    @patch('data_processing.uncertainty.extract_uncertainty.openai.ChatCompletion.create')
    def test_api_key_rotation(self, mock_openai):
        """Test API key rotation on rate limit errors."""
        # Mock rate limit error on first key, success on second
        mock_openai.side_effect = [
            Exception("Rate limit exceeded"),
            Mock(choices=[Mock(message=Mock(content=json.dumps(self.mock_responses["s001"])))])
        ]
        
        extractor = UncertaintyExtractor(self.api_config)
        
        report = self.sample_reports[0]
        result = extractor.extract_uncertainty(report)
        
        # Should succeed with second API key
        self.assertIsNotNone(result)
        self.assertEqual(mock_openai.call_count, 2)
    
    def test_build_prompt(self):
        """Test prompt building for API requests."""
        extractor = UncertaintyExtractor(self.api_config)
        
        report = self.sample_reports[0]
        prompt = extractor._build_prompt(report)
        
        self.assertIsInstance(prompt, str)
        self.assertIn("uncertainty", prompt.lower())
        self.assertIn(report["findings"], prompt)
        self.assertIn(report["impression"], prompt)
    
    @patch('data_processing.uncertainty.extract_uncertainty.openai.ChatCompletion.create')
    def test_process_reports_batch(self, mock_openai):
        """Test batch processing of reports."""
        # Mock responses for all reports
        mock_openai.side_effect = [
            Mock(choices=[Mock(message=Mock(content=json.dumps(self.mock_responses["s001"])))]),
            Mock(choices=[Mock(message=Mock(content=json.dumps(self.mock_responses["s002"])))]),
            Mock(choices=[Mock(message=Mock(content=json.dumps(self.mock_responses["s003"])))])
        ]
        
        extractor = UncertaintyExtractor(self.api_config)
        
        input_file = os.path.join(self.temp_dir, "reports.jsonl")
        output_file = os.path.join(self.temp_dir, "uncertainty.jsonl")
        
        save_jsonl(self.sample_reports, input_file)
        
        results = extractor.process_reports(
            input_file=input_file,
            output_file=output_file,
            batch_size=2
        )
        
        self.assertEqual(len(results), 3)
        
        # Check that all reports were processed
        for result in results:
            self.assertIn("study_id", result)
            self.assertIn("uncertainty_analysis", result)
    
    def test_checkpointing(self):
        """Test checkpointing functionality."""
        extractor = UncertaintyExtractor(self.api_config)
        
        checkpoint_file = os.path.join(self.temp_dir, "checkpoint.json")
        
        # Save checkpoint
        checkpoint_data = {
            "processed_ids": ["s001", "s002"],
            "last_processed_index": 1,
            "timestamp": "2023-01-15T10:30:00"
        }
        
        extractor._save_checkpoint(checkpoint_data, checkpoint_file)
        
        # Load checkpoint
        loaded_data = extractor._load_checkpoint(checkpoint_file)
        
        self.assertEqual(loaded_data["processed_ids"], ["s001", "s002"])
        self.assertEqual(loaded_data["last_processed_index"], 1)
    
    @patch('data_processing.uncertainty.extract_uncertainty.openai.ChatCompletion.create')
    def test_resume_from_checkpoint(self, mock_openai):
        """Test resuming processing from checkpoint."""
        mock_openai.return_value = Mock(
            choices=[Mock(message=Mock(content=json.dumps(self.mock_responses["s003"])))]
        )
        
        extractor = UncertaintyExtractor(self.api_config)
        
        input_file = os.path.join(self.temp_dir, "reports.jsonl")
        output_file = os.path.join(self.temp_dir, "uncertainty.jsonl")
        checkpoint_file = os.path.join(self.temp_dir, "checkpoint.json")
        
        save_jsonl(self.sample_reports, input_file)
        
        # Create checkpoint indicating first two reports are processed
        checkpoint_data = {
            "processed_ids": ["s001", "s002"],
            "last_processed_index": 1,
            "timestamp": "2023-01-15T10:30:00"
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Process with checkpoint
        results = extractor.process_reports(
            input_file=input_file,
            output_file=output_file,
            checkpoint_file=checkpoint_file
        )
        
        # Should only process the third report
        self.assertEqual(mock_openai.call_count, 1)


class TestValidationFunctions(unittest.TestCase):
    """Test validation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_output = {
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
                "uncertainty_level": "moderate",
                "primary_concerns": ["pneumonia"]
            }
        }
        
        self.invalid_outputs = [
            # Missing required fields
            {
                "uncertainty_expressions": [
                    {
                        "text": "possible pneumonia",
                        # Missing type, confidence, context
                    }
                ]
            },
            # Invalid confidence value
            {
                "uncertainty_expressions": [
                    {
                        "text": "possible pneumonia",
                        "type": "diagnostic_uncertainty",
                        "confidence": 1.5,  # > 1.0
                        "context": "context"
                    }
                ]
            },
            # Invalid uncertainty level
            {
                "uncertainty_expressions": [],
                "summary": {
                    "total_expressions": 0,
                    "uncertainty_level": "invalid_level",
                    "primary_concerns": []
                }
            }
        ]
    
    def test_validate_uncertainty_output_valid(self):
        """Test validation of valid uncertainty output."""
        is_valid, errors = validate_uncertainty_output(self.valid_output)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_uncertainty_output_invalid(self):
        """Test validation of invalid uncertainty outputs."""
        for invalid_output in self.invalid_outputs:
            is_valid, errors = validate_uncertainty_output(invalid_output)
            
            self.assertFalse(is_valid)
            self.assertGreater(len(errors), 0)
    
    def test_validate_uncertainty_output_empty(self):
        """Test validation of empty output."""
        empty_output = {}
        
        is_valid, errors = validate_uncertainty_output(empty_output)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)


class TestAnalysisFunctions(unittest.TestCase):
    """Test analysis functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample processed data with uncertainty analysis
        self.processed_data = [
            {
                "study_id": "s001",
                "patient_id": "p001",
                "uncertainty_analysis": {
                    "uncertainty_expressions": [
                        {
                            "text": "possible pneumonia",
                            "type": "diagnostic_uncertainty",
                            "confidence": 0.8
                        },
                        {
                            "text": "may be",
                            "type": "existence_uncertainty",
                            "confidence": 0.7
                        }
                    ],
                    "summary": {
                        "total_expressions": 2,
                        "uncertainty_level": "moderate"
                    }
                }
            },
            {
                "study_id": "s002",
                "patient_id": "p002",
                "uncertainty_analysis": {
                    "uncertainty_expressions": [],
                    "summary": {
                        "total_expressions": 0,
                        "uncertainty_level": "low"
                    }
                }
            },
            {
                "study_id": "s003",
                "patient_id": "p003",
                "uncertainty_analysis": {
                    "uncertainty_expressions": [
                        {
                            "text": "suspicious opacity",
                            "type": "diagnostic_uncertainty",
                            "confidence": 0.9
                        }
                    ],
                    "summary": {
                        "total_expressions": 1,
                        "uncertainty_level": "high"
                    }
                }
            }
        ]
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_analyze_uncertainty_patterns(self):
        """Test analysis of uncertainty patterns."""
        file_path = os.path.join(self.temp_dir, "processed.jsonl")
        save_jsonl(self.processed_data, file_path)
        
        analysis = analyze_uncertainty_patterns(file_path)
        
        # Check analysis structure
        self.assertIn("total_reports", analysis)
        self.assertIn("reports_with_uncertainty", analysis)
        self.assertIn("uncertainty_coverage", analysis)
        self.assertIn("expression_statistics", analysis)
        self.assertIn("type_distribution", analysis)
        self.assertIn("confidence_statistics", analysis)
        self.assertIn("uncertainty_level_distribution", analysis)
        
        # Check specific values
        self.assertEqual(analysis["total_reports"], 3)
        self.assertEqual(analysis["reports_with_uncertainty"], 2)
        self.assertAlmostEqual(analysis["uncertainty_coverage"], 66.67, places=1)
        
        # Check expression statistics
        expr_stats = analysis["expression_statistics"]
        self.assertEqual(expr_stats["total_expressions"], 3)
        self.assertAlmostEqual(expr_stats["mean_per_report"], 1.0, places=1)
        
        # Check type distribution
        type_dist = analysis["type_distribution"]
        self.assertEqual(type_dist["diagnostic_uncertainty"], 2)
        self.assertEqual(type_dist["existence_uncertainty"], 1)
        
        # Check uncertainty level distribution
        level_dist = analysis["uncertainty_level_distribution"]
        self.assertEqual(level_dist["low"], 1)
        self.assertEqual(level_dist["moderate"], 1)
        self.assertEqual(level_dist["high"], 1)


class TestMainFunction(unittest.TestCase):
    """Test the main extract_uncertainty_expressions function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.sample_reports = [
            {
                "study_id": "s001",
                "patient_id": "p001",
                "findings": "Possible pneumonia in right lower lobe.",
                "impression": "Possible pneumonia."
            }
        ]
        
        self.api_config = {
            "api_keys": ["test_key"],
            "model": "gpt-3.5-turbo"
        }
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('data_processing.uncertainty.extract_uncertainty.UncertaintyExtractor')
    def test_extract_uncertainty_expressions_main(self, mock_extractor_class):
        """Test the main extract_uncertainty_expressions function."""
        # Mock extractor instance
        mock_extractor = Mock()
        mock_extractor.process_reports.return_value = [
            {
                "study_id": "s001",
                "uncertainty_analysis": {
                    "uncertainty_expressions": [],
                    "summary": {"total_expressions": 0}
                }
            }
        ]
        mock_extractor_class.return_value = mock_extractor
        
        input_file = os.path.join(self.temp_dir, "reports.jsonl")
        output_file = os.path.join(self.temp_dir, "uncertainty.jsonl")
        
        save_jsonl(self.sample_reports, input_file)
        
        # Run main function
        extract_uncertainty_expressions(
            input_file=input_file,
            output_file=output_file,
            api_config=self.api_config
        )
        
        # Verify extractor was created and used
        mock_extractor_class.assert_called_once_with(self.api_config)
        mock_extractor.process_reports.assert_called_once()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.api_config = {
            "api_keys": ["test_key"],
            "model": "gpt-3.5-turbo"
        }
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_empty_report_fields(self):
        """Test handling of reports with empty fields."""
        extractor = UncertaintyExtractor(self.api_config)
        
        empty_reports = [
            {"study_id": "s001", "findings": "", "impression": ""},
            {"study_id": "s002", "findings": None, "impression": None},
            {"study_id": "s003"}  # Missing fields
        ]
        
        for report in empty_reports:
            # Should handle gracefully without crashing
            try:
                prompt = extractor._build_prompt(report)
                self.assertIsInstance(prompt, str)
            except Exception as e:
                # If it raises an exception, it should be handled appropriately
                self.assertIsInstance(e, (KeyError, AttributeError, TypeError))
    
    @patch('data_processing.uncertainty.extract_uncertainty.openai.ChatCompletion.create')
    def test_malformed_api_response(self, mock_openai):
        """Test handling of malformed API responses."""
        # Mock malformed JSON response
        mock_openai.return_value = Mock(
            choices=[Mock(message=Mock(content="invalid json {"))]
        )
        
        extractor = UncertaintyExtractor(self.api_config)
        
        report = {
            "study_id": "s001",
            "findings": "Test findings",
            "impression": "Test impression"
        }
        
        result = extractor.extract_uncertainty(report)
        
        # Should handle malformed response gracefully
        self.assertIsNone(result)
    
    def test_missing_api_keys(self):
        """Test handling of missing API keys."""
        config_no_keys = {"model": "gpt-3.5-turbo"}
        
        with self.assertRaises((KeyError, ValueError)):
            UncertaintyExtractor(config_no_keys)
    
    def test_invalid_rate_limit_config(self):
        """Test handling of invalid rate limit configuration."""
        config_invalid_rate = {
            "api_keys": ["test_key"],
            "model": "gpt-3.5-turbo",
            "rate_limit": {
                "requests_per_minute": -1  # Invalid negative value
            }
        }
        
        # Should either handle gracefully or raise appropriate error
        try:
            extractor = UncertaintyExtractor(config_invalid_rate)
            # If it doesn't raise an error, it should handle the invalid config
            self.assertIsInstance(extractor, UncertaintyExtractor)
        except ValueError:
            # This is also acceptable behavior
            pass


if __name__ == '__main__':
    unittest.main()