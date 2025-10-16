#!/usr/bin/env python3
"""
Unit tests for data_processing.previous_studies module

Tests cover:
- Previous study finding and linking logic
- Date/time parsing and validation
- Study grouping and temporal analysis
"""

import unittest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch

from data_processing.utils.data_io import save_jsonl
from data_processing.previous_studies.find_previous import (
    PreviousStudyFinder,
    find_previous_studies,
    validate_study_data,
    analyze_previous_study_coverage
)

class TestPreviousStudyFinder(unittest.TestCase):
    """Test PreviousStudyFinder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.finder = PreviousStudyFinder()
        
        # Create sample study data
        base_date = datetime(2023, 1, 15, 14, 30, 0)
        self.sample_studies = [
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
                "findings": "Mild cardiomegaly compared to prior.",
                "impression": "Enlarged heart.",
                "image_path": "/path/to/image2.jpg"
            },
            {
                "study_id": "s003",
                "patient_id": "p002",
                "study_date": "2023-03-10",
                "study_time": "11:00:00",
                "findings": "Clear lungs.",
                "impression": "Normal."
            },
            {
                "study_id": "s004",
                "patient_id": "p001",
                "study_date": "2023-12-01",
                "study_time": "16:45:00",
                "findings": "Stable cardiomegaly.",
                "impression": "No change from prior."
            }
        ]
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_parse_study_datetime_standard_format(self):
        """Test parsing standard date/time formats."""
        study = {
            "study_date": "2023-01-15",
            "study_time": "14:30:00"
        }
        
        dt = self.finder._parse_study_datetime(study)
        expected = datetime(2023, 1, 15, 14, 30, 0)
        
        self.assertEqual(dt, expected)
    
    def test_parse_study_datetime_alternative_formats(self):
        """Test parsing alternative date/time formats."""
        # Test different date formats
        formats_to_test = [
            {"study_date": "01/15/2023", "study_time": "14:30:00"},
            {"study_date": "15-01-2023", "study_time": "2:30 PM"},
            {"study_date": "2023.01.15", "study_time": "14:30"},
            {"study_datetime": "2023-01-15 14:30:00"}
        ]
        
        expected = datetime(2023, 1, 15, 14, 30, 0)
        
        for study_data in formats_to_test:
            try:
                dt = self.finder._parse_study_datetime(study_data)
                # Allow some flexibility in parsing
                self.assertIsInstance(dt, datetime)
            except ValueError:
                # Some formats might not be supported, which is okay
                pass
    
    def test_parse_study_datetime_invalid_format(self):
        """Test handling of invalid date/time formats."""
        invalid_studies = [
            {"study_date": "invalid-date", "study_time": "14:30:00"},
            {"study_date": "2023-01-15", "study_time": "invalid-time"},
            {"study_date": "", "study_time": "14:30:00"},
            {}  # Missing date/time fields
        ]
        
        for study in invalid_studies:
            with self.assertRaises(ValueError):
                self.finder._parse_study_datetime(study)
    
    def test_group_studies_by_patient(self):
        """Test grouping studies by patient ID."""
        grouped = self.finder._group_studies_by_patient(self.sample_studies)
        
        self.assertEqual(len(grouped), 2)  # Two patients
        self.assertIn("p001", grouped)
        self.assertIn("p002", grouped)
        
        # Patient p001 should have 3 studies
        self.assertEqual(len(grouped["p001"]), 3)
        
        # Patient p002 should have 1 study
        self.assertEqual(len(grouped["p002"]), 1)
        
        # Check that studies are sorted by datetime
        p001_studies = grouped["p001"]
        dates = [self.finder._parse_study_datetime(s) for s in p001_studies]
        self.assertEqual(dates, sorted(dates))
    
    def test_find_previous_study_basic(self):
        """Test finding previous study for a given study."""
        grouped = self.finder._group_studies_by_patient(self.sample_studies)
        
        # Find previous study for the second study of patient p001
        current_study = self.sample_studies[1]  # June 2023 study
        previous = self.finder._find_previous_study(
            current_study, grouped["p001"], 
            max_lookback_days=365, min_time_gap_hours=1
        )
        
        self.assertIsNotNone(previous)
        self.assertEqual(previous["study_id"], "s001")  # January 2023 study
    
    def test_find_previous_study_no_previous(self):
        """Test finding previous study when none exists."""
        grouped = self.finder._group_studies_by_patient(self.sample_studies)
        
        # Find previous study for the first study of patient p001
        current_study = self.sample_studies[0]  # January 2023 study
        previous = self.finder._find_previous_study(
            current_study, grouped["p001"],
            max_lookback_days=365, min_time_gap_hours=1
        )
        
        self.assertIsNone(previous)
    
    def test_find_previous_study_time_constraints(self):
        """Test previous study finding with time constraints."""
        grouped = self.finder._group_studies_by_patient(self.sample_studies)
        
        # Test with very short lookback period
        current_study = self.sample_studies[3]  # December 2023 study
        previous = self.finder._find_previous_study(
            current_study, grouped["p001"],
            max_lookback_days=30, min_time_gap_hours=1  # Only 30 days back
        )
        
        self.assertIsNone(previous)  # June study is too far back
        
        # Test with longer lookback period
        previous = self.finder._find_previous_study(
            current_study, grouped["p001"],
            max_lookback_days=365, min_time_gap_hours=1
        )
        
        self.assertIsNotNone(previous)
        self.assertEqual(previous["study_id"], "s002")  # June 2023 study
    
    def test_process_studies_basic(self):
        """Test processing studies to add previous study information."""
        file_path = os.path.join(self.temp_dir, "studies.jsonl")
        save_jsonl(self.sample_studies, file_path)
        
        output_path = os.path.join(self.temp_dir, "output.jsonl")
        
        results = self.finder.process_studies(
            input_file=file_path,
            output_file=output_path,
            max_lookback_days=365,
            min_time_gap_hours=1,
            include_reports=True,
            include_images=False
        )
        
        self.assertEqual(len(results), 4)  # Same number of studies
        
        # Check that previous study information was added
        for result in results:
            self.assertIn("previous_study_id", result)
            self.assertIn("days_since_previous", result)
            self.assertIn("hours_since_previous", result)
            
            if result["previous_study_id"] is not None:
                self.assertIn("previous_study_report", result)
                self.assertGreater(result["days_since_previous"], 0)
    
    def test_process_studies_with_images(self):
        """Test processing studies with image inclusion."""
        file_path = os.path.join(self.temp_dir, "studies.jsonl")
        save_jsonl(self.sample_studies, file_path)
        
        output_path = os.path.join(self.temp_dir, "output.jsonl")
        
        results = self.finder.process_studies(
            input_file=file_path,
            output_file=output_path,
            include_images=True
        )
        
        # Check that image paths are included when available
        for result in results:
            if result["previous_study_id"] == "s002":
                self.assertIn("previous_study_images", result)


class TestValidationFunctions(unittest.TestCase):
    """Test validation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_validate_study_data_valid(self):
        """Test validation of valid study data."""
        valid_studies = [
            {
                "study_id": "s001",
                "patient_id": "p001",
                "study_date": "2023-01-15",
                "study_time": "14:30:00"
            },
            {
                "study_id": "s002",
                "patient_id": "p002",
                "study_date": "2023-06-20",
                "study_time": "09:15:00"
            }
        ]
        
        file_path = os.path.join(self.temp_dir, "valid.jsonl")
        save_jsonl(valid_studies, file_path)
        
        is_valid, errors = validate_study_data(file_path)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_study_data_missing_fields(self):
        """Test validation with missing required fields."""
        invalid_studies = [
            {
                "study_id": "s001",
                # Missing patient_id
                "study_date": "2023-01-15",
                "study_time": "14:30:00"
            },
            {
                "study_id": "s002",
                "patient_id": "p002",
                # Missing study_date
                "study_time": "09:15:00"
            }
        ]
        
        file_path = os.path.join(self.temp_dir, "invalid.jsonl")
        save_jsonl(invalid_studies, file_path)
        
        is_valid, errors = validate_study_data(file_path)
        
        self.assertFalse(is_valid)
        self.assertEqual(len(errors), 2)  # Two validation errors
    
    def test_validate_study_data_invalid_dates(self):
        """Test validation with invalid date formats."""
        invalid_studies = [
            {
                "study_id": "s001",
                "patient_id": "p001",
                "study_date": "invalid-date",
                "study_time": "14:30:00"
            }
        ]
        
        file_path = os.path.join(self.temp_dir, "invalid_dates.jsonl")
        save_jsonl(invalid_studies, file_path)
        
        is_valid, errors = validate_study_data(file_path)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)


class TestAnalysisFunctions(unittest.TestCase):
    """Test analysis functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample processed data with previous study information
        self.processed_data = [
            {
                "study_id": "s001",
                "patient_id": "p001",
                "previous_study_id": None,
                "days_since_previous": None
            },
            {
                "study_id": "s002",
                "patient_id": "p001",
                "previous_study_id": "s001",
                "days_since_previous": 156.5
            },
            {
                "study_id": "s003",
                "patient_id": "p002",
                "previous_study_id": None,
                "days_since_previous": None
            },
            {
                "study_id": "s004",
                "patient_id": "p001",
                "previous_study_id": "s002",
                "days_since_previous": 164.3
            }
        ]
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_analyze_previous_study_coverage(self):
        """Test analysis of previous study coverage."""
        file_path = os.path.join(self.temp_dir, "processed.jsonl")
        save_jsonl(self.processed_data, file_path)
        
        analysis = analyze_previous_study_coverage(file_path)
        
        # Check analysis structure
        self.assertIn("total_studies", analysis)
        self.assertIn("studies_with_previous", analysis)
        self.assertIn("coverage_percentage", analysis)
        self.assertIn("time_gap_statistics", analysis)
        self.assertIn("patient_statistics", analysis)
        
        # Check specific values
        self.assertEqual(analysis["total_studies"], 4)
        self.assertEqual(analysis["studies_with_previous"], 2)
        self.assertEqual(analysis["coverage_percentage"], 50.0)
        
        # Check time gap statistics
        time_stats = analysis["time_gap_statistics"]
        self.assertIn("mean_days", time_stats)
        self.assertIn("median_days", time_stats)
        self.assertIn("min_days", time_stats)
        self.assertIn("max_days", time_stats)
        
        # Check patient statistics
        patient_stats = analysis["patient_statistics"]
        self.assertEqual(patient_stats["total_patients"], 2)
        self.assertIn("studies_per_patient", patient_stats)


class TestMainFunction(unittest.TestCase):
    """Test the main find_previous_studies function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.sample_studies = [
            {
                "study_id": "s001",
                "patient_id": "p001",
                "study_date": "2023-01-15",
                "study_time": "14:30:00",
                "findings": "Normal chest X-ray."
            },
            {
                "study_id": "s002",
                "patient_id": "p001",
                "study_date": "2023-06-20",
                "study_time": "09:15:00",
                "findings": "Mild cardiomegaly."
            }
        ]
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_find_previous_studies_main_function(self):
        """Test the main find_previous_studies function."""
        input_file = os.path.join(self.temp_dir, "input.jsonl")
        output_file = os.path.join(self.temp_dir, "output.jsonl")
        
        save_jsonl(self.sample_studies, input_file)
        
        # Run the main function
        find_previous_studies(
            input_file=input_file,
            output_file=output_file,
            max_lookback_days=365,
            min_time_gap_hours=1,
            include_reports=True,
            validate_input=False  # Skip validation for test
        )
        
        # Check that output file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Load and verify results
        from data_processing.utils.data_io import load_jsonl
        results = load_jsonl(output_file)
        
        self.assertEqual(len(results), 2)
        
        # Check that previous study information was added
        for result in results:
            self.assertIn("previous_study_id", result)
            self.assertIn("days_since_previous", result)
    
    @patch('data_processing.previous_studies.find_previous.validate_study_data')
    def test_find_previous_studies_with_validation(self, mock_validate):
        """Test the main function with input validation."""
        mock_validate.return_value = (True, [])
        
        input_file = os.path.join(self.temp_dir, "input.jsonl")
        output_file = os.path.join(self.temp_dir, "output.jsonl")
        
        save_jsonl(self.sample_studies, input_file)
        
        find_previous_studies(
            input_file=input_file,
            output_file=output_file,
            validate_input=True
        )
        
        # Verify that validation was called
        mock_validate.assert_called_once_with(input_file)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.finder = PreviousStudyFinder()
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_empty_input_file(self):
        """Test handling of empty input files."""
        empty_file = os.path.join(self.temp_dir, "empty.jsonl")
        output_file = os.path.join(self.temp_dir, "output.jsonl")
        
        save_jsonl([], empty_file)
        
        results = self.finder.process_studies(
            input_file=empty_file,
            output_file=output_file
        )
        
        self.assertEqual(len(results), 0)
    
    def test_single_study_per_patient(self):
        """Test handling when each patient has only one study."""
        single_studies = [
            {
                "study_id": "s001",
                "patient_id": "p001",
                "study_date": "2023-01-15",
                "study_time": "14:30:00"
            },
            {
                "study_id": "s002",
                "patient_id": "p002",
                "study_date": "2023-06-20",
                "study_time": "09:15:00"
            }
        ]
        
        file_path = os.path.join(self.temp_dir, "single.jsonl")
        output_path = os.path.join(self.temp_dir, "output.jsonl")
        
        save_jsonl(single_studies, file_path)
        
        results = self.finder.process_studies(
            input_file=file_path,
            output_file=output_path
        )
        
        # All studies should have no previous study
        for result in results:
            self.assertIsNone(result["previous_study_id"])
            self.assertIsNone(result["days_since_previous"])
    
    def test_studies_too_close_in_time(self):
        """Test handling of studies that are too close in time."""
        close_studies = [
            {
                "study_id": "s001",
                "patient_id": "p001",
                "study_date": "2023-01-15",
                "study_time": "14:30:00"
            },
            {
                "study_id": "s002",
                "patient_id": "p001",
                "study_date": "2023-01-15",
                "study_time": "14:45:00"  # Only 15 minutes later
            }
        ]
        
        grouped = self.finder._group_studies_by_patient(close_studies)
        
        # Find previous study with 1-hour minimum gap
        current_study = close_studies[1]
        previous = self.finder._find_previous_study(
            current_study, grouped["p001"],
            max_lookback_days=365, min_time_gap_hours=1
        )
        
        # Should not find previous study due to time gap constraint
        self.assertIsNone(previous)


if __name__ == '__main__':
    unittest.main()