"""
CURV Data Processing Pipeline - Test Suite

This module contains unit tests and integration tests for the CURV data processing pipeline.

Test Structure:
- test_utils/: Tests for utility functions (data I/O, validation)
- test_analysis/: Tests for data analysis and sampling functions
- test_uncertainty/: Tests for uncertainty extraction (mocked API calls)
- test_previous_studies/: Tests for previous study linking logic
- test_grounding/: Tests for bounding box and grounding utilities
- test_integration/: End-to-end integration tests

Usage:
    # Run all tests
    python -m pytest tests/
    
    # Run specific test module
    python -m pytest tests/test_utils/
    
    # Run with coverage
    python -m pytest tests/ --cov=data_processing
    
    # Run with verbose output
    python -m pytest tests/ -v

Test Data:
    Test data files are located in tests/data/ and include:
    - sample_reports.jsonl: Sample medical reports for testing
    - sample_config.yaml: Test configuration file
    - expected_outputs/: Expected output files for validation

Note:
    Tests that require API calls (uncertainty extraction) use mocked responses
    to avoid actual API usage during testing.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path for imports
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_DATA_DIR = test_dir / "data"
TEST_OUTPUT_DIR = test_dir / "output"
TEST_CONFIG_FILE = test_dir / "data" / "test_config.yaml"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Common test utilities
def get_test_data_path(filename):
    """Get the full path to a test data file."""
    return str(TEST_DATA_DIR / filename)

def get_test_output_path(filename):
    """Get the full path to a test output file."""
    return str(TEST_OUTPUT_DIR / filename)

def cleanup_test_files():
    """Clean up test output files."""
    import shutil
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Test fixtures and sample data
SAMPLE_MEDICAL_REPORTS = [
    {
        "study_id": "test_001",
        "patient_id": "p001",
        "findings": "There appears to be a possible infiltrate in the right lower lobe. The cardiac silhouette is likely enlarged.",
        "impression": "Possible pneumonia. Probable cardiomegaly.",
        "study_date": "2023-01-15",
        "study_time": "14:30:00"
    },
    {
        "study_id": "test_002",
        "patient_id": "p001", 
        "findings": "The lungs are clear. No definite abnormality seen.",
        "impression": "Normal chest radiograph.",
        "study_date": "2023-06-20",
        "study_time": "09:15:00"
    },
    {
        "study_id": "test_003",
        "patient_id": "p002",
        "findings": "Mild cardiomegaly. No acute pulmonary findings.",
        "impression": "Enlarged heart, otherwise normal.",
        "study_date": "2023-03-10",
        "study_time": "11:00:00"
    }
]

SAMPLE_UNCERTAINTY_EXPRESSIONS = [
    {
        "phrase": "appears to be",
        "confidence": 0.85,
        "context": "infiltrate",
        "bbox": [100, 150, 200, 250]
    },
    {
        "phrase": "possible",
        "confidence": 0.90,
        "context": "pneumonia",
        "bbox": [150, 200, 250, 300]
    },
    {
        "phrase": "likely",
        "confidence": 0.88,
        "context": "enlarged",
        "bbox": [200, 100, 300, 200]
    }
]

SAMPLE_BBOXES = [
    [100, 150, 200, 250],  # [x1, y1, x2, y2]
    [150, 200, 250, 300],
    [200, 100, 300, 200],
    [50, 50, 150, 150]
]