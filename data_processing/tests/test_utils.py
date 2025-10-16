#!/usr/bin/env python3
"""
Unit tests for data_processing.utils module

Tests cover:
- Data I/O operations (JSONL, JSON, text files)
- Data validation functions
- File sampling and counting utilities
"""

import unittest
import tempfile
import os
import json
from pathlib import Path

from data_processing.utils.data_io import (
    load_jsonl, save_jsonl,
    load_json, save_json,
    load_text_file, save_text_file,
    sample_jsonl, count_lines,
    validate_jsonl_format
)

class TestDataIO(unittest.TestCase):
    """Test data I/O functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = [
            {"id": 1, "name": "test1", "value": 10.5},
            {"id": 2, "name": "test2", "value": 20.3},
            {"id": 3, "name": "test3", "value": 30.7}
        ]
        self.sample_json = {"key1": "value1", "key2": [1, 2, 3]}
        self.sample_text = "Line 1\nLine 2\nLine 3\n"
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_jsonl_operations(self):
        """Test JSONL save and load operations."""
        file_path = os.path.join(self.temp_dir, "test.jsonl")
        
        # Test save
        save_jsonl(self.sample_data, file_path)
        self.assertTrue(os.path.exists(file_path))
        
        # Test load
        loaded_data = load_jsonl(file_path)
        self.assertEqual(loaded_data, self.sample_data)
        
        # Test with compression
        compressed_path = os.path.join(self.temp_dir, "test_compressed.jsonl.gz")
        save_jsonl(self.sample_data, compressed_path, compress=True)
        self.assertTrue(os.path.exists(compressed_path))
        
        loaded_compressed = load_jsonl(compressed_path)
        self.assertEqual(loaded_compressed, self.sample_data)
    
    def test_json_operations(self):
        """Test JSON save and load operations."""
        file_path = os.path.join(self.temp_dir, "test.json")
        
        # Test save
        save_json(self.sample_json, file_path)
        self.assertTrue(os.path.exists(file_path))
        
        # Test load
        loaded_data = load_json(file_path)
        self.assertEqual(loaded_data, self.sample_json)
    
    def test_text_operations(self):
        """Test text file save and load operations."""
        file_path = os.path.join(self.temp_dir, "test.txt")
        
        # Test save
        save_text_file(self.sample_text, file_path)
        self.assertTrue(os.path.exists(file_path))
        
        # Test load
        loaded_text = load_text_file(file_path)
        self.assertEqual(loaded_text, self.sample_text)
        
        # Test load as lines
        lines = load_text_file(file_path, as_lines=True)
        expected_lines = ["Line 1", "Line 2", "Line 3"]
        self.assertEqual(lines, expected_lines)
    
    def test_sample_jsonl(self):
        """Test JSONL sampling functionality."""
        file_path = os.path.join(self.temp_dir, "test.jsonl")
        save_jsonl(self.sample_data, file_path)
        
        # Test sampling with different methods
        samples = sample_jsonl(file_path, num_samples=2, method='first')
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples, self.sample_data[:2])
        
        # Test random sampling (just check length)
        random_samples = sample_jsonl(file_path, num_samples=2, method='random', seed=42)
        self.assertEqual(len(random_samples), 2)
        
        # Test sampling more than available
        all_samples = sample_jsonl(file_path, num_samples=10, method='first')
        self.assertEqual(len(all_samples), 3)  # Should return all available
    
    def test_count_lines(self):
        """Test line counting functionality."""
        file_path = os.path.join(self.temp_dir, "test.jsonl")
        save_jsonl(self.sample_data, file_path)
        
        line_count = count_lines(file_path)
        self.assertEqual(line_count, 3)
        
        # Test with empty file
        empty_path = os.path.join(self.temp_dir, "empty.jsonl")
        save_jsonl([], empty_path)
        empty_count = count_lines(empty_path)
        self.assertEqual(empty_count, 0)
    
    def test_validate_jsonl_format(self):
        """Test JSONL format validation."""
        # Test valid JSONL
        valid_path = os.path.join(self.temp_dir, "valid.jsonl")
        save_jsonl(self.sample_data, valid_path)
        
        is_valid, errors = validate_jsonl_format(valid_path)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test invalid JSONL
        invalid_path = os.path.join(self.temp_dir, "invalid.jsonl")
        with open(invalid_path, 'w') as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')
        
        is_valid, errors = validate_jsonl_format(invalid_path)
        self.assertFalse(is_valid)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['line'], 2)
    
    def test_file_not_found_errors(self):
        """Test handling of file not found errors."""
        non_existent_path = os.path.join(self.temp_dir, "non_existent.jsonl")
        
        with self.assertRaises(FileNotFoundError):
            load_jsonl(non_existent_path)
        
        with self.assertRaises(FileNotFoundError):
            load_json(non_existent_path)
        
        with self.assertRaises(FileNotFoundError):
            load_text_file(non_existent_path)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_path = os.path.join(self.temp_dir, "empty.jsonl")
        
        # Save empty data
        save_jsonl([], empty_path)
        
        # Load empty data
        loaded_data = load_jsonl(empty_path)
        self.assertEqual(loaded_data, [])
        
        # Count lines in empty file
        line_count = count_lines(empty_path)
        self.assertEqual(line_count, 0)
    
    def test_large_data_handling(self):
        """Test handling of larger datasets."""
        large_data = [{"id": i, "value": f"item_{i}"} for i in range(1000)]
        large_path = os.path.join(self.temp_dir, "large.jsonl")
        
        # Save and load large data
        save_jsonl(large_data, large_path)
        loaded_data = load_jsonl(large_path)
        
        self.assertEqual(len(loaded_data), 1000)
        self.assertEqual(loaded_data[0], {"id": 0, "value": "item_0"})
        self.assertEqual(loaded_data[-1], {"id": 999, "value": "item_999"})
        
        # Test sampling from large data
        samples = sample_jsonl(large_path, num_samples=10, method='random', seed=42)
        self.assertEqual(len(samples), 10)
        
        # Test line counting
        line_count = count_lines(large_path)
        self.assertEqual(line_count, 1000)


class TestDataValidation(unittest.TestCase):
    """Test data validation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_validate_required_fields(self):
        """Test validation of required fields."""
        # This would test a hypothetical validate_required_fields function
        # For now, we'll test the existing validate_jsonl_format function
        
        data_with_missing_fields = [
            {"id": 1, "name": "test1"},  # missing 'value'
            {"id": 2, "value": 10.5},   # missing 'name'
            {"id": 3, "name": "test3", "value": 30.7}  # complete
        ]
        
        file_path = os.path.join(self.temp_dir, "test_validation.jsonl")
        save_jsonl(data_with_missing_fields, file_path)
        
        # Basic format validation should pass
        is_valid, errors = validate_jsonl_format(file_path)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_data_types(self):
        """Test validation of data types."""
        # Test mixed data types (should be valid JSON)
        mixed_data = [
            {"id": 1, "name": "test", "value": 10.5, "active": True},
            {"id": "2", "name": None, "value": "string", "active": False},
            {"id": 3, "name": ["list", "of", "items"], "value": {"nested": "object"}}
        ]
        
        file_path = os.path.join(self.temp_dir, "mixed_types.jsonl")
        save_jsonl(mixed_data, file_path)
        
        is_valid, errors = validate_jsonl_format(file_path)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)


if __name__ == '__main__':
    unittest.main()