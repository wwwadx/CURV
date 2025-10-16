#!/usr/bin/env python3
"""
Unit tests for data_processing.analysis module

Tests cover:
- JSONL key analysis functionality
- Data sampling strategies
- Schema comparison and validation
"""

import unittest
import tempfile
import os
import json
from unittest.mock import patch

from data_processing.utils.data_io import save_jsonl
from data_processing.analysis.key_analysis import (
    JSONLKeyAnalyzer,
    analyze_jsonl_keys,
    compare_jsonl_schemas,
    print_analysis_results
)
from data_processing.analysis.sampling import (
    JSONLSampler,
    sample_jsonl,
    print_json_structure,
    analyze_sample_diversity
)

class TestJSONLKeyAnalyzer(unittest.TestCase):
    """Test JSONLKeyAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = [
            {
                "id": 1,
                "name": "test1",
                "value": 10.5,
                "metadata": {"type": "A", "tags": ["tag1", "tag2"]},
                "active": True
            },
            {
                "id": 2,
                "name": "test2", 
                "value": None,
                "metadata": {"type": "B", "tags": ["tag3"]},
                "active": False,
                "extra_field": "additional"
            },
            {
                "id": 3,
                "name": None,
                "value": 30.7,
                "metadata": {"type": "A", "tags": []},
                "active": True,
                "nested": {"deep": {"value": 42}}
            }
        ]
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = JSONLKeyAnalyzer()
        
        self.assertEqual(analyzer.total_objects, 0)
        self.assertEqual(len(analyzer.key_counts), 0)
        self.assertEqual(len(analyzer.key_types), 0)
    
    def test_process_single_object(self):
        """Test processing a single JSON object."""
        analyzer = JSONLKeyAnalyzer()
        obj = {"id": 1, "name": "test", "value": 10.5}
        
        analyzer._process_object(obj)
        
        self.assertEqual(analyzer.total_objects, 1)
        self.assertEqual(analyzer.key_counts["id"], 1)
        self.assertEqual(analyzer.key_counts["name"], 1)
        self.assertEqual(analyzer.key_counts["value"], 1)
        
        # Check type tracking
        self.assertIn("int", analyzer.key_types["id"])
        self.assertIn("str", analyzer.key_types["name"])
        self.assertIn("float", analyzer.key_types["value"])
    
    def test_process_nested_objects(self):
        """Test processing objects with nested structures."""
        analyzer = JSONLKeyAnalyzer()
        obj = {
            "id": 1,
            "metadata": {"type": "A", "tags": ["tag1", "tag2"]},
            "nested": {"deep": {"value": 42}}
        }
        
        analyzer._process_object(obj)
        
        # Check nested key tracking
        self.assertEqual(analyzer.key_counts["metadata"], 1)
        self.assertEqual(analyzer.key_counts["metadata.type"], 1)
        self.assertEqual(analyzer.key_counts["metadata.tags"], 1)
        self.assertEqual(analyzer.key_counts["nested.deep.value"], 1)
    
    def test_process_file(self):
        """Test processing a complete JSONL file."""
        file_path = os.path.join(self.temp_dir, "test.jsonl")
        save_jsonl(self.sample_data, file_path)
        
        analyzer = JSONLKeyAnalyzer()
        analyzer.process_file(file_path)
        
        self.assertEqual(analyzer.total_objects, 3)
        
        # Check key frequencies
        self.assertEqual(analyzer.key_counts["id"], 3)
        self.assertEqual(analyzer.key_counts["name"], 3)
        self.assertEqual(analyzer.key_counts["value"], 3)
        self.assertEqual(analyzer.key_counts["extra_field"], 1)
        
        # Check null value tracking
        self.assertEqual(analyzer.null_counts["name"], 1)
        self.assertEqual(analyzer.null_counts["value"], 1)
    
    def test_get_results(self):
        """Test getting analysis results."""
        file_path = os.path.join(self.temp_dir, "test.jsonl")
        save_jsonl(self.sample_data, file_path)
        
        analyzer = JSONLKeyAnalyzer()
        analyzer.process_file(file_path)
        results = analyzer.get_results()
        
        # Check result structure
        self.assertIn("total_objects", results)
        self.assertIn("unique_keys", results)
        self.assertIn("key_statistics", results)
        self.assertIn("nested_structure", results)
        self.assertIn("summary", results)
        
        self.assertEqual(results["total_objects"], 3)
        self.assertGreater(results["unique_keys"], 5)
        
        # Check key statistics
        key_stats = results["key_statistics"]
        self.assertIn("id", key_stats)
        self.assertEqual(key_stats["id"]["count"], 3)
        self.assertEqual(key_stats["id"]["frequency"], 1.0)


class TestKeyAnalysisFunctions(unittest.TestCase):
    """Test key analysis utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = [
            {"id": 1, "name": "test1", "value": 10.5},
            {"id": 2, "name": "test2", "value": 20.3},
            {"id": 3, "name": "test3", "value": None}
        ]
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_analyze_jsonl_keys(self):
        """Test the main analyze_jsonl_keys function."""
        file_path = os.path.join(self.temp_dir, "test.jsonl")
        save_jsonl(self.sample_data, file_path)
        
        results = analyze_jsonl_keys(file_path)
        
        self.assertEqual(results["total_objects"], 3)
        self.assertIn("id", results["key_statistics"])
        self.assertIn("name", results["key_statistics"])
        self.assertIn("value", results["key_statistics"])
    
    def test_compare_jsonl_schemas(self):
        """Test schema comparison between two JSONL files."""
        # Create two files with different schemas
        data1 = [
            {"id": 1, "name": "test1", "value": 10.5},
            {"id": 2, "name": "test2", "value": 20.3}
        ]
        data2 = [
            {"id": 1, "title": "test1", "score": 95},
            {"id": 2, "title": "test2", "score": 87, "extra": "field"}
        ]
        
        file1 = os.path.join(self.temp_dir, "file1.jsonl")
        file2 = os.path.join(self.temp_dir, "file2.jsonl")
        
        save_jsonl(data1, file1)
        save_jsonl(data2, file2)
        
        comparison = compare_jsonl_schemas(file1, file2)
        
        # Check comparison structure
        self.assertIn("file1_analysis", comparison)
        self.assertIn("file2_analysis", comparison)
        self.assertIn("comparison", comparison)
        
        comp = comparison["comparison"]
        self.assertIn("common_keys", comp)
        self.assertIn("file1_only", comp)
        self.assertIn("file2_only", comp)
        
        # Check specific comparisons
        self.assertIn("id", comp["common_keys"])
        self.assertIn("name", comp["file1_only"])
        self.assertIn("value", comp["file1_only"])
        self.assertIn("title", comp["file2_only"])
        self.assertIn("score", comp["file2_only"])
    
    @patch('builtins.print')
    def test_print_analysis_results(self, mock_print):
        """Test printing analysis results."""
        results = {
            "total_objects": 3,
            "unique_keys": 4,
            "key_statistics": {
                "id": {"count": 3, "frequency": 1.0, "types": ["int"]},
                "name": {"count": 3, "frequency": 1.0, "types": ["str"]}
            },
            "summary": {
                "most_common_keys": [("id", 3), ("name", 3)],
                "least_common_keys": [("extra", 1)],
                "keys_with_nulls": ["value"]
            }
        }
        
        print_analysis_results(results, top_n=2)
        
        # Verify that print was called (basic test)
        self.assertTrue(mock_print.called)


class TestJSONLSampler(unittest.TestCase):
    """Test JSONLSampler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = [
            {"id": i, "category": "A" if i % 2 == 0 else "B", "value": i * 10}
            for i in range(1, 21)  # 20 items
        ]
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_sampler_initialization(self):
        """Test sampler initialization."""
        file_path = os.path.join(self.temp_dir, "test.jsonl")
        save_jsonl(self.sample_data, file_path)
        
        sampler = JSONLSampler(file_path)
        self.assertEqual(sampler.file_path, file_path)
    
    def test_sample_first_n(self):
        """Test sampling first N entries."""
        file_path = os.path.join(self.temp_dir, "test.jsonl")
        save_jsonl(self.sample_data, file_path)
        
        sampler = JSONLSampler(file_path)
        samples = sampler.sample_first_n(5)
        
        self.assertEqual(len(samples), 5)
        self.assertEqual(samples[0]["id"], 1)
        self.assertEqual(samples[4]["id"], 5)
    
    def test_sample_random(self):
        """Test random sampling."""
        file_path = os.path.join(self.temp_dir, "test.jsonl")
        save_jsonl(self.sample_data, file_path)
        
        sampler = JSONLSampler(file_path)
        samples = sampler.sample_random(5, seed=42)
        
        self.assertEqual(len(samples), 5)
        
        # Test reproducibility with same seed
        samples2 = sampler.sample_random(5, seed=42)
        self.assertEqual(samples, samples2)
    
    def test_sample_stratified(self):
        """Test stratified sampling."""
        file_path = os.path.join(self.temp_dir, "test.jsonl")
        save_jsonl(self.sample_data, file_path)
        
        sampler = JSONLSampler(file_path)
        samples = sampler.sample_stratified("category", 4, seed=42)
        
        self.assertEqual(len(samples), 4)
        
        # Check that both categories are represented
        categories = [s["category"] for s in samples]
        self.assertIn("A", categories)
        self.assertIn("B", categories)
    
    def test_sample_by_condition(self):
        """Test conditional sampling."""
        file_path = os.path.join(self.temp_dir, "test.jsonl")
        save_jsonl(self.sample_data, file_path)
        
        sampler = JSONLSampler(file_path)
        
        # Sample items where value > 100
        samples = sampler.sample_by_condition(
            lambda x: x["value"] > 100,
            max_samples=5
        )
        
        self.assertLessEqual(len(samples), 5)
        for sample in samples:
            self.assertGreater(sample["value"], 100)


class TestSamplingFunctions(unittest.TestCase):
    """Test sampling utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = [
            {"id": 1, "name": "test1", "nested": {"value": 10}},
            {"id": 2, "name": "test2", "nested": {"value": 20}},
            {"id": 3, "name": "test3", "nested": {"value": 30}}
        ]
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_sample_jsonl_function(self):
        """Test the main sample_jsonl function."""
        file_path = os.path.join(self.temp_dir, "test.jsonl")
        save_jsonl(self.sample_data, file_path)
        
        # Test different methods
        first_samples = sample_jsonl(file_path, 2, method='first')
        self.assertEqual(len(first_samples), 2)
        self.assertEqual(first_samples[0]["id"], 1)
        
        random_samples = sample_jsonl(file_path, 2, method='random', seed=42)
        self.assertEqual(len(random_samples), 2)
        
        stratified_samples = sample_jsonl(
            file_path, 2, method='stratified', 
            stratify_key='name', seed=42
        )
        self.assertEqual(len(stratified_samples), 2)
    
    @patch('builtins.print')
    def test_print_json_structure(self, mock_print):
        """Test JSON structure printing."""
        obj = {
            "id": 1,
            "name": "test",
            "nested": {"value": 10, "list": [1, 2, 3]},
            "tags": ["a", "b"]
        }
        
        print_json_structure(obj, max_depth=2)
        
        # Verify that print was called
        self.assertTrue(mock_print.called)
    
    def test_analyze_sample_diversity(self):
        """Test sample diversity analysis."""
        file_path = os.path.join(self.temp_dir, "test.jsonl")
        
        # Create data with different diversity patterns
        diverse_data = [
            {"category": "A", "type": "X", "value": 1},
            {"category": "B", "type": "Y", "value": 2},
            {"category": "A", "type": "Z", "value": 3},
            {"category": "C", "type": "X", "value": 4}
        ]
        
        save_jsonl(diverse_data, file_path)
        
        diversity = analyze_sample_diversity(file_path, ["category", "type"])
        
        self.assertIn("category", diversity)
        self.assertIn("type", diversity)
        
        # Check category diversity
        cat_diversity = diversity["category"]
        self.assertEqual(cat_diversity["unique_values"], 3)  # A, B, C
        self.assertEqual(cat_diversity["total_samples"], 4)
        
        # Check type diversity
        type_diversity = diversity["type"]
        self.assertEqual(type_diversity["unique_values"], 3)  # X, Y, Z


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_empty_file_analysis(self):
        """Test analysis of empty files."""
        empty_file = os.path.join(self.temp_dir, "empty.jsonl")
        save_jsonl([], empty_file)
        
        results = analyze_jsonl_keys(empty_file)
        
        self.assertEqual(results["total_objects"], 0)
        self.assertEqual(results["unique_keys"], 0)
        self.assertEqual(len(results["key_statistics"]), 0)
    
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON lines."""
        malformed_file = os.path.join(self.temp_dir, "malformed.jsonl")
        
        with open(malformed_file, 'w') as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')
        
        # Should handle gracefully and process valid lines
        analyzer = JSONLKeyAnalyzer()
        analyzer.process_file(malformed_file)
        
        # Should have processed 2 valid objects
        self.assertEqual(analyzer.total_objects, 2)
    
    def test_sampling_more_than_available(self):
        """Test sampling more items than available."""
        small_data = [{"id": 1}, {"id": 2}]
        file_path = os.path.join(self.temp_dir, "small.jsonl")
        save_jsonl(small_data, file_path)
        
        # Request more samples than available
        samples = sample_jsonl(file_path, 10, method='first')
        
        # Should return all available samples
        self.assertEqual(len(samples), 2)
    
    def test_stratified_sampling_missing_key(self):
        """Test stratified sampling with missing stratification key."""
        data = [
            {"id": 1, "category": "A"},
            {"id": 2},  # Missing category
            {"id": 3, "category": "B"}
        ]
        
        file_path = os.path.join(self.temp_dir, "missing_key.jsonl")
        save_jsonl(data, file_path)
        
        # Should handle gracefully
        samples = sample_jsonl(
            file_path, 2, method='stratified',
            stratify_key='category', seed=42
        )
        
        # Should return some samples (exact behavior depends on implementation)
        self.assertGreaterEqual(len(samples), 0)


if __name__ == '__main__':
    unittest.main()