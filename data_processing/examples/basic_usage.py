#!/usr/bin/env python3
"""
CURV Data Processing Pipeline - Basic Usage Examples

This file demonstrates how to use the CURV data processing pipeline
for various common tasks.

Author: CURV Team
"""

import os
import sys
from pathlib import Path

# Add the data_processing module to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_io import load_jsonl, save_jsonl, sample_jsonl
from analysis.key_analysis import analyze_jsonl_keys, print_analysis_results
from analysis.sampling import sample_jsonl, print_json_structure
from uncertainty.extract_uncertainty import extract_uncertainty_expressions, UncertaintyConfig
from previous_studies.find_previous import find_previous_studies
from grounding.bbox_utils import scale_bbox, calculate_patch_bbox_overlap

def example_1_basic_data_exploration():
    """Example 1: Basic data exploration and analysis."""
    print("="*60)
    print("Example 1: Basic Data Exploration")
    print("="*60)
    
    # Sample data file path (replace with your actual file)
    input_file = "data/sample_data.jsonl"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Sample file {input_file} not found. Creating a sample...")
        # Create a sample file for demonstration
        sample_data = [
            {
                "study_id": "s12345",
                "patient_id": "p001",
                "findings": "No acute cardiopulmonary abnormality. The lungs are clear.",
                "impression": "Normal chest X-ray.",
                "study_date": "2023-01-15",
                "study_time": "14:30:00"
            },
            {
                "study_id": "s12346",
                "patient_id": "p001",
                "findings": "Mild cardiomegaly. No acute pulmonary findings.",
                "impression": "Enlarged heart, otherwise normal.",
                "study_date": "2023-06-20",
                "study_time": "09:15:00"
            }
        ]
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        save_jsonl(sample_data, input_file)
        print(f"Created sample file: {input_file}")
    
    # 1. Sample the data to understand structure
    print("\n1. Sampling data to understand structure:")
    samples = sample_jsonl(input_file, num_samples=3, method='first')
    
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print_json_structure(sample, max_depth=3)
    
    # 2. Analyze key structure
    print("\n2. Analyzing key structure:")
    analysis_results = analyze_jsonl_keys(input_file)
    print_analysis_results(analysis_results, top_n=10)

def example_2_uncertainty_extraction():
    """Example 2: Extract uncertainty expressions from reports."""
    print("="*60)
    print("Example 2: Uncertainty Expression Extraction")
    print("="*60)
    
    # Sample data with medical reports
    sample_data = [
        {
            "study_id": "s001",
            "patient_id": "p001",
            "findings": "There appears to be a possible infiltrate in the right lower lobe. The cardiac silhouette is likely enlarged.",
            "impression": "Possible pneumonia. Probable cardiomegaly."
        },
        {
            "study_id": "s002", 
            "patient_id": "p002",
            "findings": "The lungs are clear. No definite abnormality seen.",
            "impression": "Normal chest radiograph."
        }
    ]
    
    input_file = "data/uncertainty_sample.jsonl"
    output_file = "data/uncertainty_output.jsonl"
    
    # Save sample data
    os.makedirs("data", exist_ok=True)
    save_jsonl(sample_data, input_file)
    
    # Configure uncertainty extraction
    config = UncertaintyConfig(
        api_keys=["your_api_key_here"],  # Replace with actual API keys
        checkpoint_dir="data/checkpoints"
    )
    
    print("Note: This example requires valid API keys for uncertainty extraction.")
    print("For demonstration, we'll show the configuration setup.")
    
    # Show how to call the function (commented out to avoid API calls)
    # results = extract_uncertainty_expressions(sample_data, config, output_file)
    
    print(f"Configuration created for uncertainty extraction:")
    print(f"- Input file: {input_file}")
    print(f"- Output file: {output_file}")
    print(f"- Checkpoint directory: {config.checkpoint_dir}")

def example_3_previous_studies():
    """Example 3: Find and link previous studies."""
    print("="*60)
    print("Example 3: Previous Studies Identification")
    print("="*60)
    
    # Sample data with multiple studies per patient
    sample_data = [
        {
            "study_id": "s001",
            "patient_id": "p001",
            "findings": "Normal chest X-ray.",
            "impression": "No acute findings.",
            "study_date": "2023-01-15",
            "study_time": "14:30:00"
        },
        {
            "study_id": "s002",
            "patient_id": "p001", 
            "findings": "Mild cardiomegaly compared to prior.",
            "impression": "Enlarged heart.",
            "study_date": "2023-06-20",
            "study_time": "09:15:00"
        },
        {
            "study_id": "s003",
            "patient_id": "p002",
            "findings": "Clear lungs.",
            "impression": "Normal.",
            "study_date": "2023-03-10",
            "study_time": "11:00:00"
        }
    ]
    
    input_file = "data/previous_studies_sample.jsonl"
    output_file = "data/previous_studies_output.jsonl"
    
    # Save sample data
    save_jsonl(sample_data, input_file)
    
    # Find previous studies
    find_previous_studies(
        input_file=input_file,
        output_file=output_file,
        max_lookback_days=365,
        min_time_gap_hours=1,
        include_reports=True,
        validate_input=False  # Skip validation for demo
    )
    
    # Load and display results
    results = load_jsonl(output_file)
    
    print("\nResults with previous study information:")
    for i, result in enumerate(results):
        print(f"\nStudy {i+1}:")
        print(f"  Study ID: {result['study_id']}")
        print(f"  Patient ID: {result['patient_id']}")
        print(f"  Previous Study ID: {result.get('previous_study_id', 'None')}")
        print(f"  Days since previous: {result.get('days_since_previous', 'N/A')}")

def example_4_bounding_box_processing():
    """Example 4: Bounding box processing and grounding."""
    print("="*60)
    print("Example 4: Bounding Box Processing")
    print("="*60)
    
    # Sample bounding box data
    original_bbox = [100, 150, 300, 400]  # [x1, y1, x2, y2]
    original_size = (512, 512)
    target_size = (224, 224)
    
    print(f"Original bbox: {original_bbox}")
    print(f"Original image size: {original_size}")
    print(f"Target image size: {target_size}")
    
    # Scale bounding box
    scaled_bbox = scale_bbox(original_bbox, original_size, target_size)
    print(f"Scaled bbox: {scaled_bbox}")
    
    # Example of patch-bbox overlap calculation
    import numpy as np
    
    # Create sample uncertainty phrases and bboxes
    uncertainty_phrases = [
        {"phrase": "possible infiltrate", "bbox": scaled_bbox}
    ]
    
    bboxes = [scaled_bbox]
    
    # Calculate patch overlap
    uncertainty_mask, phrase_patch_pairs = calculate_patch_bbox_overlap(
        image_size=target_size,
        num_patches_side=14,
        bboxes=bboxes,
        uncertainty_phrases=uncertainty_phrases,
        overlap_threshold=0.05
    )
    
    print(f"\nUncertainty mask shape: {uncertainty_mask.shape}")
    print(f"Number of phrase-patch pairs: {len(phrase_patch_pairs)}")
    print(f"Patches with uncertainty: {np.sum(uncertainty_mask)}")

def example_5_full_pipeline():
    """Example 5: Running the full pipeline with configuration."""
    print("="*60)
    print("Example 5: Full Pipeline Configuration")
    print("="*60)
    
    # This example shows how to set up the full pipeline
    # (without actually running it to avoid API calls)
    
    config_example = {
        "data": {
            "input_file": "data/input.jsonl",
            "output_dir": "data/processed",
            "temp_dir": "data/temp"
        },
        "uncertainty": {
            "extract_expressions": True,
            "api_keys": ["your_api_key_1", "your_api_key_2"],
            "confidence_threshold": 0.8
        },
        "previous_studies": {
            "max_lookback_days": 365,
            "include_reports": True
        },
        "grounding": {
            "image_size": [224, 224],
            "num_patches_side": 14,
            "overlap_threshold": 0.05
        },
        "output": {
            "save_intermediate": True,
            "validate_output": True
        }
    }
    
    print("Example pipeline configuration:")
    import json
    print(json.dumps(config_example, indent=2))
    
    print("\nTo run the full pipeline:")
    print("python scripts/process_full_pipeline.py --config config/pipeline_config.yaml")

def main():
    """Run all examples."""
    print("CURV Data Processing Pipeline - Usage Examples")
    print("=" * 80)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Run examples
    try:
        example_1_basic_data_exploration()
        print("\n" + "="*80 + "\n")
        
        example_2_uncertainty_extraction()
        print("\n" + "="*80 + "\n")
        
        example_3_previous_studies()
        print("\n" + "="*80 + "\n")
        
        example_4_bounding_box_processing()
        print("\n" + "="*80 + "\n")
        
        example_5_full_pipeline()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Some examples may require additional setup (API keys, data files, etc.)")
    
    print("\n" + "="*80)
    print("Examples completed! Check the 'data/' directory for output files.")
    print("For more information, see the documentation in each module.")

if __name__ == "__main__":
    main()