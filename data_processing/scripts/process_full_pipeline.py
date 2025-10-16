#!/usr/bin/env python3
"""
Full CURV Data Processing Pipeline

This script runs the complete data processing pipeline for CURV, including:
1. Uncertainty expression extraction
2. Previous study identification and integration
3. Visual grounding data preparation
4. Data validation and quality checks

Author: CURV Team
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_io import load_jsonl, save_jsonl, validate_jsonl_format
from uncertainty.extract_uncertainty import extract_uncertainty_expressions, UncertaintyConfig
from grounding.bbox_utils import calculate_patch_bbox_overlap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CURVPipelineConfig:
    """Configuration for the CURV data processing pipeline."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.data = config_dict.get('data', {})
        self.uncertainty = config_dict.get('uncertainty', {})
        self.grounding = config_dict.get('grounding', {})
        self.previous_studies = config_dict.get('previous_studies', {})
        self.output = config_dict.get('output', {})
        
        # Set defaults
        self._set_defaults()
    
    def _set_defaults(self):
        """Set default values for configuration."""
        # Data defaults
        self.data.setdefault('input_file', 'data/input.jsonl')
        self.data.setdefault('output_dir', 'data/processed')
        self.data.setdefault('temp_dir', 'data/temp')
        
        # Uncertainty defaults
        self.uncertainty.setdefault('extract_expressions', True)
        self.uncertainty.setdefault('confidence_threshold', 0.8)
        self.uncertainty.setdefault('api_keys', [])
        
        # Grounding defaults
        self.grounding.setdefault('bbox_format', 'xyxy')
        self.grounding.setdefault('image_size', [224, 224])
        self.grounding.setdefault('num_patches_side', 14)
        self.grounding.setdefault('overlap_threshold', 0.05)
        
        # Previous studies defaults
        self.previous_studies.setdefault('max_lookback_days', 365)
        self.previous_studies.setdefault('include_reports', True)
        
        # Output defaults
        self.output.setdefault('save_intermediate', True)
        self.output.setdefault('validate_output', True)

def load_config(config_path: str) -> CURVPipelineConfig:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using defaults")
        return CURVPipelineConfig({})
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return CURVPipelineConfig(config_dict)

def validate_input_data(data: List[Dict[str, Any]]) -> bool:
    """Validate input data format and required fields."""
    required_fields = ['study_id', 'patient_id']
    optional_fields = ['findings', 'impression', 'image_path']
    
    if not data:
        logger.error("Input data is empty")
        return False
    
    # Check first few samples
    for i, item in enumerate(data[:10]):
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            logger.error(f"Sample {i} missing required fields: {missing_fields}")
            return False
    
    logger.info(f"Input data validation passed for {len(data)} items")
    return True

def process_uncertainty_stage(
    data: List[Dict[str, Any]], 
    config: CURVPipelineConfig,
    output_dir: str
) -> List[Dict[str, Any]]:
    """Process uncertainty extraction stage."""
    if not config.uncertainty['extract_expressions']:
        logger.info("Skipping uncertainty extraction (disabled in config)")
        return data
    
    logger.info("Starting uncertainty extraction...")
    
    # Create uncertainty config
    uncertainty_config = UncertaintyConfig(
        api_keys=config.uncertainty['api_keys'],
        checkpoint_dir=os.path.join(output_dir, 'uncertainty_checkpoints')
    )
    
    # Extract uncertainty expressions
    uncertainty_output = os.path.join(output_dir, 'data_with_uncertainty.jsonl')
    results = extract_uncertainty_expressions(data, uncertainty_config, uncertainty_output)
    
    logger.info(f"Uncertainty extraction completed. Results saved to {uncertainty_output}")
    return results

def process_previous_studies_stage(
    data: List[Dict[str, Any]], 
    config: CURVPipelineConfig,
    output_dir: str
) -> List[Dict[str, Any]]:
    """Process previous studies identification stage."""
    logger.info("Starting previous studies processing...")
    
    # Group data by patient
    patient_studies = {}
    for item in data:
        patient_id = item.get('patient_id')
        if patient_id:
            if patient_id not in patient_studies:
                patient_studies[patient_id] = []
            patient_studies[patient_id].append(item)
    
    # Sort studies by date for each patient
    for patient_id in patient_studies:
        patient_studies[patient_id].sort(
            key=lambda x: x.get('study_date', ''), 
            reverse=False
        )
    
    # Add previous study information
    results = []
    for patient_id, studies in patient_studies.items():
        for i, study in enumerate(studies):
            study_copy = study.copy()
            
            # Add previous study ID if available
            if i > 0:
                previous_study = studies[i-1]
                study_copy['previous_study_id'] = previous_study.get('study_id', '')
                
                if config.previous_studies['include_reports']:
                    study_copy['previous_findings'] = previous_study.get('findings', '')
                    study_copy['previous_impression'] = previous_study.get('impression', '')
            else:
                study_copy['previous_study_id'] = ''
                if config.previous_studies['include_reports']:
                    study_copy['previous_findings'] = ''
                    study_copy['previous_impression'] = ''
            
            results.append(study_copy)
    
    # Save intermediate results
    if config.output['save_intermediate']:
        previous_output = os.path.join(output_dir, 'data_with_previous_studies.jsonl')
        save_jsonl(results, previous_output)
        logger.info(f"Previous studies processing completed. Results saved to {previous_output}")
    
    return results

def process_grounding_stage(
    data: List[Dict[str, Any]], 
    config: CURVPipelineConfig,
    output_dir: str
) -> List[Dict[str, Any]]:
    """Process visual grounding data preparation stage."""
    logger.info("Starting grounding data preparation...")
    
    image_size = tuple(config.grounding['image_size'])
    num_patches_side = config.grounding['num_patches_side']
    overlap_threshold = config.grounding['overlap_threshold']
    
    results = []
    for item in data:
        item_copy = item.copy()
        
        # Process bounding boxes if available
        if 'bbox_annotations' in item and 'uncertainty_annotations' in item:
            bboxes = item['bbox_annotations']
            uncertainty_phrases = item['uncertainty_annotations']
            
            # Calculate patch-bbox overlap
            uncertainty_mask, phrase_patch_pairs = calculate_patch_bbox_overlap(
                image_size=image_size,
                num_patches_side=num_patches_side,
                bboxes=bboxes,
                uncertainty_phrases=uncertainty_phrases,
                overlap_threshold=overlap_threshold
            )
            
            item_copy['uncertainty_mask'] = uncertainty_mask.tolist()
            item_copy['phrase_patch_pairs'] = phrase_patch_pairs
        
        results.append(item_copy)
    
    # Save intermediate results
    if config.output['save_intermediate']:
        grounding_output = os.path.join(output_dir, 'data_with_grounding.jsonl')
        save_jsonl(results, grounding_output)
        logger.info(f"Grounding data preparation completed. Results saved to {grounding_output}")
    
    return results

def validate_output_data(data: List[Dict[str, Any]], config: CURVPipelineConfig) -> bool:
    """Validate final output data."""
    if not config.output['validate_output']:
        return True
    
    logger.info("Validating output data...")
    
    # Check for required fields based on processing stages
    required_fields = ['study_id', 'patient_id']
    
    if config.uncertainty['extract_expressions']:
        required_fields.extend(['findings_uncertainty', 'impression_uncertainty'])
    
    # Validate samples
    validation_errors = 0
    for i, item in enumerate(data[:100]):  # Check first 100 items
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            logger.warning(f"Item {i} missing fields: {missing_fields}")
            validation_errors += 1
    
    if validation_errors > 0:
        logger.warning(f"Found {validation_errors} validation errors in output data")
    else:
        logger.info("Output data validation passed")
    
    return validation_errors == 0

def main():
    """Main function for running the CURV data processing pipeline."""
    parser = argparse.ArgumentParser(description="CURV Data Processing Pipeline")
    parser.add_argument("--config", required=True, help="Configuration YAML file")
    parser.add_argument("--input", help="Input JSONL file (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--stage", choices=['uncertainty', 'previous', 'grounding', 'all'], 
                       default='all', help="Processing stage to run")
    parser.add_argument("--validate-input", action='store_true', 
                       help="Validate input data format")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.input:
        config.data['input_file'] = args.input
    if args.output_dir:
        config.data['output_dir'] = args.output_dir
    
    # Create output directory
    output_dir = config.data['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input data
    input_file = config.data['input_file']
    logger.info(f"Loading data from {input_file}")
    
    if not validate_jsonl_format(input_file):
        logger.error("Input file is not valid JSONL format")
        return 1
    
    data = load_jsonl(input_file)
    
    # Validate input data
    if args.validate_input and not validate_input_data(data):
        logger.error("Input data validation failed")
        return 1
    
    # Process stages
    if args.stage in ['uncertainty', 'all']:
        data = process_uncertainty_stage(data, config, output_dir)
    
    if args.stage in ['previous', 'all']:
        data = process_previous_studies_stage(data, config, output_dir)
    
    if args.stage in ['grounding', 'all']:
        data = process_grounding_stage(data, config, output_dir)
    
    # Save final results
    final_output = os.path.join(output_dir, 'final_processed_data.jsonl')
    save_jsonl(data, final_output)
    
    # Validate output
    if not validate_output_data(data, config):
        logger.warning("Output data validation failed")
        return 1
    
    logger.info(f"Pipeline completed successfully. Final output: {final_output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())