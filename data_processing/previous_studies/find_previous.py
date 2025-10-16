#!/usr/bin/env python3
"""
Previous Studies Finder Module

This module provides functionality to identify and link previous studies for each patient
in medical imaging datasets, particularly useful for longitudinal analysis.

Author: CURV Team
"""

import json
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Use relative imports
from ..utils.data_io import load_jsonl, save_jsonl, validate_jsonl_format

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

class PreviousStudyFinder:
    """Class for finding and linking previous studies for patients."""
    
    def __init__(self, max_lookback_days: int = 365, min_time_gap_hours: int = 1):
        """
        Initialize the previous study finder.
        
        Args:
            max_lookback_days: Maximum days to look back for previous studies
            min_time_gap_hours: Minimum time gap between studies in hours
        """
        self.max_lookback_days = max_lookback_days
        self.min_time_gap_hours = min_time_gap_hours
    
    def parse_datetime(self, date_str: str, time_str: str) -> datetime:
        """
        Parse date and time strings into a datetime object.
        
        Args:
            date_str: Date string (YYYY-MM-DD format)
            time_str: Time string (HH:MM:SS format)
            
        Returns:
            Parsed datetime object
        """
        if not date_str or not time_str:
            return datetime.min
        
        # Try different datetime formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M",
            "%Y%m%d %H%M%S",  # DICOM format
            "%Y%m%d %H%M%S.%f"
        ]
        
        datetime_str = f"{date_str} {time_str}"
        
        for fmt in formats:
            try:
                return datetime.strptime(datetime_str, fmt)
            except ValueError:
                continue
        
        # If all formats fail, try parsing just the date
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            try:
                return datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                print(f"Warning: Could not parse datetime: {date_str} {time_str}")
                return datetime.min
    
    def extract_study_datetime(self, entry: Dict[str, Any]) -> datetime:
        """
        Extract study datetime from a data entry.
        
        Args:
            entry: Data entry dictionary
            
        Returns:
            Study datetime
        """
        # Try different possible locations for date/time information
        dicom_metadata = entry.get('dicom_metadata', {})
        
        # Primary: DICOM metadata
        study_date = dicom_metadata.get('study_date', '')
        study_time = dicom_metadata.get('study_time', '')
        
        if study_date and study_time:
            return self.parse_datetime(study_date, study_time)
        
        # Secondary: Direct fields
        study_date = entry.get('study_date', '')
        study_time = entry.get('study_time', '')
        
        if study_date and study_time:
            return self.parse_datetime(study_date, study_time)
        
        # Tertiary: Combined datetime field
        study_datetime = entry.get('study_datetime', '')
        if study_datetime:
            try:
                return datetime.fromisoformat(study_datetime.replace('Z', '+00:00'))
            except ValueError:
                pass
        
        # Last resort: Use study_date only
        if study_date:
            return self.parse_datetime(study_date, '00:00:00')
        
        return datetime.min
    
    def group_studies_by_patient(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group studies by patient ID and sort by datetime.
        
        Args:
            data: List of study entries
            
        Returns:
            Dictionary mapping patient_id to sorted list of studies
        """
        patient_studies = defaultdict(list)
        
        for entry in data:
            patient_id = entry.get('patient_id')
            if not patient_id:
                continue
            
            # Add datetime for sorting
            entry_with_datetime = entry.copy()
            entry_with_datetime['_study_datetime'] = self.extract_study_datetime(entry)
            
            patient_studies[patient_id].append(entry_with_datetime)
        
        # Sort studies by datetime for each patient
        for patient_id in patient_studies:
            patient_studies[patient_id].sort(key=lambda x: x['_study_datetime'])
        
        return dict(patient_studies)
    
    def find_previous_study(self, current_study: Dict[str, Any], 
                           patient_studies: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find the most recent previous study for a given study.
        
        Args:
            current_study: Current study entry
            patient_studies: All studies for the patient, sorted by datetime
            
        Returns:
            Previous study entry or None if not found
        """
        current_datetime = current_study['_study_datetime']
        current_study_id = current_study.get('study_id')
        
        # Find studies before the current one
        previous_studies = []
        for study in patient_studies:
            study_datetime = study['_study_datetime']
            study_id = study.get('study_id')
            
            # Skip if it's the same study or after current study
            if study_id == current_study_id or study_datetime >= current_datetime:
                continue
            
            # Check time constraints
            time_diff = current_datetime - study_datetime
            
            # Must be at least min_time_gap_hours apart
            if time_diff.total_seconds() < self.min_time_gap_hours * 3600:
                continue
            
            # Must be within max_lookback_days
            if time_diff.days > self.max_lookback_days:
                continue
            
            previous_studies.append(study)
        
        # Return the most recent previous study
        if previous_studies:
            return max(previous_studies, key=lambda x: x['_study_datetime'])
        
        return None
    
    def process_studies(self, data: List[Dict[str, Any]], 
                       include_reports: bool = True,
                       include_images: bool = False) -> List[Dict[str, Any]]:
        """
        Process studies to add previous study information.
        
        Args:
            data: List of study entries
            include_reports: Whether to include previous study reports
            include_images: Whether to include previous study images
            
        Returns:
            List of studies with previous study information added
        """
        print("Grouping studies by patient...")
        patient_studies = self.group_studies_by_patient(data)
        
        print(f"Found {len(patient_studies)} unique patients")
        
        results = []
        total_studies = len(data)
        studies_with_previous = 0
        
        print("Processing studies to find previous studies...")
        
        iterator = tqdm(data, desc="Processing studies") if HAS_TQDM else data
        
        for entry in iterator:
            patient_id = entry.get('patient_id')
            if not patient_id:
                # Add empty previous study fields
                result_entry = entry.copy()
                result_entry['previous_study_id'] = ''
                if include_reports:
                    result_entry['previous_findings'] = ''
                    result_entry['previous_impression'] = ''
                if include_images:
                    result_entry['previous_image_path'] = ''
                results.append(result_entry)
                continue
            
            # Find previous study
            previous_study = self.find_previous_study(entry, patient_studies[patient_id])
            
            # Create result entry
            result_entry = entry.copy()
            
            if previous_study:
                result_entry['previous_study_id'] = previous_study.get('study_id', '')
                studies_with_previous += 1
                
                if include_reports:
                    result_entry['previous_findings'] = previous_study.get('findings', '')
                    result_entry['previous_impression'] = previous_study.get('impression', '')
                
                if include_images:
                    result_entry['previous_image_path'] = previous_study.get('image_path', '')
                
                # Add time difference information
                current_datetime = entry.get('_study_datetime', self.extract_study_datetime(entry))
                previous_datetime = previous_study['_study_datetime']
                time_diff = current_datetime - previous_datetime
                result_entry['days_since_previous'] = time_diff.days
            else:
                result_entry['previous_study_id'] = ''
                if include_reports:
                    result_entry['previous_findings'] = ''
                    result_entry['previous_impression'] = ''
                if include_images:
                    result_entry['previous_image_path'] = ''
                result_entry['days_since_previous'] = -1
            
            # Remove temporary datetime field
            if '_study_datetime' in result_entry:
                del result_entry['_study_datetime']
            
            results.append(result_entry)
        
        print(f"Processing complete: {studies_with_previous}/{total_studies} studies have previous studies")
        print(f"Coverage: {(studies_with_previous/total_studies)*100:.1f}%")
        
        return results

def validate_study_data(data: List[Dict[str, Any]], num_samples: int = 10) -> bool:
    """
    Validate study data for required fields and format.
    
    Args:
        data: List of study entries
        num_samples: Number of samples to check
        
    Returns:
        True if validation passes
    """
    if not data:
        print("Error: No data provided")
        return False
    
    print(f"Validating {min(num_samples, len(data))} sample entries...")
    
    required_fields = ['patient_id', 'study_id']
    issues_found = False
    
    for i, entry in enumerate(data[:num_samples]):
        # Check required fields
        missing_fields = [field for field in required_fields if field not in entry]
        if missing_fields:
            print(f"Warning: Entry {i+1} missing required fields: {missing_fields}")
            issues_found = True
        
        # Check datetime information
        finder = PreviousStudyFinder()
        study_datetime = finder.extract_study_datetime(entry)
        if study_datetime == datetime.min:
            print(f"Warning: Entry {i+1} has no valid datetime information")
            issues_found = True
        
        # Print sample info
        print(f"Sample {i+1}:")
        print(f"  patient_id: {entry.get('patient_id', 'MISSING')}")
        print(f"  study_id: {entry.get('study_id', 'MISSING')}")
        print(f"  study_datetime: {study_datetime}")
        print()
    
    if issues_found:
        print("Issues found in validation. Please review the data.")
        return False
    else:
        print("Validation passed successfully.")
        return True

def find_previous_studies(input_file: str, output_file: str,
                         max_lookback_days: int = 365,
                         min_time_gap_hours: int = 1,
                         include_reports: bool = True,
                         include_images: bool = False,
                         validate_input: bool = True) -> None:
    """
    Main function to find and add previous studies to a JSONL file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        max_lookback_days: Maximum days to look back for previous studies
        min_time_gap_hours: Minimum time gap between studies in hours
        include_reports: Whether to include previous study reports
        include_images: Whether to include previous study images
        validate_input: Whether to validate input data
    """
    # Validate input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if not validate_jsonl_format(input_file):
        raise ValueError(f"Invalid JSONL format: {input_file}")
    
    # Load data
    print(f"Loading data from {input_file}...")
    data = load_jsonl(input_file)
    print(f"Loaded {len(data)} entries")
    
    # Validate data if requested
    if validate_input and not validate_study_data(data):
        response = input("Validation failed. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting as requested.")
            return
    
    # Initialize finder
    finder = PreviousStudyFinder(
        max_lookback_days=max_lookback_days,
        min_time_gap_hours=min_time_gap_hours
    )
    
    # Process studies
    results = finder.process_studies(
        data,
        include_reports=include_reports,
        include_images=include_images
    )
    
    # Save results
    print(f"Saving results to {output_file}...")
    save_jsonl(results, output_file)
    print("Processing complete!")

def analyze_previous_study_coverage(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the coverage and distribution of previous studies.
    
    Args:
        data: List of processed study entries
        
    Returns:
        Analysis results
    """
    total_studies = len(data)
    studies_with_previous = sum(1 for entry in data if entry.get('previous_study_id'))
    
    # Analyze time gaps
    time_gaps = []
    for entry in data:
        days_since = entry.get('days_since_previous', -1)
        if days_since > 0:
            time_gaps.append(days_since)
    
    # Patient-level analysis
    patient_coverage = defaultdict(list)
    for entry in data:
        patient_id = entry.get('patient_id')
        has_previous = bool(entry.get('previous_study_id'))
        patient_coverage[patient_id].append(has_previous)
    
    patients_with_any_previous = sum(1 for studies in patient_coverage.values() 
                                   if any(studies))
    
    analysis = {
        'total_studies': total_studies,
        'studies_with_previous': studies_with_previous,
        'coverage_percentage': (studies_with_previous / total_studies * 100) if total_studies > 0 else 0,
        'total_patients': len(patient_coverage),
        'patients_with_any_previous': patients_with_any_previous,
        'patient_coverage_percentage': (patients_with_any_previous / len(patient_coverage) * 100) if patient_coverage else 0,
        'time_gap_stats': {
            'mean_days': sum(time_gaps) / len(time_gaps) if time_gaps else 0,
            'min_days': min(time_gaps) if time_gaps else 0,
            'max_days': max(time_gaps) if time_gaps else 0,
            'median_days': sorted(time_gaps)[len(time_gaps)//2] if time_gaps else 0
        }
    }
    
    return analysis

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Find previous studies for medical imaging data")
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("output_file", help="Output JSONL file path")
    parser.add_argument("--max-lookback-days", type=int, default=365,
                       help="Maximum days to look back for previous studies")
    parser.add_argument("--min-time-gap-hours", type=int, default=1,
                       help="Minimum time gap between studies in hours")
    parser.add_argument("--include-reports", action='store_true', default=True,
                       help="Include previous study reports")
    parser.add_argument("--include-images", action='store_true',
                       help="Include previous study images")
    parser.add_argument("--no-validation", action='store_true',
                       help="Skip input data validation")
    parser.add_argument("--analyze-coverage", action='store_true',
                       help="Analyze previous study coverage after processing")
    
    args = parser.parse_args()
    
    # Process the data
    find_previous_studies(
        input_file=args.input_file,
        output_file=args.output_file,
        max_lookback_days=args.max_lookback_days,
        min_time_gap_hours=args.min_time_gap_hours,
        include_reports=args.include_reports,
        include_images=args.include_images,
        validate_input=not args.no_validation
    )
    
    # Analyze coverage if requested
    if args.analyze_coverage:
        print("\nAnalyzing previous study coverage...")
        data = load_jsonl(args.output_file)
        analysis = analyze_previous_study_coverage(data)
        
        print("\nPrevious Study Coverage Analysis:")
        print(f"Total studies: {analysis['total_studies']:,}")
        print(f"Studies with previous: {analysis['studies_with_previous']:,} ({analysis['coverage_percentage']:.1f}%)")
        print(f"Total patients: {analysis['total_patients']:,}")
        print(f"Patients with any previous: {analysis['patients_with_any_previous']:,} ({analysis['patient_coverage_percentage']:.1f}%)")
        print(f"Time gap statistics:")
        print(f"  Mean: {analysis['time_gap_stats']['mean_days']:.1f} days")
        print(f"  Median: {analysis['time_gap_stats']['median_days']} days")
        print(f"  Range: {analysis['time_gap_stats']['min_days']}-{analysis['time_gap_stats']['max_days']} days")