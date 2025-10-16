#!/usr/bin/env python3
"""
Data validation utilities for CURV data processing.

This module provides functions for validating data formats,
checking data integrity, and ensuring data quality.

Author: CURV Team
"""

import json
import logging
from typing import List, Dict, Any, Optional, Set, Union
from pathlib import Path
import re

logger = logging.getLogger(__name__)

def validate_data_format(
    data: Union[Dict[str, Any], List[Dict[str, Any]]], 
    required_fields: List[str] = None,
    optional_fields: List[str] = None,
    field_types: Dict[str, type] = None
) -> Dict[str, Any]:
    """
    Validate data format and structure.
    
    Args:
        data: Data to validate (single dict or list of dicts)
        required_fields: List of required field names
        optional_fields: List of optional field names
        field_types: Dictionary mapping field names to expected types
        
    Returns:
        Validation report dictionary
    """
    if required_fields is None:
        required_fields = ['study_id']
    
    if field_types is None:
        field_types = {
            'study_id': str,
            'patient_id': str,
            'findings': str,
            'impression': str
        }
    
    # Convert single dict to list for uniform processing
    if isinstance(data, dict):
        data = [data]
    
    validation_report = {
        'total_items': len(data),
        'valid_items': 0,
        'invalid_items': 0,
        'missing_required_fields': {},
        'type_errors': {},
        'validation_errors': []
    }
    
    for i, item in enumerate(data):
        is_valid = True
        item_errors = []
        
        # Check required fields
        for field in required_fields:
            if field not in item:
                is_valid = False
                item_errors.append(f"Missing required field: {field}")
                
                if field not in validation_report['missing_required_fields']:
                    validation_report['missing_required_fields'][field] = 0
                validation_report['missing_required_fields'][field] += 1
        
        # Check field types
        for field, expected_type in field_types.items():
            if field in item and item[field] is not None:
                if not isinstance(item[field], expected_type):
                    is_valid = False
                    item_errors.append(f"Type error for field {field}: expected {expected_type.__name__}, got {type(item[field]).__name__}")
                    
                    if field not in validation_report['type_errors']:
                        validation_report['type_errors'][field] = 0
                    validation_report['type_errors'][field] += 1
        
        if is_valid:
            validation_report['valid_items'] += 1
        else:
            validation_report['invalid_items'] += 1
            validation_report['validation_errors'].append({
                'item_index': i,
                'errors': item_errors
            })
    
    # Calculate validation statistics
    if validation_report['total_items'] > 0:
        validation_report['validity_percentage'] = (validation_report['valid_items'] / validation_report['total_items']) * 100
    else:
        validation_report['validity_percentage'] = 0
    
    logger.info(f"Validation complete: {validation_report['valid_items']}/{validation_report['total_items']} items valid ({validation_report['validity_percentage']:.1f}%)")
    
    return validation_report

def check_data_integrity(
    data: List[Dict[str, Any]],
    integrity_checks: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Check data integrity and quality.
    
    Args:
        data: Data to check
        integrity_checks: Dictionary with integrity check configurations
        
    Returns:
        Integrity check report
    """
    if integrity_checks is None:
        integrity_checks = {
            'check_duplicates': True,
            'check_empty_fields': True,
            'check_text_quality': True,
            'min_text_length': 10,
            'max_text_length': 10000
        }
    
    integrity_report = {
        'total_items': len(data),
        'duplicate_count': 0,
        'empty_field_count': 0,
        'text_quality_issues': 0,
        'integrity_issues': []
    }
    
    # Check for duplicates
    if integrity_checks.get('check_duplicates', False):
        seen_items = set()
        for i, item in enumerate(data):
            # Create a simple hash of the item
            item_str = json.dumps(item, sort_keys=True)
            if item_str in seen_items:
                integrity_report['duplicate_count'] += 1
                integrity_report['integrity_issues'].append({
                    'item_index': i,
                    'issue_type': 'duplicate',
                    'description': 'Duplicate item found'
                })
            else:
                seen_items.add(item_str)
    
    # Check for empty fields
    if integrity_checks.get('check_empty_fields', False):
        text_fields = ['findings', 'impression', 'indication', 'report']
        for i, item in enumerate(data):
            for field in text_fields:
                if field in item:
                    value = item[field]
                    if not value or (isinstance(value, str) and not value.strip()):
                        integrity_report['empty_field_count'] += 1
                        integrity_report['integrity_issues'].append({
                            'item_index': i,
                            'issue_type': 'empty_field',
                            'field': field,
                            'description': f'Empty or whitespace-only field: {field}'
                        })
    
    # Check text quality
    if integrity_checks.get('check_text_quality', False):
        min_length = integrity_checks.get('min_text_length', 10)
        max_length = integrity_checks.get('max_text_length', 10000)
        text_fields = ['findings', 'impression', 'indication', 'report']
        
        for i, item in enumerate(data):
            for field in text_fields:
                if field in item and isinstance(item[field], str):
                    text = item[field].strip()
                    
                    # Check text length
                    if len(text) < min_length:
                        integrity_report['text_quality_issues'] += 1
                        integrity_report['integrity_issues'].append({
                            'item_index': i,
                            'issue_type': 'text_too_short',
                            'field': field,
                            'description': f'Text too short in {field}: {len(text)} < {min_length}'
                        })
                    elif len(text) > max_length:
                        integrity_report['text_quality_issues'] += 1
                        integrity_report['integrity_issues'].append({
                            'item_index': i,
                            'issue_type': 'text_too_long',
                            'field': field,
                            'description': f'Text too long in {field}: {len(text)} > {max_length}'
                        })
    
    logger.info(f"Integrity check complete: found {len(integrity_report['integrity_issues'])} issues")
    return integrity_report

def validate_study_ids(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate study ID formats and uniqueness.
    
    Args:
        data: Data to validate
        
    Returns:
        Study ID validation report
    """
    validation_report = {
        'total_studies': len(data),
        'unique_study_ids': 0,
        'duplicate_study_ids': 0,
        'invalid_format_count': 0,
        'missing_study_ids': 0,
        'validation_issues': []
    }
    
    study_id_counts = {}
    study_id_pattern = re.compile(r'^[A-Za-z0-9_-]+$')  # Basic alphanumeric pattern
    
    for i, item in enumerate(data):
        if 'study_id' not in item:
            validation_report['missing_study_ids'] += 1
            validation_report['validation_issues'].append({
                'item_index': i,
                'issue_type': 'missing_study_id',
                'description': 'Missing study_id field'
            })
            continue
        
        study_id = item['study_id']
        
        # Check format
        if not isinstance(study_id, str) or not study_id_pattern.match(study_id):
            validation_report['invalid_format_count'] += 1
            validation_report['validation_issues'].append({
                'item_index': i,
                'issue_type': 'invalid_format',
                'study_id': study_id,
                'description': f'Invalid study_id format: {study_id}'
            })
        
        # Count occurrences
        if study_id in study_id_counts:
            study_id_counts[study_id] += 1
        else:
            study_id_counts[study_id] = 1
    
    # Identify duplicates
    for study_id, count in study_id_counts.items():
        if count > 1:
            validation_report['duplicate_study_ids'] += count
            validation_report['validation_issues'].append({
                'issue_type': 'duplicate_study_id',
                'study_id': study_id,
                'count': count,
                'description': f'Duplicate study_id: {study_id} appears {count} times'
            })
    
    validation_report['unique_study_ids'] = len(study_id_counts)
    
    logger.info(f"Study ID validation: {validation_report['unique_study_ids']} unique IDs, {validation_report['duplicate_study_ids']} duplicates")
    return validation_report

def validate_patient_ids(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate patient ID formats and consistency.
    
    Args:
        data: Data to validate
        
    Returns:
        Patient ID validation report
    """
    validation_report = {
        'total_studies': len(data),
        'unique_patient_ids': 0,
        'missing_patient_ids': 0,
        'invalid_format_count': 0,
        'patient_study_counts': {},
        'validation_issues': []
    }
    
    patient_id_counts = {}
    patient_id_pattern = re.compile(r'^[A-Za-z0-9_-]+$')  # Basic alphanumeric pattern
    
    for i, item in enumerate(data):
        if 'patient_id' not in item:
            validation_report['missing_patient_ids'] += 1
            validation_report['validation_issues'].append({
                'item_index': i,
                'issue_type': 'missing_patient_id',
                'description': 'Missing patient_id field'
            })
            continue
        
        patient_id = item['patient_id']
        
        # Check format
        if not isinstance(patient_id, str) or not patient_id_pattern.match(patient_id):
            validation_report['invalid_format_count'] += 1
            validation_report['validation_issues'].append({
                'item_index': i,
                'issue_type': 'invalid_format',
                'patient_id': patient_id,
                'description': f'Invalid patient_id format: {patient_id}'
            })
        
        # Count studies per patient
        if patient_id in patient_id_counts:
            patient_id_counts[patient_id] += 1
        else:
            patient_id_counts[patient_id] = 1
    
    validation_report['unique_patient_ids'] = len(patient_id_counts)
    validation_report['patient_study_counts'] = patient_id_counts
    
    logger.info(f"Patient ID validation: {validation_report['unique_patient_ids']} unique patients")
    return validation_report

def validate_text_fields(
    data: List[Dict[str, Any]], 
    text_fields: List[str] = None
) -> Dict[str, Any]:
    """
    Validate text field content and quality.
    
    Args:
        data: Data to validate
        text_fields: List of text fields to validate
        
    Returns:
        Text validation report
    """
    if text_fields is None:
        text_fields = ['findings', 'impression', 'indication']
    
    validation_report = {
        'total_items': len(data),
        'field_statistics': {},
        'validation_issues': []
    }
    
    for field in text_fields:
        validation_report['field_statistics'][field] = {
            'present_count': 0,
            'missing_count': 0,
            'empty_count': 0,
            'avg_length': 0,
            'min_length': float('inf'),
            'max_length': 0,
            'total_length': 0
        }
    
    for i, item in enumerate(data):
        for field in text_fields:
            stats = validation_report['field_statistics'][field]
            
            if field not in item:
                stats['missing_count'] += 1
                validation_report['validation_issues'].append({
                    'item_index': i,
                    'issue_type': 'missing_field',
                    'field': field,
                    'description': f'Missing text field: {field}'
                })
            elif not item[field] or (isinstance(item[field], str) and not item[field].strip()):
                stats['empty_count'] += 1
                validation_report['validation_issues'].append({
                    'item_index': i,
                    'issue_type': 'empty_field',
                    'field': field,
                    'description': f'Empty text field: {field}'
                })
            else:
                stats['present_count'] += 1
                text_length = len(str(item[field]))
                stats['total_length'] += text_length
                stats['min_length'] = min(stats['min_length'], text_length)
                stats['max_length'] = max(stats['max_length'], text_length)
    
    # Calculate averages
    for field in text_fields:
        stats = validation_report['field_statistics'][field]
        if stats['present_count'] > 0:
            stats['avg_length'] = stats['total_length'] / stats['present_count']
        if stats['min_length'] == float('inf'):
            stats['min_length'] = 0
    
    logger.info(f"Text field validation complete for {len(text_fields)} fields")
    return validation_report