#!/usr/bin/env python3
"""
Uncertainty Expression Extraction for CURV

This module extracts uncertainty expressions from radiology reports using LLM-based analysis.
It identifies both structural uncertainty (in findings) and semantic uncertainty (in impressions).

Author: CURV Team
"""

import json
import re
import time
import os
import logging
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import multiprocessing
from multiprocessing import Process, Queue, Value, Lock, Manager
import threading
import queue
import random
import pickle
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common uncertainty expressions in medical radiology reports
COMMON_UNCERTAINTY_TERMS = [
    "likely", "possibly", "suggests", "may", "might", "could", "probable", 
    "potential", "suspected", "suspicious for", "consistent with", "concerning for",
    "cannot exclude", "cannot rule out", "versus", "vs", "suspicious", "suggestive of",
    "possibility of", "questionable", "equivocal", "uncertain", "unclear", "indeterminate",
    "presumed", "probable", "possible", "suggestion of", "not excluded", "presumably",
    "appears", "apparent", "compatible with", "can be", "not ruled out", "perhaps"
]

@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty extraction."""
    api_keys: List[str]
    api_base: str = "https://api.siliconflow.cn/v1"
    rpm_limit: int = 800
    tpm_limit: int = 40000
    max_workers: int = 16
    checkpoint_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    max_retries: int = 5

class RateLimiter:
    """Manages API rate limits per key with exponential backoff."""
    
    def __init__(self, rpm_limit: int, tpm_limit: int):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.key_usage = {}
        self.lock = threading.Lock()
        self.backoff_times = {}
        self.key_status = {}
    
    def wait_if_needed(self, api_key: str, estimated_tokens: int = 100) -> None:
        """Wait if necessary to respect rate limits for the given API key."""
        with self.lock:
            current_time = time.time()
            
            # Initialize tracking for this key if not already done
            if api_key not in self.key_usage:
                self.key_usage[api_key] = {
                    'request_timestamps': [],
                    'token_usage': []
                }
                self.backoff_times[api_key] = 0
                self.key_status[api_key] = "healthy"
            
            # Handle backoff period
            if self.key_status[api_key] == "cooling":
                if api_key in self.backoff_times and self.backoff_times[api_key] > 0:
                    latest_timestamp = max(self.key_usage[api_key]['request_timestamps'], default=current_time-3600)
                    elapsed = current_time - latest_timestamp
                    
                    if elapsed < self.backoff_times[api_key]:
                        sleep_time = self.backoff_times[api_key] - elapsed
                        if sleep_time > 0:
                            self.lock.release()
                            time.sleep(sleep_time)
                            self.lock.acquire()
                            current_time = time.time()
                    else:
                        self.key_status[api_key] = "healthy"
                        self.backoff_times[api_key] = 0
            
            # Clean old timestamps (older than 1 minute)
            minute_ago = current_time - 60
            self.key_usage[api_key]['request_timestamps'] = [
                ts for ts in self.key_usage[api_key]['request_timestamps'] if ts > minute_ago
            ]
            self.key_usage[api_key]['token_usage'] = [
                (ts, tokens) for ts, tokens in self.key_usage[api_key]['token_usage'] if ts > minute_ago
            ]
            
            # Check rate limits
            requests_last_minute = len(self.key_usage[api_key]['request_timestamps'])
            tokens_last_minute = sum(tokens for _, tokens in self.key_usage[api_key]['token_usage'])
            
            # Calculate wait time if needed
            wait_time = 0
            if requests_last_minute >= self.rpm_limit:
                oldest_request = min(self.key_usage[api_key]['request_timestamps'])
                wait_time = max(wait_time, 60 - (current_time - oldest_request))
            
            if tokens_last_minute + estimated_tokens > self.tpm_limit:
                oldest_token_time = min(ts for ts, _ in self.key_usage[api_key]['token_usage'])
                wait_time = max(wait_time, 60 - (current_time - oldest_token_time))
            
            if wait_time > 0:
                self.lock.release()
                time.sleep(wait_time)
                self.lock.acquire()
                current_time = time.time()
            
            # Record this request
            self.key_usage[api_key]['request_timestamps'].append(current_time)
            self.key_usage[api_key]['token_usage'].append((current_time, estimated_tokens))
    
    def handle_rate_limit_error(self, api_key: str) -> None:
        """Handle rate limit error with exponential backoff."""
        with self.lock:
            current_backoff = self.backoff_times.get(api_key, 1)
            new_backoff = min(current_backoff * 2, 300)  # Max 5 minutes
            self.backoff_times[api_key] = new_backoff
            self.key_status[api_key] = "cooling"
            logger.warning(f"Rate limit hit for key {api_key[:10]}..., backing off for {new_backoff} seconds")

def extract_uncertainty_with_llm(
    text: str, 
    client: Any, 
    api_key: str, 
    rate_limiter: RateLimiter,
    max_retries: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Extract uncertainty expressions from text using LLM.
    
    Args:
        text: Input text to analyze
        client: OpenAI client instance
        api_key: API key for the request
        rate_limiter: Rate limiter instance
        max_retries: Maximum number of retries
        
    Returns:
        Dictionary containing uncertainty analysis or None if failed
    """
    prompt = f'''
    Analyze the following medical text and identify uncertainty expressions. 
    Focus on words or phrases that indicate diagnostic uncertainty, probability, or hedging.
    
    Text: "{text}"
    
    Please provide a JSON response with:
    1. "uncertainty_phrases": List of identified uncertainty expressions
    2. "confidence_level": Overall confidence level (high/medium/low)
    3. "uncertainty_type": Type of uncertainty (structural/semantic/both)
    4. "reasoning": Brief explanation of the analysis
    
    Common uncertainty terms include: {', '.join(COMMON_UNCERTAINTY_TERMS[:10])}...
    '''
    
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed(api_key, estimated_tokens=len(prompt.split()) * 1.3)
            
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a medical text analysis expert specializing in uncertainty detection."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                # If not valid JSON, extract information manually
                return {
                    "uncertainty_phrases": extract_phrases_from_text(result_text),
                    "confidence_level": "medium",
                    "uncertainty_type": "unknown",
                    "reasoning": result_text,
                    "raw_response": result_text
                }
                
        except Exception as e:
            if "rate_limit" in str(e).lower():
                rate_limiter.handle_rate_limit_error(api_key)
                wait_time = min(2 ** attempt, 60)
                time.sleep(wait_time)
            else:
                logger.error(f"Error in uncertainty extraction (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
    
    return None

def extract_phrases_from_text(text: str) -> List[str]:
    """Extract uncertainty phrases from text using pattern matching."""
    phrases = []
    text_lower = text.lower()
    
    for term in COMMON_UNCERTAINTY_TERMS:
        if term in text_lower:
            phrases.append(term)
    
    return list(set(phrases))

def extract_uncertainty_phrases(text: str) -> List[str]:
    """Extract uncertainty phrases from text using pattern matching (alias for extract_phrases_from_text)."""
    return extract_phrases_from_text(text)

def extract_uncertainty_expressions(
    data: List[Dict[str, Any]], 
    config: UncertaintyConfig,
    output_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract uncertainty expressions from a list of medical reports.
    
    Args:
        data: List of report dictionaries
        config: Configuration for uncertainty extraction
        output_file: Optional output file path
        
    Returns:
        List of reports with uncertainty annotations
    """
    # Setup checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(config.rpm_limit, config.tpm_limit)
    
    # Process data
    results = []
    
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("OpenAI library not installed. Please install with: pip install openai")
        return data
    
    # Initialize clients for each API key
    clients = {}
    for api_key in config.api_keys:
        clients[api_key] = OpenAI(api_key=api_key, base_url=config.api_base)
    
    for i, item in enumerate(tqdm(data, desc="Extracting uncertainty")):
        # Select API key in round-robin fashion
        api_key = config.api_keys[i % len(config.api_keys)]
        client = clients[api_key]
        
        # Extract uncertainty from findings and impression
        item_copy = item.copy()
        
        if 'findings' in item and item['findings']:
            findings_uncertainty = extract_uncertainty_with_llm(
                item['findings'], client, api_key, rate_limiter, config.max_retries
            )
            if findings_uncertainty:
                item_copy['findings_uncertainty'] = findings_uncertainty
        
        if 'impression' in item and item['impression']:
            impression_uncertainty = extract_uncertainty_with_llm(
                item['impression'], client, api_key, rate_limiter, config.max_retries
            )
            if impression_uncertainty:
                item_copy['impression_uncertainty'] = impression_uncertainty
        
        results.append(item_copy)
        
        # Save checkpoint
        if (i + 1) % config.checkpoint_interval == 0:
            checkpoint_file = os.path.join(config.checkpoint_dir, f"uncertainty_checkpoint_{i+1}.pkl")
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Saved checkpoint at {i+1} items")
    
    # Save final results
    if output_file:
        with open(output_file, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved results to {output_file}")
    
    return results

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract uncertainty expressions from medical reports")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--config", help="Configuration file (JSON)")
    parser.add_argument("--api-keys", nargs="+", help="API keys for LLM service")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = UncertaintyConfig(**config_dict)
    else:
        # Default configuration
        api_keys = args.api_keys or [
            "sk-zvokgbacsccnuazgnwarnrqmoiujwbeieohksakdcyzuofau",
            "sk-bhrfmeundatpnfltqspwcuabewrhrftkpjdhzgewndkofbnb", 
            "sk-bgqhnldnnnfaozfblafklkiuzapsfoferrlnezeununqustp"
        ]
        config = UncertaintyConfig(api_keys=api_keys)
    
    # Load data
    data = []
    with open(args.input, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line: {line[:100]}...")
    
    logger.info(f"Loaded {len(data)} items from {args.input}")
    
    # Extract uncertainty
    results = extract_uncertainty_expressions(data, config, args.output)
    
    logger.info(f"Processed {len(results)} items, saved to {args.output}")

class UncertaintyExtractor:
    """Main class for extracting uncertainty expressions from medical reports."""
    
    def __init__(self, config: UncertaintyConfig):
        """Initialize the uncertainty extractor with configuration."""
        self.config = config
        self.rate_limiter = RateLimiter(config.rpm_limit, config.tpm_limit)
        
        # Initialize clients
        self.clients = {}
        try:
            from openai import OpenAI
            for api_key in config.api_keys:
                self.clients[api_key] = OpenAI(api_key=api_key, base_url=config.api_base)
        except ImportError:
            logger.error("OpenAI library not installed. Please install with: pip install openai")
            self.clients = {}
    
    def extract(self, data: List[Dict[str, Any]], output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract uncertainty expressions from data."""
        return extract_uncertainty_expressions(data, self.config, output_file)
    
    def extract_from_text(self, text: str) -> List[str]:
        """Extract uncertainty expressions from a single text."""
        return extract_uncertainty_phrases(text)

def validate_uncertainty_output(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate uncertainty extraction output.
    
    Args:
        data: List of processed reports with uncertainty annotations
        
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'total_reports': len(data),
        'reports_with_findings_uncertainty': 0,
        'reports_with_impression_uncertainty': 0,
        'total_uncertainty_expressions': 0,
        'common_expressions': Counter(),
        'validation_errors': []
    }
    
    for i, item in enumerate(data):
        # Check for findings uncertainty
        if 'findings_uncertainty' in item:
            stats['reports_with_findings_uncertainty'] += 1
            if isinstance(item['findings_uncertainty'], list):
                stats['total_uncertainty_expressions'] += len(item['findings_uncertainty'])
                stats['common_expressions'].update(item['findings_uncertainty'])
            else:
                stats['validation_errors'].append(f"Item {i}: findings_uncertainty is not a list")
        
        # Check for impression uncertainty
        if 'impression_uncertainty' in item:
            stats['reports_with_impression_uncertainty'] += 1
            if isinstance(item['impression_uncertainty'], list):
                stats['total_uncertainty_expressions'] += len(item['impression_uncertainty'])
                stats['common_expressions'].update(item['impression_uncertainty'])
            else:
                stats['validation_errors'].append(f"Item {i}: impression_uncertainty is not a list")
    
    # Calculate percentages
    if stats['total_reports'] > 0:
        stats['findings_uncertainty_percentage'] = (stats['reports_with_findings_uncertainty'] / stats['total_reports']) * 100
        stats['impression_uncertainty_percentage'] = (stats['reports_with_impression_uncertainty'] / stats['total_reports']) * 100
    
    return stats

def analyze_uncertainty_patterns(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze patterns in uncertainty expressions.
    
    Args:
        data: List of processed reports with uncertainty annotations
        
    Returns:
        Dictionary with pattern analysis results
    """
    analysis = {
        'expression_frequency': Counter(),
        'findings_vs_impression': {
            'findings_only': 0,
            'impression_only': 0,
            'both': 0,
            'neither': 0
        },
        'expression_categories': {
            'likelihood': [],
            'possibility': [],
            'exclusion': [],
            'comparison': []
        },
        'co_occurrence': Counter()
    }
    
    # Categorize expressions
    likelihood_terms = ['likely', 'probable', 'presumably', 'appears', 'apparent']
    possibility_terms = ['possibly', 'may', 'might', 'could', 'potential', 'suspected']
    exclusion_terms = ['cannot exclude', 'cannot rule out', 'not excluded', 'not ruled out']
    comparison_terms = ['versus', 'vs', 'consistent with', 'compatible with']
    
    for item in data:
        has_findings = 'findings_uncertainty' in item and item['findings_uncertainty']
        has_impression = 'impression_uncertainty' in item and item['impression_uncertainty']
        
        # Count distribution
        if has_findings and has_impression:
            analysis['findings_vs_impression']['both'] += 1
        elif has_findings:
            analysis['findings_vs_impression']['findings_only'] += 1
        elif has_impression:
            analysis['findings_vs_impression']['impression_only'] += 1
        else:
            analysis['findings_vs_impression']['neither'] += 1
        
        # Collect all expressions
        all_expressions = []
        if has_findings:
            all_expressions.extend(item['findings_uncertainty'])
        if has_impression:
            all_expressions.extend(item['impression_uncertainty'])
        
        # Update frequency counter
        analysis['expression_frequency'].update(all_expressions)
        
        # Categorize expressions
        for expr in all_expressions:
            expr_lower = expr.lower()
            if any(term in expr_lower for term in likelihood_terms):
                analysis['expression_categories']['likelihood'].append(expr)
            elif any(term in expr_lower for term in possibility_terms):
                analysis['expression_categories']['possibility'].append(expr)
            elif any(term in expr_lower for term in exclusion_terms):
                analysis['expression_categories']['exclusion'].append(expr)
            elif any(term in expr_lower for term in comparison_terms):
                analysis['expression_categories']['comparison'].append(expr)
        
        # Analyze co-occurrence (expressions appearing together)
        if len(all_expressions) > 1:
            for i in range(len(all_expressions)):
                for j in range(i + 1, len(all_expressions)):
                    pair = tuple(sorted([all_expressions[i], all_expressions[j]]))
                    analysis['co_occurrence'][pair] += 1
    
    # Convert category lists to counters for better analysis
    for category in analysis['expression_categories']:
        analysis['expression_categories'][category] = Counter(analysis['expression_categories'][category])
    
    return analysis

if __name__ == "__main__":
    main()