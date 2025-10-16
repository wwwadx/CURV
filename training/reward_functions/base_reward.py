"""
Base Reward Function Class

Provides a common interface and utilities for all medical domain reward functions.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)


class BaseRewardFunction(ABC):
    """
    Abstract base class for all reward functions used in medical RLHF training.
    
    This class provides common functionality and enforces a consistent interface
    for all reward function implementations.
    """
    
    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize the base reward function.
        
        Args:
            name: Name of the reward function
            weight: Weight for combining with other reward functions
        """
        self.name = name
        self.weight = weight
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        Calculate reward scores for a batch of completions.
        
        Args:
            completions: List of generated text completions
            **kwargs: Additional arguments specific to the reward function
            
        Returns:
            List of reward scores (one per completion)
        """
        pass
    
    def extract_report_sections(self, text: str) -> Dict[str, str]:
        """
        Extract structured sections from a medical report.
        
        Args:
            text: Raw report text
            
        Returns:
            Dictionary with extracted sections (findings, thinking, impression)
        """
        sections = {
            'findings': '',
            'thinking': '', 
            'impression': ''
        }
        
        # Extract findings section
        findings_match = re.search(r'<findings>(.*?)</findings>', text, re.DOTALL | re.IGNORECASE)
        if findings_match:
            sections['findings'] = findings_match.group(1).strip()
        
        # Extract thinking section
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL | re.IGNORECASE)
        if thinking_match:
            sections['thinking'] = thinking_match.group(1).strip()
        
        # Extract impression section
        impression_match = re.search(r'<impression>(.*?)</impression>', text, re.DOTALL | re.IGNORECASE)
        if impression_match:
            sections['impression'] = impression_match.group(1).strip()
        
        return sections
    
    def validate_completion(self, completion: str) -> bool:
        """
        Basic validation of completion format.
        
        Args:
            completion: Generated text completion
            
        Returns:
            True if completion is valid, False otherwise
        """
        if not completion or not completion.strip():
            return False
            
        # Check for basic structure
        sections = self.extract_report_sections(completion)
        return any(section.strip() for section in sections.values())
    
    def normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Normalize a score to the specified range.
        
        Args:
            score: Raw score
            min_val: Minimum value for normalization
            max_val: Maximum value for normalization
            
        Returns:
            Normalized score
        """
        return max(min_val, min(max_val, score))
    
    def log_scores(self, completions: List[str], scores: List[float], sample_size: int = 3):
        """
        Log sample completions and their scores for debugging.
        
        Args:
            completions: List of completions
            scores: List of corresponding scores
            sample_size: Number of samples to log
        """
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
            
        sample_indices = list(range(min(sample_size, len(completions))))
        
        for i in sample_indices:
            self.logger.debug(
                f"{self.name} - Sample {i+1}:\n"
                f"Completion: {completions[i][:200]}...\n"
                f"Score: {scores[i]:.4f}"
            )