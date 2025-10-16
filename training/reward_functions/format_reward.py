"""
Format Reward Function for Medical Reports

Validates that generated medical reports follow the required structured format
with proper section tags (findings, thinking, impression).
"""

import re
import os
from datetime import datetime
from typing import List
from .base_reward import BaseRewardFunction


class FormatRewardCXR(BaseRewardFunction):
    """
    Reward function that validates medical report format compliance.
    
    Checks for:
    - Presence of all three required sections: findings, thinking, impression
    - Correct order of sections
    - Exactly one occurrence of each section
    - No content outside the structured tags
    """
    
    def __init__(self, weight: float = 1.0, debug_mode: bool = False, log_path: str = None):
        """
        Initialize the format reward function.
        
        Args:
            weight: Weight for combining with other reward functions
            debug_mode: Enable debug logging
            log_path: Path for debug log files
        """
        super().__init__("FormatRewardCXR", weight)
        self.debug_mode = debug_mode or os.getenv("DEBUG_MODE") == "true"
        self.log_path = log_path or os.getenv("LOG_PATH")
        
        # Pattern to check for all three tags present exactly once with no content outside
        # The pattern ensures:
        # 1. The content starts with <findings> tag
        # 2. All three tags are present in order: findings, thinking, impression
        # 3. The content ends with </impression> tag
        # 4. No content appears outside these tags
        self.format_pattern = r"^\s*<findings>.*?</findings>\s*<thinking>.*?</thinking>\s*<impression>.*?</impression>\s*$"
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        Check if the CXR report output matches the required format with three specific tags.

        Args:
            completions: List of generated text completions

        Returns:
            List of reward scores (1.0 for correct format, 0.0 for incorrect)
        """
        
        # Check pattern match for overall structure
        pattern_matches = [
            re.search(self.format_pattern, content, re.DOTALL) is not None 
            for content in completions
        ]
        
        # Count occurrences of each tag to ensure exactly one of each
        findings_counts = [content.count("<findings>") for content in completions]
        thinking_counts = [content.count("<thinking>") for content in completions]
        impression_counts = [content.count("<impression>") for content in completions]
        
        # Check if each tag appears exactly once
        exact_counts = [
            f == 1 and t == 1 and i == 1 
            for f, t, i in zip(findings_counts, thinking_counts, impression_counts)
        ]
        
        # Final check: both pattern matches and exact count of tags
        final_scores = [
            1.0 if (pattern_match and exact_count) else 0.0
            for pattern_match, exact_count in zip(pattern_matches, exact_counts)
        ]
        
        # Debug logging
        if self.debug_mode and self.log_path:
            self._log_debug_info(completions, final_scores, findings_counts, thinking_counts, impression_counts)
        
        # Log sample scores for monitoring
        self.log_scores(completions, final_scores)
        
        return final_scores
    
    def _log_debug_info(self, completions: List[str], scores: List[float], 
                       findings_counts: List[int], thinking_counts: List[int], 
                       impression_counts: List[int]):
        """
        Log detailed debug information about format validation.
        
        Args:
            completions: List of completions
            scores: List of scores
            findings_counts: Count of findings tags per completion
            thinking_counts: Count of thinking tags per completion
            impression_counts: Count of impression tags per completion
        """
        try:
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            debug_log_path = self.log_path.replace(".txt", "_format_cxr.txt")
            
            with open(debug_log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} CXR Format reward -------------\n")
                
                for i, (content, score, f_count, t_count, i_count) in enumerate(
                    zip(completions, scores, findings_counts, thinking_counts, impression_counts)
                ):
                    f.write(f"Completion {i+1}:\n")
                    f.write(f"Content: {content[:500]}...\n")  # Truncate for readability
                    f.write(f"Findings count: {f_count}, Thinking count: {t_count}, Impression count: {i_count}\n")
                    f.write(f"Has correct format: {score == 1.0}\n")
                    f.write(f"Score: {score}\n\n")
                    
        except Exception as e:
            self.logger.warning(f"Failed to write debug log: {e}")
    
    def validate_sections(self, completion: str) -> dict:
        """
        Validate individual sections of a completion.
        
        Args:
            completion: Generated text completion
            
        Returns:
            Dictionary with validation results for each section
        """
        sections = self.extract_report_sections(completion)
        
        return {
            'has_findings': bool(sections['findings'].strip()),
            'has_thinking': bool(sections['thinking'].strip()),
            'has_impression': bool(sections['impression'].strip()),
            'findings_count': completion.count("<findings>"),
            'thinking_count': completion.count("<thinking>"),
            'impression_count': completion.count("<impression>"),
            'sections': sections
        }