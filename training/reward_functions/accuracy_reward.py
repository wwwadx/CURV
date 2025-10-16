"""
Accuracy Reward Function for Medical Reports

Evaluates the accuracy of generated medical reports by comparing
against ground truth labels and clinical findings.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from .base_reward import BaseRewardFunction


class AccuracyRewardCXR(BaseRewardFunction):
    """
    Reward function that evaluates medical report accuracy.
    
    Compares generated reports against ground truth clinical findings
    and pathology labels to compute accuracy scores.
    """
    
    def __init__(self, weight: float = 1.0, accuracy_threshold: float = 0.8, 
                 debug_mode: bool = False, log_path: str = None):
        """
        Initialize the accuracy reward function.
        
        Args:
            weight: Weight for combining with other reward functions
            accuracy_threshold: Minimum accuracy threshold for positive reward
            debug_mode: Enable debug logging
            log_path: Path for debug log files
        """
        super().__init__("AccuracyRewardCXR", weight)
        self.accuracy_threshold = accuracy_threshold
        self.debug_mode = debug_mode or os.getenv("DEBUG_MODE") == "true"
        self.log_path = log_path or os.getenv("LOG_PATH")
        
        # Common pathology terms for chest X-ray reports
        self.pathology_terms = {
            'pneumonia', 'pneumothorax', 'pleural_effusion', 'atelectasis',
            'cardiomegaly', 'consolidation', 'edema', 'emphysema', 'fibrosis',
            'fracture', 'infiltrate', 'mass', 'nodule', 'opacity'
        }
        
        # Severity indicators
        self.severity_terms = {
            'mild', 'moderate', 'severe', 'minimal', 'extensive', 'bilateral',
            'unilateral', 'left', 'right', 'upper', 'lower', 'middle'
        }
    
    def __call__(self, completions: List[str], ground_truth: List[Dict[str, Any]] = None, 
                 **kwargs) -> List[float]:
        """
        Evaluate accuracy of medical report completions.

        Args:
            completions: List of generated text completions
            ground_truth: List of ground truth data with clinical findings
            **kwargs: Additional context (images, metadata, etc.)

        Returns:
            List of accuracy scores between 0.0 and 1.0
        """
        if not ground_truth:
            # If no ground truth provided, return neutral scores
            return [0.5] * len(completions)
        
        scores = []
        
        for completion, gt_data in zip(completions, ground_truth):
            try:
                # Extract sections from completion
                sections = self.extract_report_sections(completion)
                
                # Calculate accuracy based on different components
                pathology_score = self._evaluate_pathology_accuracy(sections, gt_data)
                severity_score = self._evaluate_severity_accuracy(sections, gt_data)
                location_score = self._evaluate_location_accuracy(sections, gt_data)
                
                # Combine scores with weights
                overall_score = (
                    0.5 * pathology_score +
                    0.3 * severity_score +
                    0.2 * location_score
                )
                
                # Apply threshold
                final_score = overall_score if overall_score >= self.accuracy_threshold else 0.0
                scores.append(final_score)
                
            except Exception as e:
                self.logger.warning(f"Error evaluating accuracy: {e}")
                scores.append(0.0)
        
        # Debug logging
        if self.debug_mode and self.log_path:
            self._log_debug_info(completions, scores, ground_truth)
        
        # Log sample scores for monitoring
        self.log_scores(completions, scores)
        
        return scores
    
    def _evaluate_pathology_accuracy(self, sections: Dict[str, str], 
                                   ground_truth: Dict[str, Any]) -> float:
        """
        Evaluate accuracy of pathology identification.
        
        Args:
            sections: Extracted report sections
            ground_truth: Ground truth clinical data
            
        Returns:
            Pathology accuracy score
        """
        if 'pathologies' not in ground_truth:
            return 0.5  # Neutral score if no pathology data
        
        gt_pathologies = set(ground_truth['pathologies'])
        
        # Extract pathologies mentioned in findings and impression
        findings_text = sections.get('findings', '').lower()
        impression_text = sections.get('impression', '').lower()
        combined_text = f"{findings_text} {impression_text}"
        
        # Find mentioned pathologies
        mentioned_pathologies = set()
        for pathology in self.pathology_terms:
            if pathology.replace('_', ' ') in combined_text or pathology in combined_text:
                mentioned_pathologies.add(pathology)
        
        # Calculate precision and recall
        if not gt_pathologies and not mentioned_pathologies:
            return 1.0  # Both empty - perfect match
        
        if not gt_pathologies:
            return 0.0 if mentioned_pathologies else 1.0
        
        if not mentioned_pathologies:
            return 0.0
        
        # Calculate F1 score
        intersection = gt_pathologies.intersection(mentioned_pathologies)
        precision = len(intersection) / len(mentioned_pathologies)
        recall = len(intersection) / len(gt_pathologies)
        
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    
    def _evaluate_severity_accuracy(self, sections: Dict[str, str], 
                                  ground_truth: Dict[str, Any]) -> float:
        """
        Evaluate accuracy of severity assessment.
        
        Args:
            sections: Extracted report sections
            ground_truth: Ground truth clinical data
            
        Returns:
            Severity accuracy score
        """
        if 'severity' not in ground_truth:
            return 0.5  # Neutral score if no severity data
        
        gt_severity = ground_truth['severity'].lower()
        
        # Extract severity mentions from findings and impression
        findings_text = sections.get('findings', '').lower()
        impression_text = sections.get('impression', '').lower()
        combined_text = f"{findings_text} {impression_text}"
        
        # Check for severity term matches
        mentioned_severities = []
        for severity in self.severity_terms:
            if severity in combined_text:
                mentioned_severities.append(severity)
        
        # Simple matching - can be enhanced with more sophisticated NLP
        if gt_severity in mentioned_severities:
            return 1.0
        
        # Partial credit for related severity terms
        severity_mapping = {
            'mild': ['minimal', 'slight', 'small'],
            'moderate': ['medium', 'intermediate'],
            'severe': ['extensive', 'large', 'significant']
        }
        
        for mentioned in mentioned_severities:
            if gt_severity in severity_mapping.get(mentioned, []):
                return 0.7
        
        return 0.0
    
    def _evaluate_location_accuracy(self, sections: Dict[str, str], 
                                  ground_truth: Dict[str, Any]) -> float:
        """
        Evaluate accuracy of anatomical location identification.
        
        Args:
            sections: Extracted report sections
            ground_truth: Ground truth clinical data
            
        Returns:
            Location accuracy score
        """
        if 'locations' not in ground_truth:
            return 0.5  # Neutral score if no location data
        
        gt_locations = set(loc.lower() for loc in ground_truth['locations'])
        
        # Extract location mentions from findings and impression
        findings_text = sections.get('findings', '').lower()
        impression_text = sections.get('impression', '').lower()
        combined_text = f"{findings_text} {impression_text}"
        
        # Find mentioned locations
        location_terms = {
            'left', 'right', 'bilateral', 'unilateral',
            'upper', 'lower', 'middle', 'base', 'apex',
            'lung', 'lobe', 'field', 'zone'
        }
        
        mentioned_locations = set()
        for location in location_terms:
            if location in combined_text:
                mentioned_locations.add(location)
        
        # Calculate overlap
        if not gt_locations and not mentioned_locations:
            return 1.0
        
        if not gt_locations or not mentioned_locations:
            return 0.0
        
        intersection = gt_locations.intersection(mentioned_locations)
        union = gt_locations.union(mentioned_locations)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _log_debug_info(self, completions: List[str], scores: List[float], 
                       ground_truth: List[Dict[str, Any]]):
        """
        Log detailed debug information about accuracy evaluation.
        
        Args:
            completions: List of completions
            scores: List of accuracy scores
            ground_truth: List of ground truth data
        """
        try:
            from datetime import datetime
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            debug_log_path = self.log_path.replace(".txt", "_accuracy_cxr.txt")
            
            with open(debug_log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} CXR Accuracy reward -------------\n")
                
                for i, (completion, score, gt_data) in enumerate(
                    zip(completions, scores, ground_truth)
                ):
                    f.write(f"Completion {i+1}:\n")
                    f.write(f"Content: {completion[:300]}...\n")
                    f.write(f"Ground Truth: {json.dumps(gt_data, indent=2)}\n")
                    f.write(f"Accuracy Score: {score}\n\n")
                    
        except Exception as e:
            self.logger.warning(f"Failed to write debug log: {e}")