"""
Coherence Reward Function for Medical Report Generation

This module implements the Thinking Coherence Reward from Stage 3 which measures
coherence between thinking section and findings/impression sections using Jaccard similarity.

The reward is calculated as:
r^{coherence}_i = Jaccard(T_{thinking}, T_{findings} ∪ T_{impression})
"""

import os
import re
from datetime import datetime
from typing import List, Set, Optional

from .base_reward import BaseRewardFunction


class CoherenceRewardCXR(BaseRewardFunction):
    """
    Coherence reward function for chest X-ray report generation.
    
    Measures coherence between thinking section and findings/impression sections 
    using Jaccard similarity of medical entities extracted via RadGraph.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the coherence reward function.
        
        Args:
            debug: Enable debug logging
        """
        super().__init__(debug)
        self.name = "coherence_cxr"
        self.radgraph_client = None
        self._setup_radgraph()
    
    def _setup_radgraph(self):
        """Setup RadGraph API client for entity extraction."""
        try:
            from radgraph_api import RadGraphAPIClient
            self.radgraph_client = RadGraphAPIClient()
            if self.debug:
                print("RadGraph API client initialized successfully")
        except ImportError:
            if self.debug:
                print("RadGraph API client not available. Using fallback entity extraction.")
            self.radgraph_client = None
    
    def _extract_medical_entities(self, text: str) -> Set[str]:
        """
        Extract medical entities using RadGraph API.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            Set of medical entities (lowercase)
        """
        # Use RadGraph API client if available
        if self.radgraph_client:
            try:
                # Get entities from API
                batch_results = self.radgraph_client.extract_entities([text])
                
                # Flatten the list of lists and convert to set
                entities = set()
                for entity_list in batch_results:
                    entities.update([entity.lower() for entity in entity_list])
                
                return entities
            except Exception as e:
                if self.debug:
                    print(f"Error using RadGraph API: {e}")
                # Fallback to simpler tokenization
                return self._fallback_extract_entities(text)
        else:
            # Use fallback method
            return self._fallback_extract_entities(text)
    
    def _fallback_extract_entities(self, text: str) -> Set[str]:
        """
        Fallback entity extraction using simple tokenization.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            Set of potential medical entities (lowercase)
        """
        try:
            from radgraph_api import fallback_extract_entities
            return fallback_extract_entities(text)
        except ImportError:
            # Simple fallback: extract medical-looking terms
            import string
            
            # Remove punctuation and split into words
            text_clean = text.translate(str.maketrans('', '', string.punctuation))
            words = text_clean.lower().split()
            
            # Filter for potential medical terms (length > 3, contains medical patterns)
            medical_patterns = ['tion', 'osis', 'itis', 'emia', 'pathy', 'gram', 'scopy']
            entities = set()
            
            for word in words:
                if len(word) > 3:
                    # Add words that contain medical suffixes or are longer medical terms
                    if any(pattern in word for pattern in medical_patterns) or len(word) > 6:
                        entities.add(word)
            
            return entities
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """
        Calculate Jaccard similarity between two sets.
        
        Args:
            set1: First set
            set2: Second set
            
        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / union if union > 0 else 0.0
    
    def __call__(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> float:
        """
        Calculate coherence reward for a single response.
        
        Args:
            prompt: Input prompt (not used for coherence calculation)
            response: Generated response containing medical report sections
            ground_truth: Ground truth response (not used for coherence calculation)
            
        Returns:
            Coherence reward score (0.0 to 1.0)
        """
        # Extract sections using regex
        findings_pattern = r'<findings>(.*?)</findings>'
        thinking_pattern = r'<thinking>(.*?)</thinking>'
        impression_pattern = r'<impression>(.*?)</impression>'
        
        # Extract each section
        findings_match = re.search(findings_pattern, response, re.DOTALL)
        thinking_match = re.search(thinking_pattern, response, re.DOTALL)
        impression_match = re.search(impression_pattern, response, re.DOTALL)
        
        # Default reward if sections are missing
        reward = 0.0
        findings_terms = set()
        thinking_terms = set()
        impression_terms = set()
        
        # Calculate coherence only if all sections are present
        if findings_match and thinking_match and impression_match:
            findings_text = findings_match.group(1).strip()
            thinking_text = thinking_match.group(1).strip()
            impression_text = impression_match.group(1).strip()
            
            # Extract terms from each section using RadGraph API
            findings_terms = self._extract_medical_entities(findings_text)
            thinking_terms = self._extract_medical_entities(thinking_text)
            impression_terms = self._extract_medical_entities(impression_text)
            
            # Union of findings and impression terms
            findings_impression_terms = findings_terms.union(impression_terms)
            
            # Calculate Jaccard similarity
            reward = self._jaccard_similarity(thinking_terms, findings_impression_terms)
        
        # Debug logging
        if self.debug:
            self._log_debug_info(response, findings_terms, thinking_terms, impression_terms, reward)
        
        return reward
    
    def _log_debug_info(self, content: str, findings_terms: Set[str], 
                       thinking_terms: Set[str], impression_terms: Set[str], reward: float):
        """Log debug information for coherence calculation."""
        log_path = os.getenv("LOG_PATH", "debug_coherence.txt")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        
        with open(log_path.replace(".txt", "_coherence_cxr.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} CXR Coherence reward -------------\n")
            f.write(f"Content: {content}\n")
            f.write(f"Findings terms: {findings_terms}\n")
            f.write(f"Thinking terms: {thinking_terms}\n")
            f.write(f"Impression terms: {impression_terms}\n")
            f.write(f"Jaccard similarity: {reward}\n\n")
    
    def batch_call(self, prompts: List[str], responses: List[str], 
                   ground_truths: Optional[List[str]] = None) -> List[float]:
        """
        Calculate coherence rewards for multiple responses (batch processing).
        
        Args:
            prompts: List of input prompts
            responses: List of generated responses
            ground_truths: List of ground truth responses (not used)
            
        Returns:
            List of coherence reward scores
        """
        rewards = []
        
        for i, response in enumerate(responses):
            prompt = prompts[i] if i < len(prompts) else ""
            reward = self.__call__(prompt, response)
            rewards.append(reward)
        
        return rewards