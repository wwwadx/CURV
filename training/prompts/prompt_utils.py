"""
Prompt Utilities

Utility functions for prompt formatting, validation, and processing.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class PromptValidationResult:
    """Result of prompt validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class PromptFormatter:
    """
    Utility class for formatting and processing prompts.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def extract_sections(text: str, sections: List[str]) -> Dict[str, str]:
        """
        Extract tagged sections from text.
        
        Args:
            text: Input text containing tagged sections
            sections: List of section names to extract
            
        Returns:
            Dictionary mapping section names to their content
        """
        extracted = {}
        
        for section in sections:
            pattern = f"<{section}>(.*?)</{section}>"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            
            if match:
                content = match.group(1).strip()
                extracted[section] = content
            else:
                extracted[section] = ""
        
        return extracted
    
    @staticmethod
    def format_sections(sections: Dict[str, str]) -> str:
        """
        Format sections into tagged text.
        
        Args:
            sections: Dictionary mapping section names to content
            
        Returns:
            Formatted text with tagged sections
        """
        formatted_parts = []
        
        for section_name, content in sections.items():
            if content.strip():
                formatted_parts.append(f"<{section_name}>\n{content.strip()}\n</{section_name}>")
        
        return "\n\n".join(formatted_parts)
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text
    
    def format_conversation(
        self,
        system_prompt: str,
        user_message: str,
        assistant_message: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format a conversation for training or inference.
        
        Args:
            system_prompt: System prompt
            user_message: User message
            assistant_message: Assistant response (optional)
            image_path: Path to image (optional)
            
        Returns:
            Formatted conversation dictionary
        """
        conversation = {
            "system": self.clean_text(system_prompt),
            "messages": [
                {
                    "role": "user",
                    "content": self.clean_text(user_message)
                }
            ]
        }
        
        # Add image if provided
        if image_path:
            conversation["messages"][0]["image"] = image_path
        
        # Add assistant response if provided
        if assistant_message:
            conversation["messages"].append({
                "role": "assistant",
                "content": self.clean_text(assistant_message)
            })
        
        return conversation
    
    def batch_format_conversations(
        self,
        conversations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format multiple conversations in batch.
        
        Args:
            conversations: List of conversation dictionaries
            
        Returns:
            List of formatted conversations
        """
        formatted_conversations = []
        
        for conv in conversations:
            try:
                formatted_conv = self.format_conversation(
                    system_prompt=conv.get("system", ""),
                    user_message=conv.get("user", ""),
                    assistant_message=conv.get("assistant"),
                    image_path=conv.get("image")
                )
                formatted_conversations.append(formatted_conv)
                
            except Exception as e:
                self.logger.warning(f"Failed to format conversation: {e}")
                continue
        
        return formatted_conversations


def validate_prompt_format(text: str, required_sections: List[str] = None) -> PromptValidationResult:
    """
    Validate prompt format and structure.
    
    Args:
        text: Text to validate
        required_sections: List of required section names
        
    Returns:
        PromptValidationResult with validation details
    """
    errors = []
    warnings = []
    
    if not text or not text.strip():
        errors.append("Text is empty")
        return PromptValidationResult(False, errors, warnings)
    
    # Check for required sections if specified
    if required_sections:
        for section in required_sections:
            pattern = f"<{section}>.*?</{section}>"
            if not re.search(pattern, text, re.DOTALL | re.IGNORECASE):
                errors.append(f"Missing required section: {section}")
    
    # Check for malformed tags
    open_tags = re.findall(r'<(\w+)>', text)
    close_tags = re.findall(r'</(\w+)>', text)
    
    for tag in open_tags:
        if tag not in close_tags:
            errors.append(f"Unclosed tag: <{tag}>")
    
    for tag in close_tags:
        if tag not in open_tags:
            errors.append(f"Unmatched closing tag: </{tag}>")
    
    # Check for nested tags of the same type
    for section in set(open_tags):
        pattern = f"<{section}>.*?<{section}>"
        if re.search(pattern, text, re.DOTALL):
            warnings.append(f"Potentially nested tags detected for section: {section}")
    
    # Check for very long sections
    for section in set(open_tags):
        pattern = f"<{section}>(.*?)</{section}>"
        match = re.search(pattern, text, re.DOTALL)
        if match and len(match.group(1)) > 5000:
            warnings.append(f"Very long content in section: {section}")
    
    is_valid = len(errors) == 0
    return PromptValidationResult(is_valid, errors, warnings)


def validate_cxr_report(text: str) -> PromptValidationResult:
    """
    Validate CXR report format specifically.
    
    Args:
        text: CXR report text to validate
        
    Returns:
        PromptValidationResult with validation details
    """
    required_sections = ["findings", "thinking", "impression"]
    return validate_prompt_format(text, required_sections)


def extract_cxr_sections(text: str) -> Dict[str, str]:
    """
    Extract CXR report sections.
    
    Args:
        text: CXR report text
        
    Returns:
        Dictionary with extracted sections
    """
    sections = ["findings", "thinking", "impression"]
    return PromptFormatter.extract_sections(text, sections)


def format_cxr_report(findings: str, thinking: str, impression: str) -> str:
    """
    Format CXR report from individual sections.
    
    Args:
        findings: Findings section content
        thinking: Thinking section content
        impression: Impression section content
        
    Returns:
        Formatted CXR report
    """
    sections = {
        "findings": findings,
        "thinking": thinking,
        "impression": impression
    }
    return PromptFormatter.format_sections(sections)


def create_training_prompt(
    system_prompt: str,
    user_input: str,
    expected_output: str,
    image_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a training prompt in the standard format.
    
    Args:
        system_prompt: System instruction
        user_input: User input/question
        expected_output: Expected model output
        image_path: Path to associated image (optional)
        
    Returns:
        Training prompt dictionary
    """
    formatter = PromptFormatter()
    return formatter.format_conversation(
        system_prompt=system_prompt,
        user_message=user_input,
        assistant_message=expected_output,
        image_path=image_path
    )


def load_prompts_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load prompts from a file.
    
    Args:
        file_path: Path to the prompts file
        
    Returns:
        List of prompt dictionaries
    """
    import json
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            elif file_path.endswith('.jsonl'):
                prompts = []
                for line in f:
                    if line.strip():
                        prompts.append(json.loads(line))
                return prompts
            else:
                # Plain text file - treat as single prompt
                content = f.read()
                return [{"content": content}]
                
    except Exception as e:
        logging.error(f"Failed to load prompts from {file_path}: {e}")
        return []


def save_prompts_to_file(prompts: List[Dict[str, Any]], file_path: str):
    """
    Save prompts to a file.
    
    Args:
        prompts: List of prompt dictionaries
        file_path: Output file path
    """
    import json
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                json.dump(prompts, f, indent=2, ensure_ascii=False)
            elif file_path.endswith('.jsonl'):
                for prompt in prompts:
                    f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
            else:
                # Plain text - concatenate all content
                for prompt in prompts:
                    f.write(str(prompt.get('content', prompt)) + '\n\n')
                    
    except Exception as e:
        logging.error(f"Failed to save prompts to {file_path}: {e}")