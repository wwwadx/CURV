"""
Chest X-ray (CXR) Prompt Templates

This module contains prompt templates specifically designed for 
chest X-ray report generation tasks.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CXRPromptTemplate:
    """
    Template for chest X-ray report generation prompts.
    """
    system_prompt: str
    user_template: str
    assistant_template: str = ""
    
    def format(self, image_path: Optional[str] = None, **kwargs) -> Dict[str, str]:
        """
        Format the prompt template with provided variables.
        
        Args:
            image_path: Path to the medical image (optional)
            **kwargs: Additional variables for template formatting
            
        Returns:
            Dictionary with formatted prompts
        """
        formatted_user = self.user_template.format(
            image_path=image_path or "[IMAGE]",
            **kwargs
        )
        
        return {
            "system": self.system_prompt,
            "user": formatted_user,
            "assistant": self.assistant_template
        }


# Default CXR prompt template (preserved from original)
DEFAULT_CXR_SYSTEM_PROMPT = """You are a medical expert tasked with generating a detailed radiology report based on the provided medical image. Analyze the image carefully and produce a structured report using the following tagged sections: <findings>, <thinking>, and <impression>. Follow these guidelines for each section:

- <findings>: Describe only observable features in the image, such as abnormalities, anatomical structures, or notable patterns. Be precise and avoid speculation.

- <thinking>: Provide a logical reasoning process based on the findings, considering possible diagnoses or clinical implications.

- <impression>: Summarize the key takeaways and suggest next steps or potential diagnoses in a concise manner.


Output Format:

<findings>  Detailed description of image observations </findings>

<thinking> Reasoning based on findings </thinking>

<impression> Concise summary and recommendations </impression>"""

DEFAULT_CXR_USER_TEMPLATE = """Please analyze this chest X-ray image and generate a detailed radiology report following the specified format.

Image: {image_path}"""

DEFAULT_CXR_ASSISTANT_TEMPLATE = """<findings>
{findings}
</findings>

<thinking>
{thinking}
</thinking>

<impression>
{impression}
</impression>"""


def load_cxr_prompt(prompt_file: Optional[str] = None) -> CXRPromptTemplate:
    """
    Load CXR prompt template from file or return default template.
    
    Args:
        prompt_file: Path to custom prompt file (optional)
        
    Returns:
        CXRPromptTemplate instance
    """
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            custom_prompt = f.read().strip()
        
        return CXRPromptTemplate(
            system_prompt=custom_prompt,
            user_template=DEFAULT_CXR_USER_TEMPLATE,
            assistant_template=DEFAULT_CXR_ASSISTANT_TEMPLATE
        )
    
    return CXRPromptTemplate(
        system_prompt=DEFAULT_CXR_SYSTEM_PROMPT,
        user_template=DEFAULT_CXR_USER_TEMPLATE,
        assistant_template=DEFAULT_CXR_ASSISTANT_TEMPLATE
    )


def create_cxr_conversation(
    image_path: str,
    findings: str = "",
    thinking: str = "",
    impression: str = "",
    prompt_template: Optional[CXRPromptTemplate] = None
) -> Dict[str, Any]:
    """
    Create a conversation format for CXR report generation.
    
    Args:
        image_path: Path to the chest X-ray image
        findings: Findings section content (for training data)
        thinking: Thinking section content (for training data)
        impression: Impression section content (for training data)
        prompt_template: Custom prompt template (optional)
        
    Returns:
        Conversation dictionary
    """
    if prompt_template is None:
        prompt_template = load_cxr_prompt()
    
    # Format the prompts
    formatted_prompts = prompt_template.format(image_path=image_path)
    
    conversation = {
        "system": formatted_prompts["system"],
        "messages": [
            {
                "role": "user",
                "content": formatted_prompts["user"]
            }
        ]
    }
    
    # Add assistant response if training data is provided
    if findings or thinking or impression:
        assistant_response = prompt_template.assistant_template.format(
            findings=findings,
            thinking=thinking,
            impression=impression
        )
        conversation["messages"].append({
            "role": "assistant",
            "content": assistant_response
        })
    
    return conversation


# Predefined prompt variations for different use cases
PROMPT_VARIATIONS = {
    "detailed": CXRPromptTemplate(
        system_prompt=DEFAULT_CXR_SYSTEM_PROMPT + "\n\nProvide highly detailed observations and comprehensive reasoning.",
        user_template=DEFAULT_CXR_USER_TEMPLATE,
        assistant_template=DEFAULT_CXR_ASSISTANT_TEMPLATE
    ),
    
    "concise": CXRPromptTemplate(
        system_prompt=DEFAULT_CXR_SYSTEM_PROMPT.replace("detailed", "concise"),
        user_template=DEFAULT_CXR_USER_TEMPLATE,
        assistant_template=DEFAULT_CXR_ASSISTANT_TEMPLATE
    ),
    
    "educational": CXRPromptTemplate(
        system_prompt=DEFAULT_CXR_SYSTEM_PROMPT + "\n\nInclude educational explanations suitable for medical students.",
        user_template=DEFAULT_CXR_USER_TEMPLATE,
        assistant_template=DEFAULT_CXR_ASSISTANT_TEMPLATE
    )
}


def get_prompt_variation(variation: str = "default") -> CXRPromptTemplate:
    """
    Get a specific prompt variation.
    
    Args:
        variation: Name of the prompt variation
        
    Returns:
        CXRPromptTemplate instance
    """
    if variation == "default":
        return load_cxr_prompt()
    
    return PROMPT_VARIATIONS.get(variation, load_cxr_prompt())