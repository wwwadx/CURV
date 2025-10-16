# CURV Training Module

A comprehensive training framework for medical vision-language models, adapted from ms-swift with enhanced organization and modularity.

## Overview

The CURV training module provides a structured approach to training medical vision-language models using GRPO (Generalized Reward-based Policy Optimization) with multiple reward functions specifically designed for chest X-ray report generation.

## Directory Structure

```
CURV/
├── training/
│   ├── __init__.py                 # Main training package
│   ├── config/                     # Configuration management
│   │   ├── __init__.py
│   │   └── grpo_config.py         # GRPO training configuration
│   ├── reward_functions/           # Reward function implementations
│   │   ├── __init__.py
│   │   ├── base_reward.py         # Base reward function class
│   │   ├── format_reward.py       # Format validation reward
│   │   ├── accuracy_reward.py     # Accuracy-based reward
│   ├── trainers/                   # Training implementations
│   │   ├── __init__.py
│   │   └── grpo_trainer.py        # GRPO trainer implementation
│   ├── prompts/                    # Prompt templates and utilities
│   │   ├── __init__.py
│   │   ├── cxr_prompts.py         # CXR-specific prompts
│   │   ├── prompt_utils.py        # Prompt utilities
│   │   └── prompt_cxr.txt         # Original CXR prompt template
│   ├── utils/                      # Utility functions
│   │   ├── __init__.py
│   │   ├── data_utils.py          # Dataset handling utilities
│   │   ├── model_utils.py         # Model management utilities
│   │   └── training_utils.py      # Training utilities
│   └── scripts/                    # Training scripts
│       ├── __init__.py
│       ├── train_grpo.py          # Main Python training script
│       └── run_grpo_training.sh   # Shell script wrapper
└── README.md                       # This file
```

## Features

### Reward Functions

The module includes three main reward functions for evaluating model outputs:

- **Format Reward**: Validates report structure and required sections
- **Accuracy Reward**: Evaluates medical accuracy using pathology detection  
- **Coherence Reward**: Measures coherence between thinking, findings, and impression sections using medical entity extraction

**Note**: The coherence reward uses F1ChexBERT and RadGraph as dependencies for medical entity extraction. See the [dependency setup guide](#dependency-setup) below.

### Configuration System
- Comprehensive configuration management with `GRPOConfig`
- Support for dataset, model, training, and reward function configurations
- Easy parameter tuning and experiment management

### Training Framework
- GRPO trainer with VLLM integration for efficient inference
- Support for LoRA fine-tuning and quantization
- Comprehensive logging and monitoring
- Checkpoint management and resumption

### Prompt Management
- Structured prompt templates for CXR report generation
- Multiple prompt variations (detailed, concise, educational)
- Validation and formatting utilities

## Quick Start

### 1. Installation

```bash
# Install required dependencies
pip install torch transformers datasets
pip install vllm trl
pip install math_verify  # for reward functions
```

### 2. Configuration

Create or modify the configuration file:

```python
from training.config import create_default_config

# Create default configuration
config = create_default_config()

# Customize as needed
config.model.model_id_or_path = "your-model-path"
config.dataset.dataset_path = "your-dataset-path"
config.training.output_dir = "your-output-dir"

# Save configuration
config.save("config/my_grpo_config.py")
```

### 3. Training

#### Using Python Script

```bash
cd CURV/training/scripts
python train_grpo.py --config_file ../config/grpo_config.py
```

#### Using Shell Script

```bash
cd CURV/training/scripts
bash run_grpo_training.sh ../config/grpo_config.py
```

## Dependency Setup

The coherence reward function requires F1ChexBERT and RadGraph dependencies for optimal performance.

### Quick Setup

Run the automated setup script:

```bash
cd /save/CURV/training
./setup_dependencies.sh
source setup_env.sh
```

### Manual Setup

For detailed setup instructions, see [`setup_dependencies.md`](training/setup_dependencies.md).


## Configuration Options

### Model Configuration
- `model_id_or_path`: Path to the base model
- `torch_dtype`: Model precision (bfloat16, float16, float32)
- `model_kwargs`: Additional model arguments

### Training Configuration
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per device
- `learning_rate`: Learning rate
- `gradient_accumulation_steps`: Gradient accumulation steps
- `warmup_ratio`: Warmup ratio
- `max_grad_norm`: Maximum gradient norm

### GRPO Configuration
- `num_generations`: Number of generations per prompt
- `temperature`: Sampling temperature
- `top_p`: Top-p sampling parameter
- `top_k`: Top-k sampling parameter
- `max_completion_length`: Maximum completion length

### Reward Functions
- `format_cxr`: Format validation reward weight
- `accuracy_cxr`: Accuracy reward weight
- `coherence_cxr`: Coherence reward weight

### VLLM Configuration
- `use_vllm`: Enable VLLM for inference acceleration
- `vllm_server_host`: VLLM server host
- `vllm_server_port`: VLLM server port
- `vllm_device`: VLLM device configuration

## Advanced Usage

### Custom Reward Functions

Create custom reward functions by extending `BaseRewardFunction`:

```python
from training.reward_functions import BaseRewardFunction

class CustomReward(BaseRewardFunction):
    def __call__(self, prompt, response, ground_truth=None):
        # Implement your reward logic
        score = calculate_custom_score(response, ground_truth)
        return score
```

### Custom Prompts

Create custom prompt templates:

```python
from training.prompts import CXRPromptTemplate

custom_template = CXRPromptTemplate(
    system_template="Your custom system prompt",
    user_template="Your custom user prompt: {input}",
    assistant_template="Your custom assistant template"
)
```

### Monitoring and Logging

The training framework provides comprehensive logging:
- TensorBoard integration for metrics visualization
- Detailed training logs with reward function scores
- Model checkpointing and resumption
- GPU memory monitoring

## Original Code Preservation

This implementation preserves the original ms-swift training logic while providing better organization:

- Original shell script commands are preserved in `run_grpo_training.sh`
- Reward function implementations maintain the same logic as the original plugin system
- Training parameters and configurations match the original setup
- Prompt templates preserve the original CXR prompt structure

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- VLLM (for inference acceleration)
- TRL (for GRPO training)
- Additional medical NLP libraries (ChexBERT, RadGraph)


## License

This project is adapted from ms-swift and follows the same licensing terms.
