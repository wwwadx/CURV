"""
Example Configuration for CURV GRPO Training

This file demonstrates how to create and customize configurations for different training scenarios.
"""

import os
from training.config import GRPOConfig, DatasetConfig, RewardConfig, create_default_config

def create_basic_config():
    """Create a basic configuration for GRPO training."""
    config = create_default_config()
    
    # Model configuration
    config.model.model_id_or_path = "/save/Qwen2.5-VL/qwen-vl-finetune/cxr_reasoning_initial/v0-20250429-011631/checkpoint-14"
    config.model.torch_dtype = "bfloat16"
    config.model.model_kwargs = {"weights_only": False}
    
    # Dataset configuration
    config.dataset.dataset_path = "/save/cxr_report/reasoning_inital_sft_data/qwen_training_data_for_grpo_standard_messages_uncertain.jsonl"
    config.dataset.max_length = 1536
    config.dataset.dataset_num_proc = 4
    
    # Training configuration
    config.training.num_train_epochs = 1
    config.training.per_device_train_batch_size = 4
    config.training.per_device_eval_batch_size = 4
    config.training.gradient_accumulation_steps = 2
    config.training.learning_rate = 1e-7
    config.training.warmup_ratio = 0.05
    config.training.max_grad_norm = 0.5
    config.training.dataloader_num_workers = 4
    config.training.do_eval = False
    config.training.eval_steps = 100000000000
    config.training.save_steps = 1000
    config.training.save_total_limit = 100
    config.training.logging_steps = 1
    
    # GRPO configuration
    config.grpo.num_generations = 8
    config.grpo.temperature = 1.0
    config.grpo.top_p = 0.9
    config.grpo.top_k = 50
    config.grpo.max_completion_length = 1536
    config.grpo.async_generate = False
    config.grpo.log_completions = True
    config.grpo.num_iterations = 1
    config.grpo.num_infer_workers = 1
    
    # Reward functions (balanced for development)
    config.reward.format_cxr = 0.4
    config.reward.accuracy_cxr = 0.4
    config.reward.coherence_cxr = 0.2
    
    # VLLM configuration
    config.vllm.use_vllm = True
    config.vllm.vllm_server_host = "127.0.0.1"
    config.vllm.vllm_server_port = 8000
    config.vllm.vllm_device = "auto"
    
    # Output configuration
    config.output.output_dir = "output"
    config.output.report_to = "tensorboard"
    
    # Environment configuration
    config.environment.deepspeed = "zero3"
    config.environment.ddp_timeout = 180000000
    
    # Prompt configuration
    config.prompt.system_prompt_file = "/save/ms-swift/examples/train/grpo/prompt_cxr.txt"
    
    return config

def create_development_config():
    """Create a configuration for development/testing with smaller parameters."""
    config = create_basic_config()
    
    # Reduce parameters for faster development
    config.training.num_train_epochs = 1
    config.training.per_device_train_batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.training.save_steps = 100
    config.training.logging_steps = 10
    
    config.grpo.num_generations = 4
    config.grpo.max_completion_length = 512
    
    config.output.output_dir = "output_dev"
    
    return config

def create_production_config():
    """Create a configuration for production training with optimal parameters."""
    config = create_basic_config()
    
    # Production parameters
    config.training.num_train_epochs = 3
    config.training.per_device_train_batch_size = 8
    config.training.gradient_accumulation_steps = 4
    config.training.save_steps = 500
    config.training.eval_steps = 1000
    config.training.do_eval = True
    
    config.grpo.num_generations = 16
    config.grpo.num_iterations = 3
    
    # Enable all reward functions
    config.reward.format_cxr = 0.33
    config.reward.accuracy_cxr = 0.33
    config.reward.coherence_cxr = 0.34
    
    config.output.output_dir = "output_production"
    
    return config

def create_ablation_config(reward_function_name):
    """Create a configuration for ablation studies with single reward function."""
    config = create_basic_config()
    
    # Disable all reward functions
    config.reward.format_cxr = 0.0
    config.reward.accuracy_cxr = 0.0
    config.reward.coherence_cxr = 0.0
    
    # Enable only the specified reward function
    if hasattr(config.reward, reward_function_name):
        setattr(config.reward, reward_function_name, 1.0)
    
    config.output.output_dir = f"output_ablation_{reward_function_name}"
    
    return config

def save_example_configs():
    """Save example configurations to files."""
    configs = {
        "basic_config.py": create_basic_config(),
        "development_config.py": create_development_config(),
        "production_config.py": create_production_config(),
        "ablation_format_config.py": create_ablation_config("format_cxr"),
        "ablation_chexbert_config.py": create_ablation_config("f1_chexbert_cxr"),
        "ablation_radgraph_config.py": create_ablation_config("radgraph_f1_cxr"),
    }
    
    # Create examples directory if it doesn't exist
    examples_dir = os.path.dirname(__file__)
    
    for filename, config in configs.items():
        filepath = os.path.join(examples_dir, filename)
        config.save(filepath)
        print(f"Saved configuration to {filepath}")

if __name__ == "__main__":
    # Example usage
    print("Creating example configurations...")
    
    # Create basic configuration
    basic_config = create_basic_config()
    print("Basic configuration created")
    print(f"Model: {basic_config.model.model_id_or_path}")
    print(f"Dataset: {basic_config.dataset.dataset_path}")
    print(f"Output: {basic_config.output.output_dir}")
    
    # Save all example configurations
    save_example_configs()
    print("All example configurations saved!")