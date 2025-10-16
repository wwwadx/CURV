#!/usr/bin/env python3
"""
GRPO Training Script for Medical Vision-Language Models

This script implements GRPO (Generalized Reward-based Policy Optimization)
training for medical vision-language models using multiple reward functions.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add CURV to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.config.grpo_config import GRPOConfig, DatasetConfig, RewardConfig
from training.reward_functions import get_reward_function, list_reward_functions
from training.trainers.grpo_trainer import GRPOTrainer
from training.utils.logging_utils import setup_logging
from training.utils.model_utils import load_model_and_tokenizer
from training.utils.data_utils import prepare_dataset


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GRPO Training for Medical VLMs")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="qwen-vl-chat",
                       help="Type of model to train")
    parser.add_argument("--model_id_or_path", type=str, default="qwen/Qwen-VL-Chat",
                       help="Model ID or path")
    parser.add_argument("--sft_type", type=str, default="lora",
                       choices=["lora", "full"], help="SFT type")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, nargs="+", default=["medical-cxr"],
                       help="Dataset names")
    parser.add_argument("--dataset_test_ratio", type=float, default=0.01,
                       help="Test dataset ratio")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                       help="Gradient accumulation steps")
    
    # GRPO arguments
    parser.add_argument("--beta", type=float, default=0.1,
                       help="KL divergence coefficient")
    parser.add_argument("--reward_funcs", type=str, nargs="+",
                       default=["format_cxr", "accuracy_cxr", "f1_chexbert_cxr"],
                       help="Reward functions to use")
    
    # VLLM arguments
    parser.add_argument("--vllm_enable", action="store_true", default=True,
                       help="Enable VLLM for inference")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9,
                       help="VLLM GPU memory utilization")
    
    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory")
    parser.add_argument("--logging_dir", type=str, default=None,
                       help="Logging directory")
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    
    # Configuration file
    parser.add_argument("--config_file", type=str, default=None,
                       help="Path to configuration file")
    
    # Debug arguments
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--dry_run", action="store_true",
                       help="Perform dry run without actual training")
    
    return parser.parse_args()


def validate_reward_functions(reward_funcs: List[str]) -> List[str]:
    """
    Validate that all specified reward functions are available.
    
    Args:
        reward_funcs: List of reward function names
        
    Returns:
        List of validated reward function names
        
    Raises:
        ValueError: If any reward function is not available
    """
    available_funcs = list_reward_functions()
    invalid_funcs = [func for func in reward_funcs if func not in available_funcs]
    
    if invalid_funcs:
        raise ValueError(
            f"Invalid reward functions: {invalid_funcs}. "
            f"Available functions: {available_funcs}"
        )
    
    return reward_funcs


def setup_reward_functions(config: GRPOConfig, reward_config: RewardConfig) -> Dict[str, Any]:
    """
    Setup and initialize reward functions.
    
    Args:
        config: GRPO configuration
        reward_config: Reward configuration
        
    Returns:
        Dictionary of initialized reward functions
    """
    reward_functions = {}
    
    for func_name in config.reward_funcs:
        try:
            # Get function-specific configuration
            func_kwargs = {
                'weight': config.reward_weights.get(func_name, 1.0),
                'debug_mode': reward_config.debug_mode,
                'log_path': reward_config.log_path
            }
            
            # Add function-specific parameters
            if func_name == 'f1_chexbert_cxr':
                func_kwargs.update({
                    'model_path': reward_config.chexbert_model_path,
                    'f1_threshold': reward_config.chexbert_threshold
                })
            elif func_name == 'radgraph_f1_cxr':
                func_kwargs.update({
                    'model_path': reward_config.radgraph_model_path,
                    'f1_threshold': reward_config.radgraph_threshold
                })
            elif func_name == 'accuracy_cxr':
                func_kwargs.update({
                    'accuracy_threshold': reward_config.accuracy_threshold
                })
            
            # Initialize reward function
            reward_func = get_reward_function(func_name, **func_kwargs)
            reward_functions[func_name] = reward_func
            
            logging.info(f"Initialized reward function: {func_name}")
            
        except Exception as e:
            logging.error(f"Failed to initialize reward function {func_name}: {e}")
            raise
    
    return reward_functions


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(
        log_level=logging.DEBUG if args.debug else logging.INFO,
        log_file=os.path.join(args.output_dir, "training.log") if args.output_dir else None
    )
    
    logging.info("Starting GRPO training for medical vision-language models")
    logging.info(f"Arguments: {vars(args)}")
    
    try:
        # Load or create configuration
        if args.config_file and os.path.exists(args.config_file):
            logging.info(f"Loading configuration from {args.config_file}")
            config = GRPOConfig.load(args.config_file)
        else:
            logging.info("Creating configuration from arguments")
            config = GRPOConfig()
            
            # Update config with command line arguments
            for key, value in vars(args).items():
                if hasattr(config, key) and value is not None:
                    setattr(config, key, value)
        
        # Create additional configurations
        dataset_config = DatasetConfig()
        reward_config = RewardConfig(
            debug_mode=args.debug,
            log_path=os.path.join(config.output_dir, "reward_logs.txt")
        )
        
        # Save configuration
        config_path = os.path.join(config.output_dir, "grpo_config.json")
        config.save(config_path)
        logging.info(f"Saved configuration to {config_path}")
        
        # Validate reward functions
        validate_reward_functions(config.reward_funcs)
        
        # Setup reward functions
        logging.info("Setting up reward functions...")
        reward_functions = setup_reward_functions(config, reward_config)
        
        if args.dry_run:
            logging.info("Dry run completed successfully")
            return
        
        # Load model and tokenizer
        logging.info("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(
            model_id_or_path=config.model_id_or_path,
            model_type=config.model_type,
            sft_type=config.sft_type,
            lora_config={
                'rank': config.lora_rank,
                'alpha': config.lora_alpha,
                'dropout': config.lora_dropout,
                'target_modules': config.lora_target_modules
            }
        )
        
        # Prepare dataset
        logging.info("Preparing dataset...")
        train_dataset, eval_dataset = prepare_dataset(
            dataset_names=config.dataset,
            dataset_config=dataset_config,
            tokenizer=tokenizer,
            test_ratio=config.dataset_test_ratio,
            seed=config.dataset_seed
        )
        
        # Initialize trainer
        logging.info("Initializing GRPO trainer...")
        trainer = GRPOTrainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            reward_functions=reward_functions
        )
        
        # Start training
        logging.info("Starting training...")
        trainer.train()
        
        # Save final model
        logging.info("Saving final model...")
        trainer.save_model()
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()