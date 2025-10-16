"""
GRPO Trainer for Medical Vision-Language Models

Implementation of GRPO (Generalized Reward-based Policy Optimization) trainer
for training medical vision-language models with multiple reward functions.
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from transformers import (
    Trainer, 
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer
)

from ..config.grpo_config import GRPOConfig
from ..reward_functions.base_reward import BaseRewardFunction


@dataclass
class GRPOTrainingArguments(TrainingArguments):
    """
    Extended training arguments for GRPO training.
    """
    beta: float = 0.1
    ref_model_path: Optional[str] = None
    vllm_enable: bool = True
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_max_model_len: int = 2048


class GRPOTrainer(Trainer):
    """
    GRPO Trainer for medical vision-language models.
    
    This trainer implements GRPO optimization with multiple reward functions
    specifically designed for medical report generation tasks.
    """
    
    def __init__(
        self,
        config: GRPOConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset,
        eval_dataset,
        reward_functions: Dict[str, BaseRewardFunction],
        ref_model: Optional[PreTrainedModel] = None,
        **kwargs
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            config: GRPO configuration
            model: Policy model to train
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            reward_functions: Dictionary of reward functions
            ref_model: Reference model for KL divergence (optional)
            **kwargs: Additional arguments for Trainer
        """
        self.config = config
        self.reward_functions = reward_functions
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        
        # Create training arguments
        training_args = GRPOTrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            save_total_limit=config.save_total_limit,
            seed=config.seed,
            dataloader_num_workers=config.dataloader_num_workers,
            gradient_checkpointing=config.gradient_checkpointing,
            local_rank=config.local_rank,
            ddp_backend=config.ddp_backend,
            logging_dir=config.logging_dir,
            beta=config.beta,
            vllm_enable=config.vllm_enable,
            vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            vllm_tensor_parallel_size=config.vllm_tensor_parallel_size,
            vllm_max_model_len=config.vllm_max_model_len,
        )
        
        # Initialize parent trainer
        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs
        )
        
        # Setup VLLM if enabled
        self.vllm_engine = None
        if config.vllm_enable:
            self._setup_vllm()
        
        # Initialize metrics tracking
        self.reward_history = {name: [] for name in reward_functions.keys()}
        self.kl_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_vllm(self):
        """Setup VLLM engine for fast inference during training."""
        try:
            # This is a placeholder for VLLM setup
            # Actual implementation would depend on VLLM library
            self.logger.info("Setting up VLLM engine for inference...")
            
            # Mock VLLM engine setup
            self.vllm_engine = MockVLLMEngine(
                model_path=self.config.model_id_or_path,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                tensor_parallel_size=self.config.vllm_tensor_parallel_size,
                max_model_len=self.config.vllm_max_model_len
            )
            
            self.logger.info("VLLM engine setup completed")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup VLLM: {e}")
            self.logger.warning("Falling back to standard inference")
            self.vllm_engine = None
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute GRPO loss with multiple reward functions.
        
        Args:
            model: The model
            inputs: Input batch
            return_outputs: Whether to return outputs
            
        Returns:
            Loss tensor and optionally outputs
        """
        # Generate responses using current policy
        with torch.no_grad():
            generated_texts = self._generate_responses(inputs)
        
        # Compute rewards
        rewards = self._compute_rewards(generated_texts, inputs)
        
        # Compute KL divergence with reference model
        kl_divergence = self._compute_kl_divergence(model, inputs, generated_texts)
        
        # Compute GRPO loss
        policy_loss = self._compute_policy_loss(model, inputs, generated_texts, rewards)
        
        # Combine losses
        total_loss = policy_loss + self.args.beta * kl_divergence
        
        # Log metrics
        self._log_training_metrics(rewards, kl_divergence, policy_loss, total_loss)
        
        if return_outputs:
            return total_loss, {"generated_texts": generated_texts, "rewards": rewards}
        return total_loss
    
    def _generate_responses(self, inputs: Dict[str, torch.Tensor]) -> List[str]:
        """
        Generate responses using the current policy model.
        
        Args:
            inputs: Input batch
            
        Returns:
            List of generated text responses
        """
        if self.vllm_engine:
            # Use VLLM for fast generation
            return self._generate_with_vllm(inputs)
        else:
            # Use standard generation
            return self._generate_with_model(inputs)
    
    def _generate_with_vllm(self, inputs: Dict[str, torch.Tensor]) -> List[str]:
        """Generate responses using VLLM engine."""
        # This is a placeholder implementation
        # Actual VLLM integration would be more complex
        prompts = self._extract_prompts_from_inputs(inputs)
        return self.vllm_engine.generate(prompts)
    
    def _generate_with_model(self, inputs: Dict[str, torch.Tensor]) -> List[str]:
        """Generate responses using the model directly."""
        with torch.no_grad():
            # Extract input_ids and attention_mask
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask')
            
            # Generate responses
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode generated text
            generated_texts = []
            for i, generated_seq in enumerate(generated_ids):
                # Remove input tokens to get only generated part
                input_length = input_ids[i].shape[0]
                generated_tokens = generated_seq[input_length:]
                
                # Decode to text
                generated_text = self.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True
                )
                generated_texts.append(generated_text)
            
            return generated_texts
    
    def _extract_prompts_from_inputs(self, inputs: Dict[str, torch.Tensor]) -> List[str]:
        """Extract prompts from input tensors."""
        input_ids = inputs.get('input_ids')
        prompts = []
        
        for input_seq in input_ids:
            prompt = self.tokenizer.decode(input_seq, skip_special_tokens=True)
            prompts.append(prompt)
        
        return prompts
    
    def _compute_rewards(self, generated_texts: List[str], inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute rewards using all configured reward functions.
        
        Args:
            generated_texts: Generated text responses
            inputs: Input batch (may contain ground truth for some reward functions)
            
        Returns:
            Combined reward tensor
        """
        batch_size = len(generated_texts)
        total_rewards = torch.zeros(batch_size, device=self.model.device)
        
        # Extract ground truth if available
        ground_truth = self._extract_ground_truth(inputs)
        
        for func_name, reward_func in self.reward_functions.items():
            try:
                # Compute rewards for this function
                if ground_truth:
                    func_rewards = reward_func(generated_texts, ground_truth=ground_truth)
                else:
                    func_rewards = reward_func(generated_texts)
                
                # Convert to tensor and apply weight
                func_rewards_tensor = torch.tensor(
                    func_rewards, 
                    dtype=torch.float32, 
                    device=self.model.device
                )
                
                weight = self.config.reward_weights.get(func_name, 1.0)
                total_rewards += weight * func_rewards_tensor
                
                # Track reward history
                self.reward_history[func_name].extend(func_rewards)
                
            except Exception as e:
                self.logger.warning(f"Error computing rewards for {func_name}: {e}")
        
        return total_rewards
    
    def _extract_ground_truth(self, inputs: Dict[str, torch.Tensor]) -> Optional[List[str]]:
        """Extract ground truth from inputs if available."""
        # This would depend on your dataset format
        # For now, return None as placeholder
        return None
    
    def _compute_kl_divergence(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor], 
                              generated_texts: List[str]) -> torch.Tensor:
        """
        Compute KL divergence between current policy and reference model.
        
        Args:
            model: Current policy model
            inputs: Input batch
            generated_texts: Generated responses
            
        Returns:
            KL divergence tensor
        """
        if self.ref_model is None:
            return torch.tensor(0.0, device=model.device)
        
        # This is a simplified implementation
        # Actual KL computation would be more sophisticated
        batch_size = len(generated_texts)
        kl_div = torch.zeros(batch_size, device=model.device)
        
        # Placeholder KL computation
        # In practice, you would compute log probabilities from both models
        # and calculate KL divergence
        
        return kl_div.mean()
    
    def _compute_policy_loss(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor],
                           generated_texts: List[str], rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute policy gradient loss.
        
        Args:
            model: Policy model
            inputs: Input batch
            generated_texts: Generated responses
            rewards: Reward values
            
        Returns:
            Policy loss tensor
        """
        # This is a simplified implementation
        # Actual policy gradient computation would be more complex
        
        # For now, return a placeholder loss
        return -rewards.mean()
    
    def _log_training_metrics(self, rewards: torch.Tensor, kl_divergence: torch.Tensor,
                            policy_loss: torch.Tensor, total_loss: torch.Tensor):
        """Log training metrics."""
        # Log to console and tensorboard/wandb if available
        metrics = {
            'train/total_loss': total_loss.item(),
            'train/policy_loss': policy_loss.item(),
            'train/kl_divergence': kl_divergence.item(),
            'train/mean_reward': rewards.mean().item(),
            'train/reward_std': rewards.std().item()
        }
        
        # Log individual reward function scores
        for func_name in self.reward_functions.keys():
            if self.reward_history[func_name]:
                recent_scores = self.reward_history[func_name][-len(rewards):]
                metrics[f'train/reward_{func_name}'] = np.mean(recent_scores)
        
        # Log metrics (this would integrate with your logging framework)
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value:.4f}")
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Evaluate the model using reward functions.
        
        Args:
            eval_dataset: Evaluation dataset
            ignore_keys: Keys to ignore in evaluation
            metric_key_prefix: Prefix for metric names
            
        Returns:
            Evaluation metrics
        """
        eval_dataset = eval_dataset or self.eval_dataset
        
        # Generate responses for evaluation
        eval_results = {}
        total_rewards = []
        
        for batch in self.get_eval_dataloader(eval_dataset):
            with torch.no_grad():
                generated_texts = self._generate_responses(batch)
                rewards = self._compute_rewards(generated_texts, batch)
                total_rewards.extend(rewards.cpu().numpy())
        
        # Compute evaluation metrics
        eval_results[f"{metric_key_prefix}/mean_reward"] = np.mean(total_rewards)
        eval_results[f"{metric_key_prefix}/reward_std"] = np.std(total_rewards)
        
        # Individual reward function metrics
        for func_name in self.reward_functions.keys():
            if self.reward_history[func_name]:
                recent_scores = self.reward_history[func_name][-100:]  # Last 100 scores
                eval_results[f"{metric_key_prefix}/reward_{func_name}"] = np.mean(recent_scores)
        
        return eval_results
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the trained model and training state.
        
        Args:
            output_dir: Output directory
            _internal_call: Whether this is an internal call
        """
        output_dir = output_dir or self.args.output_dir
        
        # Save the model
        super().save_model(output_dir, _internal_call)
        
        # Save reward function configurations and history
        reward_state = {
            'reward_functions': list(self.reward_functions.keys()),
            'reward_weights': self.config.reward_weights,
            'reward_history': self.reward_history,
            'kl_history': self.kl_history
        }
        
        import json
        with open(os.path.join(output_dir, 'reward_state.json'), 'w') as f:
            json.dump(reward_state, f, indent=2)
        
        self.logger.info(f"Model and reward state saved to {output_dir}")


class MockVLLMEngine:
    """Mock VLLM engine for testing purposes."""
    
    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.9,
                 tensor_parallel_size: int = 1, max_model_len: int = 2048):
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
    
    def generate(self, prompts: List[str]) -> List[str]:
        """Mock generation method."""
        # Return placeholder responses
        responses = []
        for prompt in prompts:
            response = f"<findings>Mock findings for prompt.</findings><thinking>Mock thinking process.</thinking><impression>Mock impression based on findings.</impression>"
            responses.append(response)
        return responses