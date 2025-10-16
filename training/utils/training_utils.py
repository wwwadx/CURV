"""
Training Utilities

Utilities for setting up training environment, logging, and monitoring.
"""

import os
import json
import logging
import time
import psutil
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import numpy as np
from transformers import TrainerCallback, TrainerState, TrainerControl


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    epoch: float
    loss: float
    learning_rate: float
    grad_norm: Optional[float] = None
    rewards: Optional[Dict[str, float]] = None
    kl_divergence: Optional[float] = None
    timestamp: Optional[str] = None


class TrainingMonitor(TrainerCallback):
    """
    Callback for monitoring training progress and metrics.
    """
    
    def __init__(self, log_dir: str, save_interval: int = 100):
        """
        Initialize training monitor.
        
        Args:
            log_dir: Directory for saving logs
            save_interval: Interval for saving metrics
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        
        self.metrics_history: List[TrainingMetrics] = []
        self.best_metrics = {}
        self.start_time = None
        
        self.logger = logging.getLogger(__name__)
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training."""
        self.start_time = time.time()
        self.logger.info("Training started")
        
        # Log system information
        self._log_system_info()
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called when logging."""
        if logs is None:
            return
        
        # Create metrics object
        metrics = TrainingMetrics(
            step=state.global_step,
            epoch=state.epoch,
            loss=logs.get('train_loss', 0.0),
            learning_rate=logs.get('learning_rate', 0.0),
            grad_norm=logs.get('grad_norm'),
            rewards=self._extract_reward_metrics(logs),
            kl_divergence=logs.get('kl_divergence'),
            timestamp=datetime.now().isoformat()
        )
        
        self.metrics_history.append(metrics)
        
        # Save metrics periodically
        if state.global_step % self.save_interval == 0:
            self._save_metrics()
        
        # Update best metrics
        self._update_best_metrics(logs)
        
        # Log to console
        self._log_metrics(metrics)
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called after evaluation."""
        if logs is None:
            return
        
        self.logger.info(f"Evaluation at step {state.global_step}:")
        for key, value in logs.items():
            if key.startswith('eval_'):
                self.logger.info(f"  {key}: {value:.4f}")
    
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when saving checkpoint."""
        self._save_metrics()
        self._save_training_state(state)
        self.logger.info(f"Checkpoint saved at step {state.global_step}")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        training_time = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Total steps: {state.global_step}")
        self.logger.info(f"Final loss: {state.log_history[-1].get('train_loss', 'N/A')}")
        
        # Save final metrics
        self._save_metrics()
        self._save_training_summary(state, training_time)
    
    def _extract_reward_metrics(self, logs: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract reward metrics from logs."""
        reward_metrics = {}
        
        for key, value in logs.items():
            if key.startswith('reward_') or key.startswith('train/reward_'):
                reward_name = key.replace('train/', '').replace('reward_', '')
                reward_metrics[reward_name] = value
        
        return reward_metrics if reward_metrics else None
    
    def _update_best_metrics(self, logs: Dict[str, Any]):
        """Update best metrics tracking."""
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                if key not in self.best_metrics:
                    self.best_metrics[key] = {'value': value, 'step': len(self.metrics_history)}
                else:
                    # For loss metrics, lower is better
                    if 'loss' in key.lower():
                        if value < self.best_metrics[key]['value']:
                            self.best_metrics[key] = {'value': value, 'step': len(self.metrics_history)}
                    # For reward/accuracy metrics, higher is better
                    elif any(term in key.lower() for term in ['reward', 'accuracy', 'f1']):
                        if value > self.best_metrics[key]['value']:
                            self.best_metrics[key] = {'value': value, 'step': len(self.metrics_history)}
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log metrics to console."""
        log_msg = f"Step {metrics.step} | Loss: {metrics.loss:.4f} | LR: {metrics.learning_rate:.2e}"
        
        if metrics.rewards:
            reward_str = " | ".join([f"{k}: {v:.3f}" for k, v in metrics.rewards.items()])
            log_msg += f" | Rewards: {reward_str}"
        
        if metrics.kl_divergence is not None:
            log_msg += f" | KL: {metrics.kl_divergence:.4f}"
        
        self.logger.info(log_msg)
    
    def _save_metrics(self):
        """Save metrics to file."""
        metrics_file = self.log_dir / 'training_metrics.jsonl'
        
        with open(metrics_file, 'w') as f:
            for metrics in self.metrics_history:
                f.write(json.dumps(asdict(metrics)) + '\n')
    
    def _save_training_state(self, state: TrainerState):
        """Save training state."""
        state_file = self.log_dir / 'training_state.json'
        
        state_dict = {
            'global_step': state.global_step,
            'epoch': state.epoch,
            'max_steps': state.max_steps,
            'num_train_epochs': state.num_train_epochs,
            'best_metrics': self.best_metrics
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_dict, f, indent=2)
    
    def _save_training_summary(self, state: TrainerState, training_time: float):
        """Save training summary."""
        summary_file = self.log_dir / 'training_summary.json'
        
        summary = {
            'training_completed': True,
            'total_steps': state.global_step,
            'total_epochs': state.epoch,
            'training_time_seconds': training_time,
            'best_metrics': self.best_metrics,
            'final_metrics': asdict(self.metrics_history[-1]) if self.metrics_history else None
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _log_system_info(self):
        """Log system information."""
        system_info = {
            'python_version': os.sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                system_info[f'gpu_{i}_name'] = gpu_name
                system_info[f'gpu_{i}_memory_gb'] = gpu_memory
        
        system_file = self.log_dir / 'system_info.json'
        with open(system_file, 'w') as f:
            json.dump(system_info, f, indent=2)
        
        self.logger.info("System information logged")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Log file path (optional)
        log_format: Log format string (optional)
        
    Returns:
        Configured logger
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Add file handler if specified
    handlers = [console_handler]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Configure logger
    logger = logging.getLogger()
    logger.handlers = handlers
    
    return logger


def setup_environment(
    seed: int = 42,
    deterministic: bool = False,
    cuda_deterministic: bool = False
):
    """
    Setup training environment.
    
    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms
        cuda_deterministic: Whether to use CUDA deterministic algorithms
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if cuda_deterministic:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.use_deterministic_algorithms(True)
    
    # Set environment variables for better performance
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def save_training_state(
    state_dict: Dict[str, Any],
    output_dir: str,
    filename: str = "training_state.json"
):
    """
    Save training state to file.
    
    Args:
        state_dict: State dictionary to save
        output_dir: Output directory
        filename: Output filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(state_dict, f, indent=2)


def load_training_state(
    state_file: str
) -> Dict[str, Any]:
    """
    Load training state from file.
    
    Args:
        state_file: Path to state file
        
    Returns:
        Loaded state dictionary
    """
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"State file not found: {state_file}")
    
    with open(state_file, 'r') as f:
        return json.load(f)


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Get GPU memory usage statistics.
    
    Returns:
        Dictionary with GPU memory usage
    """
    if not torch.cuda.is_available():
        return {}
    
    memory_stats = {}
    
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)   # GB
        memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        
        memory_stats[f'gpu_{i}'] = {
            'allocated_gb': memory_allocated,
            'reserved_gb': memory_reserved,
            'total_gb': memory_total,
            'utilization_percent': (memory_allocated / memory_total) * 100
        }
    
    return memory_stats


def log_model_info(model: torch.nn.Module, logger: Optional[logging.Logger] = None):
    """
    Log model information.
    
    Args:
        model: Model to analyze
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model Information:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable percentage: {(trainable_params/total_params)*100:.2f}%")
    
    # Log memory usage if CUDA is available
    if torch.cuda.is_available():
        memory_stats = get_gpu_memory_usage()
        for gpu_id, stats in memory_stats.items():
            logger.info(f"  {gpu_id} memory: {stats['allocated_gb']:.2f}GB / {stats['total_gb']:.2f}GB")


def create_output_directory(base_dir: str, experiment_name: str) -> str:
    """
    Create output directory with timestamp.
    
    Args:
        base_dir: Base output directory
        experiment_name: Name of the experiment
        
    Returns:
        Created output directory path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def cleanup_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 3,
    keep_best: bool = True
):
    """
    Cleanup old checkpoints to save disk space.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to keep the best checkpoint
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # Find all checkpoint directories
    checkpoint_dirs = []
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path) and item.startswith('checkpoint-'):
            try:
                step = int(item.split('-')[1])
                checkpoint_dirs.append((step, item_path))
            except (IndexError, ValueError):
                continue
    
    # Sort by step number
    checkpoint_dirs.sort(key=lambda x: x[0])
    
    # Keep the last N checkpoints
    to_keep = set()
    if len(checkpoint_dirs) > keep_last_n:
        for step, path in checkpoint_dirs[-keep_last_n:]:
            to_keep.add(path)
    else:
        for step, path in checkpoint_dirs:
            to_keep.add(path)
    
    # Keep the best checkpoint if specified
    if keep_best:
        # This would require additional logic to identify the best checkpoint
        # For now, we'll keep the last checkpoint as a proxy
        if checkpoint_dirs:
            to_keep.add(checkpoint_dirs[-1][1])
    
    # Remove old checkpoints
    for step, path in checkpoint_dirs:
        if path not in to_keep:
            import shutil
            shutil.rmtree(path)
            logging.info(f"Removed old checkpoint: {path}")


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    steps_per_second: float
) -> Dict[str, float]:
    """
    Estimate training time.
    
    Args:
        num_samples: Number of training samples
        batch_size: Batch size
        num_epochs: Number of epochs
        steps_per_second: Processing speed in steps per second
        
    Returns:
        Dictionary with time estimates
    """
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * num_epochs
    total_seconds = total_steps / steps_per_second
    
    return {
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'estimated_seconds': total_seconds,
        'estimated_minutes': total_seconds / 60,
        'estimated_hours': total_seconds / 3600
    }