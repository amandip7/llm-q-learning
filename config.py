"""
Configuration for Q-Learning LLM experiment.
Minimal settings for proof-of-concept.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Model settings
    model_name: str = "gpt2"  # Small model for POC
    
    # Q-Learning hyperparameters
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update rate for target network
    learning_rate: float = 1e-5
    
    # Training settings
    batch_size: int = 4
    max_seq_length: int = 256
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    
    # Reward settings
    correct_reward: float = 1.0
    incorrect_reward: float = -1.0
    
    # Token-level reward distribution
    # "uniform" = spread reward equally across all tokens
    # "exponential" = weight more toward final tokens (outcome-based)
    # "heuristic" = try to localize rewards to specific tokens
    reward_distribution: str = "exponential"
    reward_decay: float = 0.9  # For exponential distribution
    
    # Dataset settings
    dataset_name: str = "gsm8k"  # Grade school math
    max_train_samples: Optional[int] = 1000  # Limit for POC
    max_eval_samples: Optional[int] = 100
    
    # Device
    device: str = "cuda"  # Will fallback to CPU if unavailable
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    output_dir: str = "./outputs"

