"""
Reward modeling for Q-Learning LLM.
Implements token-level reward assignment using RLVR approach.
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import re


def verify_math_answer(generated_text: str, expected_answer: str) -> bool:
    """
    Verify if a generated solution arrives at the correct answer.
    Simple string matching for POC.
    """
    # Try to extract #### answer format (GSM8K style)
    match = re.search(r'####\s*(-?[\d,]+)', generated_text)
    if match:
        extracted = match.group(1).replace(',', '')
        return extracted == expected_answer
    
    # Fallback: check if answer appears at end
    return expected_answer in generated_text[-50:]


def compute_token_rewards_uniform(
    seq_length: int,
    is_correct: bool,
    correct_reward: float = 1.0,
    incorrect_reward: float = -1.0,
    prompt_length: int = 0
) -> torch.Tensor:
    """
    Distribute reward uniformly across all generated tokens.
    Simplest approach - equal reward for each token.
    """
    total_reward = correct_reward if is_correct else incorrect_reward
    
    # Only reward generated tokens (after prompt)
    rewards = torch.zeros(seq_length)
    gen_length = seq_length - prompt_length
    
    if gen_length > 0:
        per_token_reward = total_reward / gen_length
        rewards[prompt_length:] = per_token_reward
    
    return rewards


def compute_token_rewards_exponential(
    seq_length: int,
    is_correct: bool,
    correct_reward: float = 1.0,
    incorrect_reward: float = -1.0,
    prompt_length: int = 0,
    decay: float = 0.9
) -> torch.Tensor:
    """
    Distribute reward with exponential weighting toward end.
    Later tokens (closer to outcome) get more reward.
    This reflects that final answer matters most.
    """
    total_reward = correct_reward if is_correct else incorrect_reward
    rewards = torch.zeros(seq_length)
    gen_length = seq_length - prompt_length
    
    if gen_length > 0:
        # Create exponentially increasing weights
        positions = torch.arange(gen_length, dtype=torch.float32)
        weights = decay ** (gen_length - 1 - positions)  # Higher weight for later tokens
        weights = weights / weights.sum()  # Normalize
        
        rewards[prompt_length:] = total_reward * weights
    
    return rewards


def compute_token_rewards_heuristic(
    input_ids: torch.Tensor,
    seq_length: int,
    is_correct: bool,
    tokenizer,
    correct_reward: float = 1.0,
    incorrect_reward: float = -1.0,
    prompt_length: int = 0
) -> torch.Tensor:
    """
    Heuristically localize rewards to specific tokens.
    
    For CORRECT solutions:
    - Higher reward for calculation tokens (=, numbers)
    - Higher reward for final answer tokens
    
    For INCORRECT solutions:
    - Negative reward focused on calculation tokens
    - These are likely where errors occur
    """
    rewards = torch.zeros(seq_length)
    gen_length = seq_length - prompt_length
    
    if gen_length <= 0:
        return rewards
    
    base_reward = correct_reward if is_correct else incorrect_reward
    
    # Decode tokens to identify important ones
    tokens = [tokenizer.decode([tid]) for tid in input_ids[prompt_length:]]
    
    weights = torch.ones(gen_length)
    
    for i, token in enumerate(tokens):
        # Boost weight for calculation indicators
        if any(c in token for c in ['=', '+', '-', '*', '/', '×', '÷']):
            weights[i] *= 2.0
        
        # Boost weight for numbers (where errors often occur)
        if any(c.isdigit() for c in token):
            weights[i] *= 1.5
        
        # Boost weight for answer marker
        if '####' in token or '#' in token:
            weights[i] *= 3.0
    
    # Normalize and apply
    weights = weights / weights.sum()
    rewards[prompt_length:] = base_reward * weights
    
    return rewards


def compute_token_rewards(
    input_ids: torch.Tensor,
    is_correct: bool,
    prompt_length: int,
    tokenizer=None,
    method: str = "exponential",
    correct_reward: float = 1.0,
    incorrect_reward: float = -1.0,
    decay: float = 0.9
) -> torch.Tensor:
    """
    Main entry point for computing token-level rewards.
    
    Args:
        input_ids: Token IDs of the full sequence
        is_correct: Whether the solution is correct
        prompt_length: Length of the prompt (no reward for prompt tokens)
        tokenizer: Tokenizer (needed for heuristic method)
        method: "uniform", "exponential", or "heuristic"
        correct_reward: Reward for correct solutions
        incorrect_reward: Reward for incorrect solutions
        decay: Decay factor for exponential method
    """
    seq_length = input_ids.shape[0] if len(input_ids.shape) == 1 else input_ids.shape[1]
    
    if method == "uniform":
        return compute_token_rewards_uniform(
            seq_length, is_correct, correct_reward, incorrect_reward, prompt_length
        )
    elif method == "exponential":
        return compute_token_rewards_exponential(
            seq_length, is_correct, correct_reward, incorrect_reward, prompt_length, decay
        )
    elif method == "heuristic":
        if tokenizer is None:
            raise ValueError("Tokenizer required for heuristic reward computation")
        return compute_token_rewards_heuristic(
            input_ids, seq_length, is_correct, tokenizer,
            correct_reward, incorrect_reward, prompt_length
        )
    else:
        raise ValueError(f"Unknown reward method: {method}")

