"""
Training script for Q-Learning LLM.
Implements Double DQN training loop with off-policy data.
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from typing import Dict, List
import json

from config import Config
from dataset import MathDataset
from q_network import QLearningLLM
from reward import compute_token_rewards


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for variable-length sequences."""
    # Find max length in batch
    max_len = max(item["input_ids"].shape[0] for item in batch)
    
    # Pad sequences
    input_ids = []
    attention_masks = []
    prompt_lengths = []
    is_correct = []
    
    for item in batch:
        seq_len = item["input_ids"].shape[0]
        padding_len = max_len - seq_len
        
        # Pad input_ids with pad token (0 for GPT-2)
        padded_ids = F.pad(item["input_ids"], (0, padding_len), value=0)
        input_ids.append(padded_ids)
        
        # Pad attention mask with 0s
        padded_mask = F.pad(item["attention_mask"], (0, padding_len), value=0)
        attention_masks.append(padded_mask)
        
        prompt_lengths.append(item["prompt_length"])
        is_correct.append(item["is_correct"])
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "prompt_lengths": prompt_lengths,
        "is_correct": is_correct
    }


def train_step(
    model: QLearningLLM,
    batch: Dict,
    config: Config,
    optimizer: torch.optim.Optimizer
) -> Dict[str, float]:
    """
    Single training step with Double DQN loss.
    
    Loss = MSE(Q_online(s, a), r + γ * Q_target(s', argmax Q_online(s', a')))
    """
    device = model.device
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    batch_size, seq_len = input_ids.shape
    
    # Compute token-level rewards for each sequence
    rewards_list = []
    for i in range(batch_size):
        r = compute_token_rewards(
            input_ids=input_ids[i],
            is_correct=batch["is_correct"][i],
            prompt_length=batch["prompt_lengths"][i],
            tokenizer=model.tokenizer,
            method=config.reward_distribution,
            correct_reward=config.correct_reward,
            incorrect_reward=config.incorrect_reward,
            decay=config.reward_decay
        )
        rewards_list.append(r)
    
    rewards = torch.stack(rewards_list).to(device)
    
    # Create terminal mask (last real token is terminal)
    terminal_mask = torch.zeros_like(input_ids, dtype=torch.float)
    for i in range(batch_size):
        real_len = attention_mask[i].sum().item()
        if real_len > 0:
            terminal_mask[i, int(real_len) - 1] = 1.0
    
    # Get current Q-values for taken actions
    current_q = model.get_action_q_values(input_ids, attention_mask, use_target=False)
    
    # Get target Q-values using Double DQN
    target_q = model.compute_target_q_values(
        input_ids, attention_mask, rewards,
        gamma=config.gamma,
        terminal_mask=terminal_mask
    )
    
    # Mask out padding positions
    valid_mask = attention_mask[:, 1:].float()  # Shifted for action positions
    
    # TD Loss (MSE)
    td_error = (current_q - target_q) * valid_mask
    loss = (td_error ** 2).sum() / valid_mask.sum()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.online_network.parameters(), 1.0)
    optimizer.step()
    
    # Soft update target network
    model.soft_update_target()
    
    return {
        "loss": loss.item(),
        "mean_q": current_q.mean().item(),
        "mean_target_q": target_q.mean().item()
    }


def train(config: Config):
    """Main training loop."""
    print(f"Starting Q-Learning LLM training...")
    print(f"Model: {config.model_name}")
    print(f"Reward distribution: {config.reward_distribution}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize model
    model = QLearningLLM(
        model_name=config.model_name,
        device=config.device,
        tau=config.tau
    )
    print(f"Model loaded on {model.device}")
    
    # Load dataset
    print("Loading GSM8K dataset...")
    train_dataset = MathDataset(
        tokenizer=model.tokenizer,
        max_samples=config.max_train_samples,
        split="train",
        max_length=config.max_seq_length
    )
    print(f"Loaded {len(train_dataset)} training examples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = AdamW(
        model.online_network.parameters(),
        lr=config.learning_rate
    )
    
    # Training loop
    global_step = 0
    metrics_history = []
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Training")
        for batch in pbar:
            metrics = train_step(model, batch, config, optimizer)
            epoch_loss += metrics["loss"]
            global_step += 1
            
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "Q": f"{metrics['mean_q']:.2f}"
            })
            
            if global_step % config.log_interval == 0:
                metrics_history.append({
                    "step": global_step,
                    **metrics
                })
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.online_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config
        }, checkpoint_path)
    
    # Save final model and metrics
    torch.save(model.online_network.state_dict(), 
               os.path.join(config.output_dir, "final_model.pt"))
    
    with open(os.path.join(config.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_history, f)
    
    print(f"\nTraining complete! Model saved to {config.output_dir}")
    return model


if __name__ == "__main__":
    config = Config()
    train(config)

