"""
DPO (Direct Preference Optimization) training for LLM post-training.
Alternative to Q-learning for ablation study comparison.

DPO Loss:
    L = -log(sigmoid(beta * (log_prob_chosen - log_prob_rejected)))

This implementation uses the EXACT same dataset as Q-learning:
- Same correct solutions (chosen)
- Same incorrect solutions (rejected)
- Same random seed for reproducibility
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from typing import Dict, List, Tuple
import json
import copy

from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from dataset import MathDataset


class DPOTrainer:
    """DPO trainer for preference-based LLM training."""

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda",
        beta: float = 0.1  # KL penalty coefficient
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.beta = beta

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Policy model (the one we train)
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.policy_model.to(self.device)

        # Reference model (frozen, for KL regularization)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.ref_model.to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int,
        is_policy: bool = True
    ) -> torch.Tensor:
        """
        Compute log probabilities of the response tokens.
        Only considers tokens after the prompt.

        Args:
            is_policy: If True, compute with gradients (for policy model)
        """
        if is_policy:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        else:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        # Only consider response tokens (after prompt)
        response_mask = shift_mask.clone().float()
        response_mask[:, :prompt_length-1] = 0

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Mask and sum
        masked_log_probs = token_log_probs * response_mask
        total_log_prob = masked_log_probs.sum(dim=-1)

        return total_log_prob

    def compute_dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        chosen_prompt_len: int,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
        rejected_prompt_len: int
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute DPO loss for a preference pair.

        DPO Loss = -log(sigmoid(beta * (
            (log_pi(y_w|x) - log_ref(y_w|x)) -
            (log_pi(y_l|x) - log_ref(y_l|x))
        )))
        """
        # Policy log probs (with gradients)
        self.policy_model.train()
        policy_chosen_logp = self.get_log_probs(
            self.policy_model, chosen_ids, chosen_mask, chosen_prompt_len, is_policy=True
        )
        policy_rejected_logp = self.get_log_probs(
            self.policy_model, rejected_ids, rejected_mask, rejected_prompt_len, is_policy=True
        )

        # Reference log probs (no grad)
        ref_chosen_logp = self.get_log_probs(
            self.ref_model, chosen_ids, chosen_mask, chosen_prompt_len, is_policy=False
        )
        ref_rejected_logp = self.get_log_probs(
            self.ref_model, rejected_ids, rejected_mask, rejected_prompt_len, is_policy=False
        )

        # DPO loss
        chosen_rewards = self.beta * (policy_chosen_logp - ref_chosen_logp)
        rejected_rewards = self.beta * (policy_rejected_logp - ref_rejected_logp)

        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        # Metrics
        metrics = {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item()
        }

        return loss, metrics


def collate_dpo_pairs(batch: List[Tuple[Dict, Dict]]) -> Dict:
    """Collate function for DPO preference pairs."""
    chosen_list = [pair[0] for pair in batch]
    rejected_list = [pair[1] for pair in batch]

    # Find max lengths
    max_chosen_len = max(item["input_ids"].shape[0] for item in chosen_list)
    max_rejected_len = max(item["input_ids"].shape[0] for item in rejected_list)

    def pad_batch(items, max_len):
        input_ids = []
        attention_masks = []
        prompt_lengths = []

        for item in items:
            seq_len = item["input_ids"].shape[0]
            padding_len = max_len - seq_len

            padded_ids = F.pad(item["input_ids"], (0, padding_len), value=0)
            input_ids.append(padded_ids)

            padded_mask = F.pad(item["attention_mask"], (0, padding_len), value=0)
            attention_masks.append(padded_mask)

            prompt_lengths.append(item["prompt_length"])

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "prompt_lengths": prompt_lengths
        }

    return {
        "chosen": pad_batch(chosen_list, max_chosen_len),
        "rejected": pad_batch(rejected_list, max_rejected_len)
    }


def train_dpo(config: Config):
    """Main DPO training loop."""
    print(f"Starting DPO training...")
    print(f"Model: {config.model_name}")
    print(f"Beta (KL penalty): {config.beta}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize trainer
    trainer = DPOTrainer(
        model_name=config.model_name,
        device=config.device,
        beta=config.beta
    )
    print(f"Model loaded on {trainer.device}")

    # Load dataset with SAME random seed as Q-learning
    print("Loading GSM8K dataset (same as Q-learning)...")
    train_dataset = MathDataset(
        tokenizer=trainer.tokenizer,
        max_samples=config.max_train_samples,
        split="train",
        max_length=config.max_seq_length,
        random_seed=42  # Same seed as Q-learning for fair comparison
    )

    # Get preference pairs
    pairs = train_dataset.get_preference_pairs()
    print(f"Created {len(pairs)} preference pairs from {len(train_dataset)} examples")

    if len(pairs) == 0:
        print("ERROR: No preference pairs created. Need both correct and incorrect examples.")
        return None

    # Create dataloader
    train_loader = DataLoader(
        pairs,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_dpo_pairs
    )

    # Optimizer
    optimizer = AdamW(
        trainer.policy_model.parameters(),
        lr=config.learning_rate
    )

    # Training loop
    global_step = 0
    metrics_history = []

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        epoch_loss = 0

        pbar = tqdm(train_loader, desc="Training DPO")
        for batch in pbar:
            chosen = batch["chosen"]
            rejected = batch["rejected"]

            # Move to device
            chosen_ids = chosen["input_ids"].to(trainer.device)
            chosen_mask = chosen["attention_mask"].to(trainer.device)
            rejected_ids = rejected["input_ids"].to(trainer.device)
            rejected_mask = rejected["attention_mask"].to(trainer.device)

            # Use first prompt length (they should be same for a pair)
            chosen_prompt_len = chosen["prompt_lengths"][0]
            rejected_prompt_len = rejected["prompt_lengths"][0]

            # Compute loss
            loss, metrics = trainer.compute_dpo_loss(
                chosen_ids, chosen_mask, chosen_prompt_len,
                rejected_ids, rejected_mask, rejected_prompt_len
            )

            epoch_loss += metrics["loss"]
            global_step += 1

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.policy_model.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "margin": f"{metrics['reward_margin']:.2f}"
            })

            if global_step % config.log_interval == 0:
                metrics_history.append({"step": global_step, **metrics})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(config.output_dir, f"dpo_checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": trainer.policy_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config
        }, checkpoint_path)

    # Save final model
    torch.save(trainer.policy_model.state_dict(),
               os.path.join(config.output_dir, "dpo_final_model.pt"))

    with open(os.path.join(config.output_dir, "dpo_metrics.json"), "w") as f:
        json.dump(metrics_history, f)

    print(f"\nDPO Training complete! Model saved to {config.output_dir}")
    return trainer


if __name__ == "__main__":
    config = Config()
    train_dpo(config)

