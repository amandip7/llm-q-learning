"""
Q-Learning for LLM Post-Training: Proof of Concept

This experiment implements a value-based (Q-learning) approach for LLM
post-training as an alternative to policy-based methods like PPO/GRPO.

Key Concepts:
-------------
1. States (s) = Token sequence prefix
2. Actions (a) = Next token from vocabulary
3. Q(s, a) = Model logits interpreted as Q-values
4. s' = s + a (next state is prefix concatenated with action)

Advantages over Policy Methods (PPO/GRPO):
-----------------------------------------
- Off-policy: Can train on ANY data, not just self-generated rollouts
- No importance sampling needed
- Can leverage existing datasets of correct/incorrect solutions
- More sample efficient for problems with verifiable rewards

Implementation Details:
-----------------------
- Double DQN with soft target updates
- Token-level reward assignment (RLVR approach)
- GSM8K dataset for math problem ablation study
- DPO alternative for ablation comparison

Usage:
------
    # Train with Q-learning (default)
    python main.py --mode train --method qlearning

    # Train with DPO
    python main.py --mode train --method dpo

    # Evaluate the model
    python main.py --mode eval --checkpoint outputs/final_model.pt

    # Run both training and eval
    python main.py --mode both --method qlearning
"""
import argparse
import torch

from config import Config
from train import train as train_qlearning
from train_dpo import train_dpo
from evaluate import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Q-Learning for LLM Post-Training"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train",
        choices=["train", "eval", "both"],
        help="Mode: train, eval, or both"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model name (default: gpt2)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--reward_method",
        type=str,
        default="exponential",
        choices=["uniform", "exponential", "heuristic"],
        help="Token-level reward distribution method"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=7000,
        help="Maximum training samples (for quick POC testing)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="qlearning",
        choices=["qlearning", "dpo"],
        help="Training method: qlearning (Double DQN) or dpo (Direct Preference Optimization)"
    )

    args = parser.parse_args()

    # Create config with CLI overrides
    config = Config(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        reward_distribution=args.reward_method,
        max_train_samples=args.max_samples
    )

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU (training will be slow)")
        config.device = "cpu"
    else:
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
        config.device = "cuda"

    method_name = "Q-LEARNING (Double DQN)" if args.method == "qlearning" else "DPO (Direct Preference Optimization)"

    print("\n" + "=" * 60)
    print(f"LLM POST-TRAINING: {method_name}")
    print("=" * 60)
    print(f"Model:            {config.model_name}")
    print(f"Method:           {args.method}")
    if args.method == "qlearning":
        print(f"Reward method:    {config.reward_distribution}")
        print(f"Gamma (discount): {config.gamma}")
        print(f"Tau (soft update):{config.tau}")
    else:
        print(f"Beta (KL penalty): 0.1")
    print(f"Batch size:       {config.batch_size}")
    print(f"Learning rate:    {config.learning_rate}")
    print(f"Max samples:      {config.max_train_samples}")
    print("=" * 60 + "\n")

    if args.mode in ["train", "both"]:
        print(f"Starting {args.method.upper()} training...")

        if args.method == "qlearning":
            model = train_qlearning(config)
            checkpoint_name = "final_model.pt"
        else:
            model = train_dpo(config)
            checkpoint_name = "dpo_final_model.pt"

        # Set checkpoint for evaluation if doing both
        if args.mode == "both":
            args.checkpoint = f"{config.output_dir}/{checkpoint_name}"

    if args.mode in ["eval", "both"]:
        evaluate_checkpoint(config, args.checkpoint)


if __name__ == "__main__":
    main()

