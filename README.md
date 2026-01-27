# Q-Learning for LLM Post-Training

A proof-of-concept implementation comparing **Q-Learning (Double DQN)** vs **DPO (Direct Preference Optimization)** for LLM post-training on math reasoning tasks.

## Overview

This research project explores value-based reinforcement learning as an alternative to policy-based methods (PPO/GRPO) for LLM post-training. Instead of treating the LLM as a policy that outputs action probabilities, we interpret model logits as Q-values representing the expected cumulative reward for each token choice.

### Key Concepts

| Concept | RL Interpretation | LLM Context |
|---------|-------------------|-------------|
| **State (s)** | Current environment state | Token sequence prefix |
| **Action (a)** | Action to take | Next token from vocabulary |
| **Q(s, a)** | Expected cumulative reward | Model logits → Q-value projection |
| **Next State (s')** | Resulting state | Prefix concatenated with chosen token |

### Advantages over Policy-Based Methods (PPO/GRPO)

- **Off-Policy Learning**: Can train on ANY data, not just self-generated rollouts
- **No Importance Sampling**: Eliminates the need for complex importance sampling corrections
- **Dataset Reuse**: Can leverage existing datasets of correct/incorrect solutions
- **Sample Efficiency**: More efficient for problems with verifiable rewards (math, code, etc.)

## Features

- ✅ **Double DQN** with soft target network updates
- ✅ **Learned Q-Value Head**: Separate projection layer (hidden_size → vocab_size) for Q-values
- ✅ **Token-Level Reward Assignment** using RLVR approach
- ✅ **Multiple Reward Distribution Methods**:
  - `uniform`: Equal reward for each generated token
  - `exponential`: Higher weight for tokens closer to the outcome
  - `heuristic`: Localized rewards based on calculation tokens
- ✅ **DPO Baseline** for ablation comparison
- ✅ **GSM8K Dataset** for math reasoning evaluation
- ✅ **Hyperparameter Search** functionality for both methods

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended for training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd llm_q_learning

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training with Q-Learning (Double DQN)

```bash
# Default training
python main.py --mode train --method qlearning

# With custom parameters
python main.py --mode train --method qlearning \
    --model Qwen/Qwen3-0.6B \
    --epochs 3 \
    --batch_size 8 \
    --reward_method exponential \
    --max_samples 7000
```

### Training with DPO (Baseline)

```bash
python main.py --mode train --method dpo \
    --model Qwen/Qwen3-0.6B \
    --epochs 3 \
    --batch_size 8
```

### Evaluation

```bash
# Evaluate a trained checkpoint
python main.py --mode eval --checkpoint outputs/final_model.pt

# Train and evaluate in one run
python main.py --mode both --method qlearning
```

### Hyperparameter Search

```bash
# Search Q-Learning hyperparameters
python hyperparam_search.py --method qlearning

# Search DPO hyperparameters
python hyperparam_search.py --method dpo

# Quick test mode
python hyperparam_search.py --quick

# Search specific parameters
python hyperparam_search.py --method qlearning --search learning_rate gamma tau
```

## Project Structure

```
llm_q_learning/
├── main.py              # Main entry point with CLI
├── config.py            # Configuration dataclass with hyperparameters
├── train.py             # Q-Learning (Double DQN) training loop
├── train_dpo.py         # DPO training implementation
├── q_network.py         # QLearningLLM model with Q-value head
├── dataset.py           # GSM8K dataset loading and preprocessing
├── reward.py            # Token-level reward computation (RLVR)
├── evaluate.py          # Model evaluation on test set
├── hyperparam_search.py # Grid search for optimal hyperparameters
├── requirements.txt     # Python dependencies
└── outputs/             # Saved models and metrics
```

## Configuration

Key hyperparameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.99 | Discount factor for future rewards |
| `tau` | 0.005 | Soft update rate for target network |
| `learning_rate` | 1e-5 | AdamW learning rate |
| `beta` | 0.1 | KL penalty coefficient (DPO only) |
| `reward_distribution` | "uniform" | Token reward distribution method |
| `reward_decay` | 0.9 | Decay factor for exponential rewards |
| `correct_reward` | 1.0 | Reward for correct solutions |
| `incorrect_reward` | -1.0 | Reward for incorrect solutions |

## Implementation Details

### Double DQN Architecture

The Q-Learning implementation uses a **Double DQN** architecture to reduce overestimation bias:

1. **Online Network**: LLM + Q-value head that is actively trained
2. **Target Network**: Frozen copy updated slowly via soft updates (τ = 0.005)

**Q-Value Head Architecture**:
- Linear projection: `hidden_size → vocab_size`
- Initialized with small random weights (std=0.01) for near-zero initial Q-values
- Separate from the LLM's language modeling head

**Training Loss** (TD Error):
```
Q_target(s, a) = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
Loss = MSE(Q_online(s, a), Q_target(s, a))
```

### Token-Level Reward Assignment

Following the RLVR (Reinforcement Learning with Verifiable Rewards) approach, sequence-level rewards are distributed to individual tokens:

- **Uniform**: `r_t = R_total / num_tokens`
- **Exponential**: Later tokens receive higher weight (outcome-focused)
- **Heuristic**: Higher rewards for calculation tokens (`=`, `+`, numbers, `####`)

### Dataset Generation

The GSM8K dataset is augmented with incorrect solutions generated through:
- Arithmetic errors in intermediate calculations
- Logical reasoning errors (swapping operations)
- Thought divergence (solution path changes midway)
- Number perturbation throughout the solution

This creates balanced correct/incorrect pairs for both Q-Learning and DPO training.

## Evaluation

Models are evaluated on the GSM8K test set by:
1. Generating solutions using the trained model
2. Extracting the final answer (after `####` marker)
3. Comparing with ground truth answers
4. Computing accuracy as `correct / total`

Example evaluation output:
```
RESULTS SUMMARY
================
Trained model: 25/50 = 50.00%

EXAMPLE PREDICTIONS
==================
📝 QUESTION: Janet's ducks lay 16 eggs per day...
🤖 GENERATED SOLUTION: Step 1: Calculate total eggs...
📊 ANSWER ANALYSIS:
   Extracted Answer: 64
   Ground Truth: 64
   Result: ✓ CORRECT
```

## Research Context

This implementation is designed for ablation studies comparing:
- **Q-Learning**: Value-based, off-policy, can reuse any trajectory data
- **DPO**: Policy-based, requires preference pairs, no RL infrastructure

Both methods use the **exact same dataset** (same random seed) for fair comparison.

