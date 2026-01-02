"""
Hyperparameter Search Script for LLM Post-Training (Q-Learning & DPO)

This script runs multiple training experiments with different hyperparameter
configurations to find optimal settings. It does NOT modify any existing code.

Supports both training methods:
- Q-Learning (Double DQN): gamma, tau, reward_distribution, reward_decay
- DPO (Direct Preference Optimization): beta

Usage:
------
    # Run grid search for Q-Learning (default)
    python hyperparam_search.py --method qlearning

    # Run grid search for DPO
    python hyperparam_search.py --method dpo

    # Run both methods for comparison
    python hyperparam_search.py --method both

    # Quick test mode (fewer samples, fewer configs)
    python hyperparam_search.py --quick

    # Resume from existing results
    python hyperparam_search.py --resume

    # Only search specific hyperparameters
    python hyperparam_search.py --method qlearning --search learning_rate gamma tau
    python hyperparam_search.py --method dpo --search learning_rate beta
"""
import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Dict, List, Any
import torch

from config import Config
from train import train as train_qlearning
from train_dpo import DPOTrainer
from evaluate import evaluate_model
from dataset import MathDataset


@dataclass
class SearchConfig:
    """Configuration for hyperparameter search."""
    # Model
    model_name: str = "Qwen/Qwen3-0.6B"

    # Common search spaces
    learning_rates: tuple = (1e-6, 5e-6, 1e-5, 5e-5)
    batch_sizes: tuple = (8)

    # Q-Learning specific search spaces
    gammas: tuple = (0.9, 0.95, 0.99, 0.999)
    taus: tuple = (0.001, 0.005, 0.01)
    reward_distributions: tuple = ("exponential", "uniform")
    reward_decays: tuple = (0.8, 0.9, 0.95, 0.99, 0.999)

    # DPO specific search spaces
    betas: tuple = (0.05, 0.1, 0.2, 0.5, 1.0)

    # Training settings
    max_train_samples: int = 1000  # Smaller for faster search
    max_eval_samples: int = 10
    num_epochs: int = 1  # Single epoch for search

    # Output
    output_dir: str = "./hyperparam_search_results"
    

def get_search_space(
    search_config: SearchConfig,
    params_to_search: List[str],
    method: str = "qlearning"
) -> List[Dict]:
    """
    Generate all hyperparameter combinations to search.

    Args:
        search_config: SearchConfig with parameter ranges
        params_to_search: List of parameter names to vary (others use defaults)
        method: "qlearning" or "dpo"

    Returns:
        List of config dictionaries with 'method' key included
    """
    # Default values for each method
    qlearning_defaults = {
        "learning_rate": 1e-5,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 4,
        "reward_distribution": "exponential",
        "reward_decay": 0.9,
    }

    dpo_defaults = {
        "learning_rate": 1e-5,
        "beta": 0.1,
        "batch_size": 4,
    }

    param_values = {}

    if method == "qlearning":
        defaults = qlearning_defaults

        if "learning_rate" in params_to_search:
            param_values["learning_rate"] = search_config.learning_rates
        else:
            param_values["learning_rate"] = (defaults["learning_rate"],)

        if "gamma" in params_to_search:
            param_values["gamma"] = search_config.gammas
        else:
            param_values["gamma"] = (defaults["gamma"],)

        if "tau" in params_to_search:
            param_values["tau"] = search_config.taus
        else:
            param_values["tau"] = (defaults["tau"],)

        if "batch_size" in params_to_search:
            param_values["batch_size"] = search_config.batch_sizes
        else:
            param_values["batch_size"] = (defaults["batch_size"],)

        if "reward_distribution" in params_to_search:
            param_values["reward_distribution"] = search_config.reward_distributions
        else:
            param_values["reward_distribution"] = (defaults["reward_distribution"],)

        if "reward_decay" in params_to_search:
            param_values["reward_decay"] = search_config.reward_decays
        else:
            param_values["reward_decay"] = (defaults["reward_decay"],)

    else:  # DPO
        defaults = dpo_defaults

        if "learning_rate" in params_to_search:
            param_values["learning_rate"] = search_config.learning_rates
        else:
            param_values["learning_rate"] = (defaults["learning_rate"],)

        if "beta" in params_to_search:
            param_values["beta"] = search_config.betas
        else:
            param_values["beta"] = (defaults["beta"],)

        if "batch_size" in params_to_search:
            param_values["batch_size"] = search_config.batch_sizes
        else:
            param_values["batch_size"] = (defaults["batch_size"],)

    # Generate all combinations
    keys = list(param_values.keys())
    values = [param_values[k] for k in keys]

    configs = []
    for combo in product(*values):
        config_dict = dict(zip(keys, combo))
        config_dict["method"] = method
        configs.append(config_dict)

    return configs


def _train_dpo_with_beta(config: Config, beta: float):
    """
    Train DPO model with a specific beta value.
    Returns the DPOTrainer instance.
    """
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from tqdm import tqdm
    from train_dpo import collate_dpo_pairs

    print(f"Starting DPO training with beta={beta}...")
    print(f"Model: {config.model_name}")

    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize trainer with custom beta
    trainer = DPOTrainer(
        model_name=config.model_name,
        device=config.device,
        beta=beta
    )
    print(f"Model loaded on {trainer.device}")

    # Load dataset
    print("Loading GSM8K dataset...")
    train_dataset = MathDataset(
        tokenizer=trainer.tokenizer,
        max_samples=config.max_train_samples,
        split="train",
        max_length=config.max_seq_length,
        random_seed=42
    )

    pairs = train_dataset.get_preference_pairs()
    print(f"Created {len(pairs)} preference pairs")

    if len(pairs) == 0:
        raise ValueError("No preference pairs created")

    train_loader = DataLoader(
        pairs,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_dpo_pairs
    )

    optimizer = AdamW(
        trainer.policy_model.parameters(),
        lr=config.learning_rate
    )

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        pbar = tqdm(train_loader, desc="Training DPO")
        for batch in pbar:
            chosen = batch["chosen"]
            rejected = batch["rejected"]

            chosen_ids = chosen["input_ids"].to(trainer.device)
            chosen_mask = chosen["attention_mask"].to(trainer.device)
            rejected_ids = rejected["input_ids"].to(trainer.device)
            rejected_mask = rejected["attention_mask"].to(trainer.device)

            chosen_prompt_len = chosen["prompt_lengths"][0]
            rejected_prompt_len = rejected["prompt_lengths"][0]

            loss, metrics = trainer.compute_dpo_loss(
                chosen_ids, chosen_mask, chosen_prompt_len,
                rejected_ids, rejected_mask, rejected_prompt_len
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.policy_model.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix({"loss": f"{metrics['loss']:.4f}"})

    print("DPO Training complete!")
    return trainer


def _evaluate_dpo_model(trainer, dataset: MathDataset, num_samples: int) -> Dict:
    """
    Evaluate a DPO-trained model.
    """
    from evaluate import extract_answer_gsm8k
    from tqdm import tqdm

    correct = 0
    total = 0

    samples = [s for s in dataset.examples if s.is_correct]
    if num_samples:
        samples = samples[:num_samples]

    print(f"Evaluating on {len(samples)} problems...")
    trainer.policy_model.eval()

    for example in tqdm(samples):
        # Generate solution
        prompt = f"Solve the following math problem step by step. Show your reasoning clearly, then write your final numeric answer after ####\n\nQuestion: {example.question}\nSolution:"

        inputs = trainer.tokenizer(prompt, return_tensors="pt").to(trainer.device)

        with torch.no_grad():
            outputs = trainer.policy_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=trainer.tokenizer.pad_token_id,
            )

        generated = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip()

        extracted = extract_answer_gsm8k(response)
        if extracted == example.answer:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def run_single_experiment(
    config: Config,
    experiment_id: int,
    total_experiments: int,
    method: str = "qlearning",
    beta: float = 0.1
) -> Dict[str, Any]:
    """
    Run a single training + evaluation experiment.

    Args:
        config: Training configuration
        experiment_id: Current experiment number
        total_experiments: Total number of experiments
        method: "qlearning" or "dpo"
        beta: DPO beta parameter (only used for DPO)

    Returns:
        Dictionary with hyperparameters and results
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {experiment_id}/{total_experiments} [{method.upper()}]")
    print(f"{'='*60}")
    print(f"  method:              {method}")
    print(f"  learning_rate:       {config.learning_rate}")
    print(f"  batch_size:          {config.batch_size}")

    if method == "qlearning":
        print(f"  gamma:               {config.gamma}")
        print(f"  tau:                 {config.tau}")
        print(f"  reward_distribution: {config.reward_distribution}")
        print(f"  reward_decay:        {config.reward_decay}")
    else:  # DPO
        print(f"  beta:                {beta}")
    print(f"{'='*60}")

    start_time = time.time()

    # Build result dict based on method
    if method == "qlearning":
        result = {
            "experiment_id": experiment_id,
            "method": method,
            "hyperparameters": {
                "learning_rate": config.learning_rate,
                "gamma": config.gamma,
                "tau": config.tau,
                "batch_size": config.batch_size,
                "reward_distribution": config.reward_distribution,
                "reward_decay": config.reward_decay,
            },
            "model_name": config.model_name,
            "max_train_samples": config.max_train_samples,
            "num_epochs": config.num_epochs,
        }
    else:  # DPO
        result = {
            "experiment_id": experiment_id,
            "method": method,
            "hyperparameters": {
                "learning_rate": config.learning_rate,
                "beta": beta,
                "batch_size": config.batch_size,
            },
            "model_name": config.model_name,
            "max_train_samples": config.max_train_samples,
            "num_epochs": config.num_epochs,
        }

    try:
        if method == "qlearning":
            # Train Q-Learning
            model = train_qlearning(config)
            tokenizer = model.tokenizer
        else:
            # Train DPO - need to patch beta into the trainer
            trainer = _train_dpo_with_beta(config, beta)
            model = trainer
            tokenizer = trainer.tokenizer

        # Evaluate
        print("\nEvaluating trained model...")
        test_dataset = MathDataset(
            tokenizer=tokenizer,
            max_samples=config.max_eval_samples,
            split="test",
            max_length=config.max_seq_length
        )

        if method == "qlearning":
            eval_results = evaluate_model(model, test_dataset, num_samples=config.max_eval_samples)
        else:
            # For DPO, wrap the policy model for evaluation
            eval_results = _evaluate_dpo_model(trainer, test_dataset, config.max_eval_samples)

        result["accuracy"] = eval_results["accuracy"]
        result["correct"] = eval_results["correct"]
        result["total"] = eval_results["total"]
        result["status"] = "success"

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        print(f"❌ Experiment failed: {e}")
        traceback.print_exc()
        result["accuracy"] = 0.0
        result["status"] = "failed"
        result["error"] = str(e)

    result["duration_seconds"] = time.time() - start_time

    print(f"\n✓ Experiment {experiment_id} complete")
    print(f"  Accuracy: {result.get('accuracy', 0):.2%}")
    print(f"  Duration: {result['duration_seconds']:.1f}s")
    
    return result


def save_results(results: List[Dict], output_dir: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"search_results_{timestamp}.json")
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {filename}")
    return filename


def load_existing_results(output_dir: str) -> List[Dict]:
    """Load most recent results file for resuming."""
    if not os.path.exists(output_dir):
        return []
    
    files = [f for f in os.listdir(output_dir) if f.startswith("search_results_")]
    if not files:
        return []
    
    latest = sorted(files)[-1]
    filepath = os.path.join(output_dir, latest)
    
    with open(filepath, "r") as f:
        results = json.load(f)
    
    print(f"✓ Loaded {len(results)} existing results from {filepath}")
    return results


def print_summary(results: List[Dict]):
    """Print summary of hyperparameter search results."""
    successful = [r for r in results if r.get("status") == "success"]

    if not successful:
        print("\n❌ No successful experiments to summarize")
        return

    # Group by method
    qlearning_results = [r for r in successful if r.get("method") == "qlearning"]
    dpo_results = [r for r in successful if r.get("method") == "dpo"]

    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*70)
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"  - Q-Learning: {len(qlearning_results)}")
    print(f"  - DPO: {len(dpo_results)}")
    print(f"Failed: {len(results) - len(successful)}")

    # Print results for each method
    for method_name, method_results in [("Q-LEARNING", qlearning_results), ("DPO", dpo_results)]:
        if not method_results:
            continue

        sorted_results = sorted(method_results, key=lambda x: x["accuracy"], reverse=True)

        print("\n" + "-"*70)
        print(f"TOP 5 {method_name} CONFIGURATIONS")
        print("-"*70)

        for i, r in enumerate(sorted_results[:5], 1):
            hp = r["hyperparameters"]
            print(f"\n#{i}: Accuracy = {r['accuracy']:.2%}")
            print(f"    learning_rate: {hp['learning_rate']}")
            print(f"    batch_size: {hp['batch_size']}")

            if method_name == "Q-LEARNING":
                print(f"    gamma: {hp['gamma']}")
                print(f"    tau: {hp['tau']}")
                print(f"    reward_distribution: {hp['reward_distribution']}")
                print(f"    reward_decay: {hp['reward_decay']}")
            else:  # DPO
                print(f"    beta: {hp['beta']}")

        # Best configuration for this method
        best = sorted_results[0]
        hp = best["hyperparameters"]

        print(f"\n{'='*70}")
        print(f"BEST {method_name} CONFIGURATION")
        print(f"{'='*70}")

        if method_name == "Q-LEARNING":
            print(f"""
To train with optimal Q-Learning hyperparameters, run:

python main.py --mode train --method qlearning \\
    --model {best['model_name']} \\
    --batch_size {hp['batch_size']} \\
    --reward_method {hp['reward_distribution']} \\
    --max_samples 7000 \\
    --epochs 3

Note: Also update config.py with:
    learning_rate = {hp['learning_rate']}
    gamma = {hp['gamma']}
    tau = {hp['tau']}
    reward_decay = {hp['reward_decay']}
""")
        else:  # DPO
            print(f"""
To train with optimal DPO hyperparameters, run:

python main.py --mode train --method dpo \\
    --model {best['model_name']} \\
    --batch_size {hp['batch_size']} \\
    --max_samples 7000 \\
    --epochs 3

Note: Also update train_dpo.py DPOTrainer with:
    learning_rate = {hp['learning_rate']}
    beta = {hp['beta']}
""")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Search for LLM Post-Training (Q-Learning & DPO)"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Base model")
    parser.add_argument("--method", type=str, default="qlearning",
                       choices=["qlearning", "dpo", "both"],
                       help="Training method: qlearning, dpo, or both")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    parser.add_argument("--max_experiments", type=int, default=None, help="Max experiments per method")
    parser.add_argument("--search", nargs="+", default=None,
                       help="Parameters to search (default depends on method)")
    parser.add_argument("--output_dir", type=str, default="./hyperparam_search_results")

    args = parser.parse_args()

    # Set default search params based on method
    if args.search is None:
        if args.method == "dpo":
            args.search = ["learning_rate", "beta"]
        else:  # qlearning or both
            args.search = ["learning_rate", "gamma", "tau"]

    # Validate search params for method
    qlearning_params = {"learning_rate", "gamma", "tau", "batch_size", "reward_distribution", "reward_decay"}
    dpo_params = {"learning_rate", "beta", "batch_size"}

    for param in args.search:
        if args.method == "qlearning" and param not in qlearning_params:
            parser.error(f"Parameter '{param}' not valid for Q-Learning. Valid: {qlearning_params}")
        elif args.method == "dpo" and param not in dpo_params:
            parser.error(f"Parameter '{param}' not valid for DPO. Valid: {dpo_params}")

    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠ CUDA not available. Hyperparameter search will be very slow.")
        device = "cpu"
    else:
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
        device = "cuda"

    # Create search config
    if args.quick:
        search_config = SearchConfig(
            model_name=args.model,
            learning_rates=(1e-5, 5e-5),
            gammas=(0.95, 0.99),
            taus=(0.005,),
            batch_sizes=(4,),
            reward_distributions=("exponential",),
            reward_decays=(0.9,),
            betas=(0.1, 0.2),
            max_train_samples=100,
            max_eval_samples=20,
            num_epochs=1,
            output_dir=args.output_dir,
        )
    else:
        search_config = SearchConfig(
            model_name=args.model,
            output_dir=args.output_dir,
        )

    # Determine which methods to run
    methods_to_run = []
    if args.method == "both":
        methods_to_run = ["qlearning", "dpo"]
    else:
        methods_to_run = [args.method]

    # Generate search space for each method
    all_configs = []
    for method in methods_to_run:
        # Filter search params for this method
        if method == "qlearning":
            method_search = [p for p in args.search if p in qlearning_params]
            if not method_search:
                method_search = ["learning_rate", "gamma", "tau"]
        else:  # dpo
            method_search = [p for p in args.search if p in dpo_params]
            if not method_search:
                method_search = ["learning_rate", "beta"]

        configs = get_search_space(search_config, method_search, method=method)
        all_configs.extend(configs)
        print(f"\n📊 {method.upper()} search space: {len(configs)} configurations")
        print(f"   Searching: {', '.join(method_search)}")

    # Limit experiments if specified
    if args.max_experiments:
        all_configs = all_configs[:args.max_experiments]
        print(f"\n   Limited to {args.max_experiments} total experiments")

    # Load existing results if resuming
    results = []
    completed_configs = set()
    if args.resume:
        results = load_existing_results(search_config.output_dir)
        for r in results:
            hp = r["hyperparameters"]
            method = r.get("method", "qlearning")
            if method == "qlearning":
                key = (method, hp["learning_rate"], hp.get("gamma"), hp.get("tau"),
                       hp["batch_size"], hp.get("reward_distribution"), hp.get("reward_decay"))
            else:
                key = (method, hp["learning_rate"], hp.get("beta"), hp["batch_size"])
            completed_configs.add(key)

    # Run experiments
    total = len(all_configs)
    for i, params in enumerate(all_configs, 1):
        method = params["method"]

        # Build key for deduplication
        if method == "qlearning":
            key = (method, params["learning_rate"], params.get("gamma"), params.get("tau"),
                   params["batch_size"], params.get("reward_distribution"), params.get("reward_decay"))
        else:
            key = (method, params["learning_rate"], params.get("beta"), params["batch_size"])

        if key in completed_configs:
            print(f"\n⏭ Skipping experiment {i}/{total} (already completed)")
            continue

        # Create config for this experiment
        if method == "qlearning":
            config = Config(
                model_name=search_config.model_name,
                learning_rate=params["learning_rate"],
                gamma=params.get("gamma", 0.99),
                tau=params.get("tau", 0.005),
                batch_size=params["batch_size"],
                reward_distribution=params.get("reward_distribution", "exponential"),
                reward_decay=params.get("reward_decay", 0.9),
                max_train_samples=search_config.max_train_samples,
                max_eval_samples=search_config.max_eval_samples,
                num_epochs=search_config.num_epochs,
                device=device,
                output_dir=os.path.join(search_config.output_dir, f"exp_{i}_{method}"),
            )
            beta = 0.1  # Not used for qlearning
        else:  # DPO
            config = Config(
                model_name=search_config.model_name,
                learning_rate=params["learning_rate"],
                batch_size=params["batch_size"],
                max_train_samples=search_config.max_train_samples,
                max_eval_samples=search_config.max_eval_samples,
                num_epochs=search_config.num_epochs,
                device=device,
                output_dir=os.path.join(search_config.output_dir, f"exp_{i}_{method}"),
            )
            beta = params.get("beta", 0.1)

        # Run experiment
        result = run_single_experiment(config, i, total, method=method, beta=beta)
        results.append(result)

        # Save intermediate results
        save_results(results, search_config.output_dir)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()

