"""
Evaluation script for Q-Learning LLM.
Tests the trained model on math problems.
"""
import torch
import re
from typing import Optional, Dict, List
from tqdm import tqdm

from config import Config
from dataset import MathDataset, extract_answer_gsm8k, SYSTEM_PROMPT
from q_network import QLearningLLM


def generate_solution(
    model: QLearningLLM,
    question: str,
    max_new_tokens: int = 1500,
    temperature: float = 0.7
) -> str:
    """
    Generate a solution using the Q-learning trained model.
    Uses the online network's Q-values (logits) for generation.
    """
    # Use same system prompt as training for consistent format
    prompt = f"{SYSTEM_PROMPT}Question: {question}\nSolution:"
    
    inputs = model.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(model.device)
    
    # Generate using the online network
    with torch.no_grad():
        outputs = model.online_network.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id
        )
    
    generated = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the solution part
    if "Solution:" in generated:
        generated = generated.split("Solution:")[1].strip()
    
    return generated


def evaluate_model(
    model: QLearningLLM,
    dataset: MathDataset,
    num_samples: Optional[int] = None
) -> Dict:
    """
    Evaluate model on math problems.
    Returns accuracy and sample predictions.
    """
    correct = 0
    total = 0
    predictions = []
    
    samples = dataset.examples
    if num_samples:
        samples = samples[:num_samples]
    
    # Only evaluate on correct examples (we want to see if model can reproduce)
    samples = [s for s in samples if s.is_correct]
    
    print(f"Evaluating on {len(samples)} problems...")
    
    for example in tqdm(samples):
        generated = generate_solution(model, example.question)

        extracted_answer = extract_answer_gsm8k(generated)
        
        is_correct = extracted_answer == example.answer
        if is_correct:
            correct += 1
        total += 1
        
        predictions.append({
            "question": example.question[:100] + "...",
            "expected": example.answer,
            "generated_answer": extracted_answer,
            "correct": is_correct
        })
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions[:10]  # Sample predictions
    }


def evaluate_checkpoint(
    config: Config,
    checkpoint_path: Optional[str] = None
):
    """
    Evaluate a trained model checkpoint.
    """
    print("=" * 60)
    print("LLM Evaluation")
    print("=" * 60)

    # Load model for tokenizer
    print("\n1. Loading model...")
    model = QLearningLLM(
        model_name=config.model_name,
        device=config.device
    )

    # Load test dataset
    print("\n2. Loading test dataset...")
    test_dataset = MathDataset(
        tokenizer=model.tokenizer,
        max_samples=config.max_eval_samples,
        split="test",
        max_length=config.max_seq_length
    )
    print(f"Loaded {len(test_dataset)} test examples")
    
    # Load trained model if checkpoint provided
    if checkpoint_path:
        print(f"\n3. Loading trained model from {checkpoint_path}...")
        trained_model = QLearningLLM(
            model_name=config.model_name,
            device=config.device
        )
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        # Handle both raw state_dict and wrapped checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            trained_model.online_network.load_state_dict(checkpoint["model_state_dict"])
        else:
            trained_model.online_network.load_state_dict(checkpoint)

        print("\n4. Evaluating trained model...")
        trained_results = evaluate_model(trained_model, test_dataset, num_samples=50)
        print(f"Trained model accuracy: {trained_results['accuracy']:.2%}")

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Trained model: {trained_results['correct']}/{trained_results['total']} = {trained_results['accuracy']:.2%}")

        # Show sample predictions
        print("\n" + "=" * 60)
        print("SAMPLE PREDICTIONS (Trained Model)")
        print("=" * 60)
        for i, pred in enumerate(trained_results["predictions"][:3]):
            print(f"\n{i+1}. {pred['question']}")
            print(f"   Expected: {pred['expected']}")
            print(f"   Got: {pred['generated_answer']}")
            print(f"   Correct: {'✓' if pred['correct'] else '✗'}")
    else:
        print("\nNo checkpoint provided.")
        print("Run training first, then evaluate with:")
        print("  python evaluate.py --checkpoint outputs/final_model.pt")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    args = parser.parse_args()
    
    config = Config()
    evaluate_checkpoint(config, args.checkpoint)

