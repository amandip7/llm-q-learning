"""
Dataset handling for Q-Learning LLM experiment.
Uses GSM8K (Grade School Math) for ablation study.
Generates both correct and incorrect solutions for off-policy training.
"""
import re
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from datasets import load_dataset
from transformers import PreTrainedTokenizer


# System prompt to instruct model on GSM8K output format
SYSTEM_PROMPT = """Solve the following math problem step by step. Show your reasoning clearly, then write your final numeric answer after ####

"""


@dataclass
class MathExample:
    """A single math problem with solution and correctness label."""
    question: str
    solution: str
    answer: str  # Final numeric answer
    is_correct: bool


def extract_answer_gsm8k(solution: str) -> Optional[str]:
    """Extract the final numeric answer from GSM8K format."""
    # GSM8K uses #### to mark final answer
    match = re.search(r'####\s*(-?[\d,]+)', solution)
    if match:
        return match.group(1).replace(',', '')
    return None


def create_incorrect_solution(question: str, correct_solution: str, correct_answer: str) -> str:
    """
    Create a plausibly incorrect solution by perturbing the correct one.
    This simulates off-policy data with wrong reasoning.
    """
    # Strategy 1: Modify a number in the solution
    numbers = re.findall(r'\b\d+\b', correct_solution)
    if numbers and len(numbers) > 1:
        # Pick a random number to modify (not the final answer)
        idx = random.randint(0, len(numbers) - 2)
        old_num = numbers[idx]
        # Perturb by a small factor
        new_num = str(int(old_num) + random.choice([-2, -1, 1, 2, 5, 10]))
        # Replace first occurrence
        incorrect = correct_solution.replace(old_num, new_num, 1)
        # Also change the final answer
        wrong_answer = str(int(correct_answer) + random.randint(-10, 10))
        incorrect = re.sub(r'####\s*(-?[\d,]+)', f'#### {wrong_answer}', incorrect)
        return incorrect
    
    # Fallback: just change the final answer
    wrong_answer = str(int(correct_answer) + random.randint(1, 10))
    return re.sub(r'####\s*(-?[\d,]+)', f'#### {wrong_answer}', correct_solution)


def load_gsm8k_data(
    tokenizer: PreTrainedTokenizer,
    max_samples: Optional[int] = None,
    split: str = "train",
    include_incorrect: bool = True,
    random_seed: int = 42
) -> List[MathExample]:
    """
    Load GSM8K dataset and prepare for Q-learning training.

    For training split: Creates exactly one correct + one incorrect solution per question.
    This ensures fair comparison between Q-learning and DPO:
    - Q-learning trains on all examples (both correct and incorrect)
    - DPO gets exactly one preference pair per question (correct=chosen, incorrect=rejected)

    Args:
        tokenizer: Tokenizer for the model
        max_samples: Maximum number of QUESTIONS to load (actual examples = 2x for train)
        split: Dataset split ("train" or "test")
        include_incorrect: Whether to generate incorrect solutions (always True for train)
        random_seed: Random seed for reproducibility (same data for Q-learning and DPO)
    """
    # Set random seed for reproducible incorrect solution generation
    random.seed(random_seed)

    dataset = load_dataset("gsm8k", "main", split=split)

    examples = []
    questions_processed = 0

    for i, item in enumerate(dataset):
        # max_samples refers to number of questions, not total examples
        if max_samples and questions_processed >= max_samples:
            break

        question = item["question"]
        solution = item["answer"]
        answer = extract_answer_gsm8k(solution)

        if answer is None:
            continue

        questions_processed += 1

        # Add correct example
        examples.append(MathExample(
            question=question,
            solution=solution,
            answer=answer,
            is_correct=True
        ))

        # For training split: ALWAYS add exactly one incorrect example per question
        # This ensures:
        # 1. Q-learning has balanced correct/incorrect data
        # 2. DPO has exactly one preference pair per question
        if include_incorrect and split == "train":
            incorrect_solution = create_incorrect_solution(question, solution, answer)
            examples.append(MathExample(
                question=question,
                solution=incorrect_solution,
                answer=answer,  # Keep original answer for verification
                is_correct=False
            ))

    # Shuffle but maintain reproducibility
    random.shuffle(examples)
    return examples


def format_for_training(example: MathExample, tokenizer: PreTrainedTokenizer, max_length: int = 512) -> Dict:
    """
    Format a math example for Q-learning training.
    Returns tokenized input with reward signal.
    """
    # Format with system prompt for GSM8K output format
    prompt = f"{SYSTEM_PROMPT}Question: {example.question}\nSolution:"
    full_text = f"{prompt} {example.solution}"

    # Tokenize
    prompt_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    full_tokens = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)

    return {
        "input_ids": full_tokens["input_ids"].squeeze(0),
        "attention_mask": full_tokens["attention_mask"].squeeze(0),
        "prompt_length": prompt_tokens["input_ids"].shape[1],
        "is_correct": example.is_correct,
        "question": example.question,
        "answer": example.answer
    }


class MathDataset:
    """Dataset wrapper for Q-learning and DPO training."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_samples: Optional[int] = None,
        split: str = "train",
        max_length: int = 512,
        random_seed: int = 42
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.random_seed = random_seed

        # Load raw examples with fixed random seed for reproducibility
        self.examples = load_gsm8k_data(
            tokenizer=tokenizer,
            max_samples=max_samples,
            split=split,
            include_incorrect=(split == "train"),
            random_seed=random_seed
        )

        # Pre-process all examples
        self.data = [
            format_for_training(ex, tokenizer, max_length)
            for ex in self.examples
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_preference_pairs(self) -> List[Tuple[Dict, Dict]]:
        """
        Get preference pairs for DPO training.
        Returns pairs of (chosen, rejected) where chosen is correct and rejected is incorrect.
        """
        # Group examples by question
        question_to_examples = {}
        for i, ex in enumerate(self.examples):
            q = ex.question
            if q not in question_to_examples:
                question_to_examples[q] = {"correct": [], "incorrect": []}
            if ex.is_correct:
                question_to_examples[q]["correct"].append(self.data[i])
            else:
                question_to_examples[q]["incorrect"].append(self.data[i])

        # Create pairs
        pairs = []
        for q, examples in question_to_examples.items():
            for correct in examples["correct"]:
                for incorrect in examples["incorrect"]:
                    pairs.append((correct, incorrect))

        return pairs

