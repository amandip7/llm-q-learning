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


def _introduce_arithmetic_error(solution: str) -> Tuple[str, bool]:
    """
    Introduce arithmetic errors in calculations.
    Returns (modified_solution, was_modified).
    """
    # Find patterns like "X + Y = Z" or "X * Y = Z" and corrupt the result
    patterns = [
        (r'(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)', lambda m: f"{m.group(1)} + {m.group(2)} = {int(m.group(3)) + random.choice([-1, 1, 2, -2])}"),
        (r'(\d+)\s*-\s*(\d+)\s*=\s*(\d+)', lambda m: f"{m.group(1)} - {m.group(2)} = {int(m.group(3)) + random.choice([-1, 1, 2, -2])}"),
        (r'(\d+)\s*\*\s*(\d+)\s*=\s*(\d+)', lambda m: f"{m.group(1)} * {m.group(2)} = {int(int(m.group(3)) * random.choice([0.9, 1.1]))}"),
        (r'(\d+)\s*×\s*(\d+)\s*=\s*(\d+)', lambda m: f"{m.group(1)} × {m.group(2)} = {int(int(m.group(3)) * random.choice([0.9, 1.1]))}"),
        (r'(\d+)\s*/\s*(\d+)\s*=\s*(\d+)', lambda m: f"{m.group(1)} / {m.group(2)} = {int(m.group(3)) + random.choice([-1, 1])}"),
    ]

    for pattern, replacer in patterns:
        matches = list(re.finditer(pattern, solution))
        if matches:
            match = random.choice(matches)
            new_calc = replacer(match)
            return solution[:match.start()] + new_calc + solution[match.end():], True

    return solution, False


def _introduce_logical_error(question: str, solution: str) -> Tuple[str, bool]:
    """
    Introduce logical/reasoning errors by swapping operations or misinterpreting.
    Returns (modified_solution, was_modified).
    """
    modifications = []

    # Swap addition with subtraction
    if ' + ' in solution and random.random() < 0.5:
        parts = solution.split(' + ', 1)
        if len(parts) == 2:
            modifications.append((' + ', ' - '))

    # Swap multiplication with division
    if ' * ' in solution or ' × ' in solution:
        if random.random() < 0.5:
            modifications.append((' * ', ' / '))
            modifications.append((' × ', ' / '))

    # Swap "each" interpretation (common GSM8K error)
    if 'each' in question.lower():
        if 'divided by' in solution.lower():
            modifications.append(('divided by', 'multiplied by'))
        elif 'multiplied by' in solution.lower():
            modifications.append(('multiplied by', 'divided by'))

    if modifications:
        old, new = random.choice(modifications)
        if old in solution:
            return solution.replace(old, new, 1), True

    return solution, False


def _introduce_thought_divergence(solution: str) -> Tuple[str, bool]:
    """
    Make the solution diverge partway through with different reasoning.
    Returns (modified_solution, was_modified).
    """
    # Split solution into lines/steps
    lines = solution.split('\n')
    if len(lines) < 3:
        return solution, False

    # Find a point to diverge (not first or last line)
    diverge_point = random.randint(1, max(1, len(lines) - 2))

    # Keep lines up to diverge point
    kept_lines = lines[:diverge_point]

    # Generate divergent reasoning
    divergent_phrases = [
        "Wait, let me reconsider. Actually, we should",
        "On second thought, the correct approach is to",
        "I realize I need to account for",
        "Let me try a different method:",
        "Actually, this is simpler than I thought.",
    ]

    # Create nonsensical but plausible-looking continuation
    wrong_calculations = [
        f"So we have {random.randint(10, 100)} items at ${random.randint(1, 50)} each.",
        f"This gives us {random.randint(50, 500)} total.",
        f"After {random.choice(['adding', 'subtracting', 'dividing by'])} {random.randint(2, 20)}, we get the answer.",
        f"The {random.choice(['total', 'remaining', 'final amount'])} is {random.randint(10, 200)}.",
    ]

    kept_lines.append(random.choice(divergent_phrases))
    kept_lines.extend(random.sample(wrong_calculations, min(2, len(wrong_calculations))))

    return '\n'.join(kept_lines), True


def _perturb_numbers_throughout(solution: str, correct_answer: str) -> str:
    """
    Perturb multiple numbers throughout the solution, not just one.
    Creates more substantial divergence from correct solution.
    """
    # Find all numbers in solution (excluding the final #### answer)
    answer_match = re.search(r'####\s*(-?[\d,]+)', solution)
    if answer_match:
        solution_without_answer = solution[:answer_match.start()]
    else:
        solution_without_answer = solution

    numbers = list(re.finditer(r'\b(\d+)\b', solution_without_answer))

    if len(numbers) >= 2:
        # Perturb 30-50% of numbers
        num_to_perturb = max(1, len(numbers) // 3)
        indices_to_perturb = random.sample(range(len(numbers)), min(num_to_perturb, len(numbers)))

        # Sort in reverse order to replace from end to start (preserves indices)
        indices_to_perturb.sort(reverse=True)

        for idx in indices_to_perturb:
            match = numbers[idx]
            old_num = int(match.group(1))
            # More aggressive perturbation
            perturbation = random.choice([
                old_num + random.randint(-5, 5),
                int(old_num * random.choice([0.5, 0.8, 1.2, 1.5, 2])),
                old_num + random.randint(1, 20),
                max(1, old_num - random.randint(1, 10)),
            ])
            new_num = str(max(0, perturbation))
            solution_without_answer = (
                solution_without_answer[:match.start()] +
                new_num +
                solution_without_answer[match.end():]
            )

    # Generate wrong final answer
    try:
        wrong_answer = str(int(correct_answer) + random.choice([-50, -20, -10, 10, 20, 50]))
    except ValueError:
        wrong_answer = str(random.randint(1, 100))

    return f"{solution_without_answer}\n#### {wrong_answer}"


def create_incorrect_solution(question: str, correct_solution: str, correct_answer: str) -> str:
    """
    Create a plausibly incorrect solution with diverse error types.
    This simulates off-policy data with various failure modes:

    1. Mathematical/Calculation Errors: Arithmetic mistakes in intermediate steps
    2. Logical/Reasoning Errors: Wrong operations or misinterpretation
    3. Thought Divergence: Solution path diverges partway through
    4. Format Errors: Missing #### or wrong structure (10% chance)
    5. Number Perturbation: Multiple numbers changed throughout

    Returns a solution string in GSM8K format with #### [wrong_answer] at the end.
    (Note: 50% of samples will also have format errors added at the end)
    """
    # Choose primary error type (weighted by realism)
    error_weights = [
        ('arithmetic', 0.25),      # Calculation errors
        ('logical', 0.20),         # Reasoning errors
        ('divergence', 0.20),      # Path divergence
        ('perturb_many', 0.35),    # Multiple number changes
    ]

    rand_val = random.random()
    cumulative = 0
    error_type = 'perturb_many'  # default
    for etype, weight in error_weights:
        cumulative += weight
        if rand_val < cumulative:
            error_type = etype
            break

    modified_solution = correct_solution
    was_modified = False

    if error_type == 'arithmetic':
        modified_solution, was_modified = _introduce_arithmetic_error(correct_solution)
    elif error_type == 'logical':
        modified_solution, was_modified = _introduce_logical_error(question, correct_solution)
    elif error_type == 'divergence':
        modified_solution, was_modified = _introduce_thought_divergence(correct_solution)

    # If primary error didn't work, fall back to number perturbation
    if not was_modified or error_type == 'perturb_many':
        modified_solution = _perturb_numbers_throughout(correct_solution, correct_answer)
    else:
        # Still need to change the final answer for other error types
        try:
            wrong_answer = str(int(correct_answer) + random.choice([-30, -15, -5, 5, 15, 30]))
            if wrong_answer == correct_answer:
                wrong_answer = str(int(correct_answer) + 10)
        except ValueError:
            wrong_answer = str(random.randint(1, 100))
        modified_solution = re.sub(r'####\s*(-?[\d,]+)', f'#### {wrong_answer}', modified_solution)

    # Ensure we have a valid format before potentially corrupting it
    if '####' not in modified_solution:
        try:
            wrong_answer = str(int(correct_answer) + random.randint(-50, 50))
        except ValueError:
            wrong_answer = str(random.randint(1, 100))
        modified_solution = f"{modified_solution}\n#### {wrong_answer}"

    # 50% chance to also add format error (misplace or remove ####)
    if random.random() < 0.50:
        modified_solution = _corrupt_answer_format(modified_solution)

    return modified_solution


def _corrupt_answer_format(solution: str) -> str:
    """
    Corrupt the #### format by moving it or removing it.
    Assumes solution already has #### somewhere.
    """
    # Extract the answer number
    match = re.search(r'####\s*(-?[\d,]+)', solution)
    if not match:
        return solution

    answer = match.group(1)
    # Remove the existing #### answer
    base_solution = re.sub(r'\n?####\s*(-?[\d,]+)', '', solution).strip()

    error_type = random.choice(['start', 'middle', 'remove', 'no_hash', 'text_after'])

    if error_type == 'start':
        # Put #### at the very start
        return f"#### {answer}\n{base_solution}"
    elif error_type == 'middle':
        # Put #### somewhere in the middle
        lines = base_solution.split('\n')
        if len(lines) > 2:
            insert_pos = random.randint(1, len(lines) - 1)
            lines.insert(insert_pos, f"#### {answer}")
            return '\n'.join(lines)
        else:
            return f"{base_solution}\n#### {answer}"  # fallback to normal
    elif error_type == 'remove':
        # Remove #### entirely, just end with the solution
        return base_solution
    elif error_type == 'no_hash':
        # Put answer without #### marker
        return f"{base_solution}\nThe answer is {answer}"
    else:  # text_after
        # Put extra text after the answer number
        return f"{base_solution}\n#### {answer} dollars total"


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
        clean_solution = re.sub(r"<<.*?>>", "", solution)
        answer = extract_answer_gsm8k(solution)

        if answer is None:
            continue

        questions_processed += 1

        # Add correct example
        examples.append(MathExample(
            question=question,
            solution=clean_solution,
            answer=answer,
            is_correct=True
        ))

        # For training split: ALWAYS add exactly one incorrect example per question
        # This ensures:
        # 1. Q-learning has balanced correct/incorrect data
        # 2. DPO has exactly one preference pair per question
        if include_incorrect and split == "train":
            incorrect_solution = create_incorrect_solution(question, solution, answer)
            clean_incorrect_solution = re.sub(r"<<.*?>>", "", incorrect_solution)
            examples.append(MathExample(
                question=question,
                solution=clean_incorrect_solution,
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

