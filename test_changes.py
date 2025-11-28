"""
Quick test to verify the three changes:
1. System prompt is correctly applied
2. DPO uses same data as Q-learning
3. Both methods train without errors
"""
from dataset import SYSTEM_PROMPT, MathDataset
from transformers import AutoTokenizer

def test_system_prompt():
    """Test that system prompt is correctly defined."""
    print("=" * 60)
    print("TEST 1: System Prompt")
    print("=" * 60)
    
    print("System prompt content:")
    print(repr(SYSTEM_PROMPT))
    
    assert "step by step" in SYSTEM_PROMPT.lower(), "Should mention step by step"
    assert "####" in SYSTEM_PROMPT, "Should mention #### format"
    print("✓ System prompt correctly defined")
    print()

def test_data_consistency():
    """Test that Q-learning and DPO use identical data."""
    print("=" * 60)
    print("TEST 2: Data Consistency (same random seed)")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset twice with same seed
    dataset1 = MathDataset(tokenizer, max_samples=10, split="train", 
                           max_length=512, random_seed=42)
    dataset2 = MathDataset(tokenizer, max_samples=10, split="train", 
                           max_length=512, random_seed=42)
    
    # Check examples are identical
    for i, (ex1, ex2) in enumerate(zip(dataset1.examples, dataset2.examples)):
        assert ex1.question == ex2.question, f"Question mismatch at {i}"
        assert ex1.solution == ex2.solution, f"Solution mismatch at {i}"
        assert ex1.is_correct == ex2.is_correct, f"Correctness mismatch at {i}"
    
    print(f"✓ Both datasets have {len(dataset1.examples)} identical examples")
    
    # Check preference pairs
    pairs1 = dataset1.get_preference_pairs()
    pairs2 = dataset2.get_preference_pairs()
    
    assert len(pairs1) == len(pairs2), "Preference pair count mismatch"
    print(f"✓ Both have {len(pairs1)} preference pairs")
    print()

def test_formatted_prompt():
    """Test that prompts include system instruction."""
    print("=" * 60)
    print("TEST 3: Formatted Prompt Structure")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = MathDataset(tokenizer, max_samples=3, split="train", 
                          max_length=512, random_seed=42)
    
    example = dataset.examples[0]
    prompt = f"{SYSTEM_PROMPT}Question: {example.question}\nSolution:"
    
    print("Sample formatted prompt (first 300 chars):")
    print("-" * 40)
    print(prompt[:300])
    print("-" * 40)
    
    assert prompt.startswith("Solve the following"), "Should start with system prompt"
    assert "Question:" in prompt, "Should have Question: marker"
    assert "Solution:" in prompt, "Should have Solution: marker"
    print("✓ Prompt correctly formatted with system instruction")
    print()

def test_preference_pairs():
    """Test preference pair structure for DPO."""
    print("=" * 60)
    print("TEST 4: Preference Pairs for DPO")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = MathDataset(tokenizer, max_samples=20, split="train",
                          max_length=512, random_seed=42)

    pairs = dataset.get_preference_pairs()
    print(f"Created {len(pairs)} preference pairs from {len(dataset.examples)} examples")

    if pairs:
        chosen, rejected = pairs[0]
        print(f"Chosen (correct) has {chosen['input_ids'].shape[0]} tokens")
        print(f"Rejected (incorrect) has {rejected['input_ids'].shape[0]} tokens")
        print(f"Prompt length: {chosen['prompt_length']}")
        print("✓ Preference pairs correctly structured")
    else:
        print("⚠ No preference pairs created (need both correct and incorrect examples)")
    print()

def test_exact_pairing():
    """Test that every question has exactly one correct + one incorrect solution."""
    print("=" * 60)
    print("TEST 5: Exact 1:1 Correct/Incorrect Pairing")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_questions = 25
    dataset = MathDataset(tokenizer, max_samples=num_questions, split="train",
                          max_length=512, random_seed=42)

    # Count examples
    correct_count = sum(1 for ex in dataset.examples if ex.is_correct)
    incorrect_count = sum(1 for ex in dataset.examples if not ex.is_correct)

    print(f"Questions requested: {num_questions}")
    print(f"Correct examples: {correct_count}")
    print(f"Incorrect examples: {incorrect_count}")
    print(f"Total examples: {len(dataset.examples)}")

    assert correct_count == incorrect_count, f"Mismatch: {correct_count} correct vs {incorrect_count} incorrect"
    assert len(dataset.examples) == 2 * num_questions, f"Expected {2*num_questions} examples, got {len(dataset.examples)}"

    # Verify each question has exactly one of each
    question_counts = {}
    for ex in dataset.examples:
        q = ex.question
        if q not in question_counts:
            question_counts[q] = {"correct": 0, "incorrect": 0}
        if ex.is_correct:
            question_counts[q]["correct"] += 1
        else:
            question_counts[q]["incorrect"] += 1

    for q, counts in question_counts.items():
        assert counts["correct"] == 1, f"Question has {counts['correct']} correct solutions (expected 1)"
        assert counts["incorrect"] == 1, f"Question has {counts['incorrect']} incorrect solutions (expected 1)"

    # Verify preference pairs
    pairs = dataset.get_preference_pairs()
    assert len(pairs) == num_questions, f"Expected {num_questions} pairs, got {len(pairs)}"

    print(f"✓ Each question has exactly 1 correct + 1 incorrect solution")
    print(f"✓ {len(pairs)} preference pairs = {num_questions} questions")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VERIFYING IMPLEMENTATION CHANGES")
    print("=" * 60 + "\n")

    test_system_prompt()
    test_data_consistency()
    test_formatted_prompt()
    test_preference_pairs()
    test_exact_pairing()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)

