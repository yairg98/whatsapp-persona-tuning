"""
Estimate training costs before fine-tuning.

This script reads the training files and calculates the estimated cost
based on token counts and the configured cost per million tokens.

Usage:
    python -m training.estimate_cost
"""

import json
from pathlib import Path

from config import (
    TRAINING_DIR,
    TRAINING_COST_PER_MILLION_TOKENS,
    get_all_training_files
)
from processing.tokenizer import count_tokens


def count_file_tokens(filepath: Path) -> tuple:
    """
    Count tokens and examples in a training file.
    
    Args:
        filepath: Path to the JSONL training file
        
    Returns:
        Tuple of (total_tokens, num_examples, min_tokens, max_tokens)
    """
    total_tokens = 0
    num_examples = 0
    min_tokens = float('inf')
    max_tokens = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            example_json = json.dumps(example, ensure_ascii=False)
            tokens = count_tokens(example_json)
            
            total_tokens += tokens
            num_examples += 1
            min_tokens = min(min_tokens, tokens)
            max_tokens = max(max_tokens, tokens)
    
    if num_examples == 0:
        min_tokens = 0
    
    return total_tokens, num_examples, min_tokens, max_tokens


def main():
    """Estimate training costs for all personas."""
    print("=" * 70)
    print("Training Cost Estimation")
    print("=" * 70)
    print()
    
    person_files = get_all_training_files()
    
    if not person_files:
        print("No personas configured. Check TARGET_SENDERS in config.py")
        return
    
    print(f"Cost per million tokens: ${TRAINING_COST_PER_MILLION_TOKENS:.2f}")
    print(f"Training directory: {TRAINING_DIR}")
    print()
    
    total_tokens_all = 0
    total_examples_all = 0
    
    print(f"{'Persona':<25} {'Examples':>10} {'Tokens':>12} {'Avg Tok':>10} {'Est. Cost':>12}")
    print("-" * 70)
    
    for filename, person_name in person_files.items():
        filepath = TRAINING_DIR / filename
        
        if not filepath.exists():
            print(f"{person_name:<25} {'(file not found)':<45}")
            continue
        
        tokens, examples, min_tok, max_tok = count_file_tokens(filepath)
        avg_tokens = tokens / examples if examples > 0 else 0
        
        # Estimate cost (assumes ~2 epochs on average)
        estimated_epochs = 2
        estimated_cost = (tokens * estimated_epochs / 1_000_000) * TRAINING_COST_PER_MILLION_TOKENS
        
        print(f"{person_name:<25} {examples:>10,} {tokens:>12,} {avg_tokens:>10.1f} ${estimated_cost:>11.2f}")
        
        total_tokens_all += tokens
        total_examples_all += examples
    
    print("-" * 70)
    
    # Calculate totals
    avg_tokens_all = total_tokens_all / total_examples_all if total_examples_all > 0 else 0
    estimated_epochs = 2
    total_estimated_cost = (total_tokens_all * estimated_epochs / 1_000_000) * TRAINING_COST_PER_MILLION_TOKENS
    
    print(f"{'TOTAL':<25} {total_examples_all:>10,} {total_tokens_all:>12,} {avg_tokens_all:>10.1f} ${total_estimated_cost:>11.2f}")
    print()
    
    print("=" * 70)
    print("Cost Breakdown")
    print("=" * 70)
    print()
    print(f"Total training examples: {total_examples_all:,}")
    print(f"Total tokens (per epoch): {total_tokens_all:,}")
    print()
    print("Estimated costs by epoch count:")
    for epochs in [1, 2, 3]:
        cost = (total_tokens_all * epochs / 1_000_000) * TRAINING_COST_PER_MILLION_TOKENS
        print(f"  {epochs} epoch{'s' if epochs > 1 else ' '}: ${cost:.2f}")
    print()
    print("Note: OpenAI auto-selects epochs based on dataset size (typically 1-3).")
    print("Actual costs may vary. Check OpenAI's pricing page for current rates.")
    print()


if __name__ == "__main__":
    main()

