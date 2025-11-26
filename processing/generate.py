"""
Main script for processing WhatsApp chats into OpenAI fine-tuning datasets.

This script:
1. Loads all WhatsApp chat files
2. Extracts messages from each target sender
3. Builds context windows for each message
4. Formats as OpenAI fine-tuning JSONL
5. Writes separate training files for each person
6. Outputs statistics for cost evaluation

Usage:
    python -m processing.generate                    # Process all files
    python -m processing.generate filename.txt      # Process specific file
    python -m processing.generate file1.txt file2.txt # Process multiple files
"""

import argparse
import json
from pathlib import Path
from config import (
    TARGET_SENDERS,
    MAX_CONTEXT_TOKENS,
    MAX_CONTEXT_MESSAGES,
    INCLUDE_SYSTEM_MESSAGE,
    MIN_ASSISTANT_LENGTH,
    FILTER_ENCRYPTION_NOTICES,
    FILTER_MEDIA_ONLY_TARGETS,
    SAMPLE_RATE,
    PREVENT_SAMPLE_OVERLAP,
    WHATSAPP_DIR,
    TRAINING_DIR,
    TRAINING_COST_PER_MILLION_TOKENS
)
from processing.utils import load_all_messages, extract_sender_messages, get_all_unique_senders
from processing.context import build_context_window
from processing.formatter import format_openai_example
from processing.tokenizer import count_tokens
from processing.parser import Message


def get_filename_from_sender(sender_name: str) -> str:
    """
    Convert sender name to filename-safe format.
    
    Args:
        sender_name: Full name of sender
        
    Returns:
        Filename-safe string (lowercase with underscores)
    """
    return sender_name.lower().replace(" ", "_")


def process_sender(
    all_messages: list[Message],
    sender_name: str,
    output_dir: Path,
    max_context_tokens: int,
    max_context_messages: int
) -> dict:
    """
    Process messages for a single sender and generate training file.
    
    Args:
        all_messages: List of all messages from all chats
        sender_name: Name of sender to process
        output_dir: Directory to write output JSONL file
        max_context_tokens: Maximum tokens for context window
        max_context_messages: Maximum number of preceding messages
        
    Returns:
        Dictionary with statistics about the generated dataset
    """
    print(f"\n{'='*60}")
    print(f"Processing: {sender_name}")
    print(f"{'='*60}")
    
    # Extract sender's messages
    sender_messages = extract_sender_messages(all_messages, sender_name)
    
    if not sender_messages:
        print(f"Warning: No messages found for {sender_name}")
        return {
            "sender": sender_name,
            "total_messages": 0,
            "training_examples": 0,
            "total_tokens": 0,
            "avg_tokens_per_example": 0,
            "avg_context_messages": 0,
            "output_file": None
        }
    
    print(f"Found {len(sender_messages)} messages from {sender_name}")
    
    # Sort all messages chronologically for context building
    all_messages_sorted = sorted(all_messages, key=lambda x: x.timestamp)
    
    # Create a set of sender messages for fast lookup
    # Use a tuple of (timestamp, sender, content) as unique identifier
    sender_msg_ids = {
        (msg.timestamp, msg.sender.strip(), msg.content)
        for msg in sender_messages
    }
    
    # Generate training examples
    training_examples = []
    total_tokens = 0
    total_context_messages = 0
    valid_messages_seen = 0  # Count of valid messages after filtering
    filtered_count = 0
    encryption_filtered = 0
    length_filtered = 0
    media_only_filtered = 0
    overlap_skipped = 0
    
    # Track messages that have been used as context to prevent overlap
    messages_used_as_context = set()
    
    encryption_text = "Messages and calls are end-to-end encrypted"
    # Media omission patterns (case-insensitive, will check stripped content)
    media_only_patterns = [
        "image omitted", "audio omitted", "video omitted", "gif omitted",
        "document omitted", "sticker omitted", "location omitted",
        "contact omitted", "card omitted", "note omitted"
    ]
    
    for global_idx, msg in enumerate(all_messages_sorted):
        # Check if this is a message from our target sender
        msg_id = (msg.timestamp, msg.sender.strip(), msg.content)
        if msg_id not in sender_msg_ids:
            continue
        
        # Apply filters
        # Filter encryption notices
        if FILTER_ENCRYPTION_NOTICES and encryption_text in msg.content:
            encryption_filtered += 1
            filtered_count += 1
            continue
        
        # Filter short messages
        if len(msg.content) < MIN_ASSISTANT_LENGTH:
            length_filtered += 1
            filtered_count += 1
            continue
        
        # Filter media-only messages (where content is only "image omitted", etc.)
        if FILTER_MEDIA_ONLY_TARGETS:
            content_stripped = msg.content.strip()
            # Remove any leading invisible Unicode characters
            content_stripped = content_stripped.lstrip('\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069')
            if content_stripped.lower() in [p.lower() for p in media_only_patterns]:
                media_only_filtered += 1
                filtered_count += 1
                continue
        
        # Apply sampling (only include every Nth valid message)
        # Count valid messages (after filtering) for sampling
        if valid_messages_seen % SAMPLE_RATE != 0:
            valid_messages_seen += 1
            continue
        
        valid_messages_seen += 1
        
        # Check for overlap BEFORE building context: skip if target was already used
        if PREVENT_SAMPLE_OVERLAP and msg_id in messages_used_as_context:
            overlap_skipped += 1
            filtered_count += 1
            continue
        
        # This is a target message - build context window from all messages
        context_messages, context_tokens = build_context_window(
            all_messages_sorted,
            global_idx,
            max_context_tokens,
            max_context_messages
        )
        
        # Check for overlap: skip if any context messages were already used as targets/context
        if PREVENT_SAMPLE_OVERLAP:
            has_overlap = False
            for context_msg in context_messages:
                context_msg_id = (context_msg.timestamp, context_msg.sender.strip(), context_msg.content)
                if context_msg_id in messages_used_as_context:
                    has_overlap = True
                    break
            
            if has_overlap:
                overlap_skipped += 1
                filtered_count += 1
                continue
            
            # No overlap found - track all messages (context + target) as used
            for context_msg in context_messages:
                context_msg_id = (context_msg.timestamp, context_msg.sender.strip(), context_msg.content)
                messages_used_as_context.add(context_msg_id)
            # Also track the target message itself
            messages_used_as_context.add(msg_id)
        
        # Format as OpenAI example
        example = format_openai_example(context_messages, msg, sender_name, INCLUDE_SYSTEM_MESSAGE)
        
        # Count tokens in the full example (for accurate cost estimation)
        example_json = json.dumps(example, ensure_ascii=False)
        example_tokens = count_tokens(example_json)
        
        training_examples.append(example)
        total_tokens += example_tokens
        total_context_messages += len(context_messages)
        
        if len(training_examples) % 100 == 0:
            print(f"  Generated {len(training_examples)} training examples...")
    
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} messages:")
        if encryption_filtered > 0:
            print(f"    - Encryption notices: {encryption_filtered}")
        if length_filtered > 0:
            print(f"    - Short messages (<{MIN_ASSISTANT_LENGTH} chars): {length_filtered}")
        if media_only_filtered > 0:
            print(f"    - Media-only targets (image/audio omitted): {media_only_filtered}")
        if overlap_skipped > 0:
            print(f"    - Overlap prevention (used as context): {overlap_skipped}")
    
    # Write JSONL file
    filename = f"{get_filename_from_sender(sender_name)}_training.jsonl"
    output_path = output_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Wrote {len(training_examples)} training examples to {output_path}")
    
    # Calculate statistics
    stats = {
        "sender": sender_name,
        "total_messages": len(sender_messages),
        "training_examples": len(training_examples),
        "total_tokens": total_tokens,
        "avg_tokens_per_example": total_tokens / len(training_examples) if training_examples else 0,
        "avg_context_messages": total_context_messages / len(training_examples) if training_examples else 0,
        "output_file": str(output_path)
    }
    
    return stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Process WhatsApp chats into OpenAI fine-tuning datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m processing.generate                    # Process all files
  python -m processing.generate filename.txt      # Process specific file
  python -m processing.generate file1.txt file2.txt # Process multiple files
        """
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional: Specific .txt files to process (relative to WhatsApp directory or absolute paths)"
    )
    
    args = parser.parse_args()
    
    # Setup paths from config
    whatsapp_dir = WHATSAPP_DIR
    output_dir = TRAINING_DIR
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("WhatsApp to OpenAI Fine-Tuning Data Pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Max Context Tokens: {MAX_CONTEXT_TOKENS}")
    print(f"  Max Context Messages: {MAX_CONTEXT_MESSAGES}")
    print(f"  Include System Message: {INCLUDE_SYSTEM_MESSAGE}")
    print(f"  Min Assistant Length: {MIN_ASSISTANT_LENGTH} chars")
    print(f"  Filter Encryption Notices: {FILTER_ENCRYPTION_NOTICES}")
    print(f"  Filter Media-Only Targets: {FILTER_MEDIA_ONLY_TARGETS}")
    print(f"  Sample Rate: {SAMPLE_RATE} (every {SAMPLE_RATE} message)")
    print(f"  Prevent Sample Overlap: {PREVENT_SAMPLE_OVERLAP}")
    print(f"  Target Senders: {len(TARGET_SENDERS)}")
    print(f"  Output Directory: {output_dir}")
    print()
    
    # Determine which files to process
    specific_files = None
    if args.files:
        specific_files = [Path(f) for f in args.files]
        print(f"Processing {len(specific_files)} specified file(s)...")
    else:
        print("Loading all WhatsApp chat files...")
    
    # Load messages
    all_messages = load_all_messages(whatsapp_dir, specific_files=specific_files)
    print(f"\nTotal messages loaded: {len(all_messages)}")
    
    # Get all unique senders for reference
    unique_senders = get_all_unique_senders(all_messages)
    print(f"Unique senders found: {len(unique_senders)}")
    
    # Process each target sender
    all_stats = []
    for sender_name in TARGET_SENDERS:
        stats = process_sender(
            all_messages,
            sender_name,
            output_dir,
            MAX_CONTEXT_TOKENS,
            MAX_CONTEXT_MESSAGES
        )
        all_stats.append(stats)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}\n")
    
    total_tokens_all = 0
    total_examples_all = 0
    
    for stats in all_stats:
        print(f"{stats['sender']}:")
        print(f"  Messages: {stats['total_messages']}")
        print(f"  Training Examples: {stats['training_examples']}")
        print(f"  Total Tokens: {stats['total_tokens']:,}")
        print(f"  Avg Tokens/Example: {stats['avg_tokens_per_example']:.1f}")
        print(f"  Avg Context Messages: {stats['avg_context_messages']:.1f}")
        output_file = stats.get('output_file') or 'N/A (no file created)'
        print(f"  Output File: {output_file}")
        print()
        
        total_tokens_all += stats['total_tokens']
        total_examples_all += stats['training_examples']
    
    print(f"{'='*60}")
    print("TOTALS (All People Combined):")
    print(f"  Total Training Examples: {total_examples_all:,}")
    print(f"  Total Tokens: {total_tokens_all:,}")
    estimated_cost = (total_tokens_all / 1_000_000) * TRAINING_COST_PER_MILLION_TOKENS
    print(f"  Estimated Training Cost: ${estimated_cost:.2f}")
    print(f"    (at ${TRAINING_COST_PER_MILLION_TOKENS:.2f} per million tokens)")
    print(f"{'='*60}")
    
    print("\n✓ Pipeline complete! Training datasets are ready for upload to OpenAI.")


if __name__ == "__main__":
    main()

