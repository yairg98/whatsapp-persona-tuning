"""
Validate WhatsApp export files before processing.

This script checks WhatsApp chat exports for:
- Valid file format
- Parseable timestamps
- Detected senders
- Potential issues

Run this before generate.py to catch format problems early.

Usage:
    python -m processing.validate                    # Validate all files
    python -m processing.validate filename.txt       # Validate specific file
    python -m processing.validate file1.txt file2.txt # Validate multiple files
"""

import argparse
from pathlib import Path
from collections import Counter

from config import WHATSAPP_DIR, TARGET_SENDERS
from processing.parser import (
    split_messages_by_timestamp,
    parse_messages_from_strings,
    TIMESTAMP_PATTERN
)


def validate_file(filepath: Path) -> dict:
    """
    Validate a single WhatsApp export file.
    
    Args:
        filepath: Path to the .txt file
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "filename": filepath.name,
        "valid": True,
        "total_lines": 0,
        "message_count": 0,
        "parse_errors": 0,
        "senders": Counter(),
        "issues": []
    }
    
    try:
        # Count total lines
        with open(filepath, 'r', encoding='utf-8') as f:
            results["total_lines"] = sum(1 for _ in f)
        
        # Parse messages
        message_strings = split_messages_by_timestamp(str(filepath))
        messages = parse_messages_from_strings(message_strings)
        
        results["message_count"] = len(messages)
        results["parse_errors"] = len(message_strings) - len(messages)
        
        # Count senders
        for msg in messages:
            results["senders"][msg.sender.strip()] += 1
        
        # Check for issues
        if results["message_count"] == 0:
            results["valid"] = False
            results["issues"].append("No messages parsed - file may be empty or wrong format")
        
        if results["parse_errors"] > results["message_count"] * 0.1:
            results["issues"].append(f"High parse error rate: {results['parse_errors']} errors")
        
        # Check timestamp format in first few lines
        with open(filepath, 'r', encoding='utf-8') as f:
            first_lines = [f.readline() for _ in range(5)]
        
        timestamp_found = False
        for line in first_lines:
            line_stripped = line.lstrip('\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069')
            if TIMESTAMP_PATTERN.match(line) or TIMESTAMP_PATTERN.match(line_stripped):
                timestamp_found = True
                break
        
        if not timestamp_found:
            results["issues"].append("No timestamp found in first 5 lines - may not be WhatsApp format")
    
    except Exception as e:
        results["valid"] = False
        results["issues"].append(f"Error reading file: {str(e)}")
    
    return results


def main():
    """Validate WhatsApp files in the data directory or specific files."""
    parser = argparse.ArgumentParser(
        description="Validate WhatsApp export files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m processing.validate                    # Validate all files
  python -m processing.validate filename.txt       # Validate specific file
  python -m processing.validate file1.txt file2.txt # Validate multiple files
        """
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional: Specific .txt files to validate (relative to WhatsApp directory or absolute paths)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("WhatsApp Export Validation")
    print("=" * 70)
    print()
    
    # Determine which files to validate
    if args.files:
        # Validate specific files
        txt_files = []
        for file_arg in args.files:
            # If path is relative or just a filename, resolve it relative to whatsapp_dir
            file_path = Path(file_arg)
            if not file_path.is_absolute():
                resolved_path = WHATSAPP_DIR / file_path
            else:
                resolved_path = file_path
            
            # Ensure file exists
            if not resolved_path.exists():
                print(f"Error: File not found: {resolved_path}")
                continue
            
            # Skip example files
            if ".example." in resolved_path.name:
                print(f"Warning: Skipping example file: {resolved_path.name}")
                continue
            
            txt_files.append(resolved_path)
        
        if not txt_files:
            print("No valid files found from the specified list.")
            return
        
        print(f"Validating {len(txt_files)} specified file(s)")
    else:
        # Find all .txt files (skip example files)
        print(f"Checking files in: {WHATSAPP_DIR}")
        txt_files = [f for f in WHATSAPP_DIR.glob("*.txt") if ".example." not in f.name]
        
        if not txt_files:
            print("No .txt files found in the WhatsApp directory.")
            print(f"Please add WhatsApp exports to: {WHATSAPP_DIR}")
            print("(Note: Example files with .example.txt extension are skipped)")
            return
        
        print(f"Found {len(txt_files)} file(s)")
    
    print()
    
    all_senders = Counter()
    valid_count = 0
    total_messages = 0
    
    for filepath in sorted(txt_files):
        results = validate_file(filepath)
        
        # Update totals
        all_senders.update(results["senders"])
        total_messages += results["message_count"]
        if results["valid"] and not results["issues"]:
            valid_count += 1
        
        # Print results
        status = "✓" if results["valid"] and not results["issues"] else "⚠"
        print(f"{status} {results['filename']}")
        print(f"    Messages: {results['message_count']:,}")
        print(f"    Unique senders: {len(results['senders'])}")
        
        if results["parse_errors"] > 0:
            print(f"    Parse errors: {results['parse_errors']}")
        
        if results["issues"]:
            for issue in results["issues"]:
                print(f"    ⚠ {issue}")
        
        print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"Files checked: {len(txt_files)}")
    print(f"Valid files: {valid_count}")
    print(f"Total messages: {total_messages:,}")
    print(f"Unique senders: {len(all_senders)}")
    print()
    
    # Check target senders
    print("Target Senders Status:")
    print("-" * 40)
    for sender in TARGET_SENDERS:
        count = all_senders.get(sender, 0)
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {sender}: {count:,} messages")
    
    # Show top senders not in targets
    other_senders = {s: c for s, c in all_senders.items() if s not in TARGET_SENDERS}
    if other_senders:
        print()
        print("Other senders found (top 10):")
        print("-" * 40)
        for sender, count in sorted(other_senders.items(), key=lambda x: -x[1])[:10]:
            print(f"    {sender}: {count:,} messages")
    
    print()
    
    # Final status
    missing_targets = [s for s in TARGET_SENDERS if all_senders.get(s, 0) == 0]
    if missing_targets:
        print("⚠ Warning: Some target senders not found in chat files:")
        for sender in missing_targets:
            print(f"    - {sender}")
        print()
        print("Check that names in config.py match exactly as they appear in WhatsApp.")
    else:
        print("✓ All target senders found in chat files.")
    print()


if __name__ == "__main__":
    main()

