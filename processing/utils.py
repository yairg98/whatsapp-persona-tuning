"""
Utility functions for processing WhatsApp chat files.

Reuses parsing logic from parser.py and extends it with additional functionality.
"""

from pathlib import Path
from processing.parser import Message, split_messages_by_timestamp, parse_messages_from_strings


def load_all_messages(whatsapp_dir: Path, specific_files: list[Path] | None = None) -> list[Message]:
    """
    Load and parse WhatsApp chat files from a directory.
    
    Skips example files (*.example.txt) to avoid loading sample data.
    
    Args:
        whatsapp_dir: Path to directory containing .txt chat files
        specific_files: Optional list of specific file paths to load.
                        If None, loads all .txt files from whatsapp_dir.
        
    Returns:
        List of all Message objects from all files
    """
    all_messages = []
    
    if specific_files is not None:
        # Load only the specified files
        txt_files = []
        for file_path in specific_files:
            # If path is relative or just a filename, resolve it relative to whatsapp_dir
            if not file_path.is_absolute():
                resolved_path = whatsapp_dir / file_path
            else:
                resolved_path = file_path
            
            # Ensure file exists
            if not resolved_path.exists():
                print(f"Warning: File not found: {resolved_path}")
                continue
            
            # Skip example files
            if ".example." in resolved_path.name:
                print(f"Warning: Skipping example file: {resolved_path.name}")
                continue
            
            txt_files.append(resolved_path)
    else:
        # Find all .txt files (skip example files)
        txt_files = [f for f in sorted(whatsapp_dir.glob("*.txt")) if ".example." not in f.name]
    
    if not txt_files:
        if specific_files:
            print("No valid files found from the specified list.")
        else:
            print("No .txt files found in the WhatsApp directory.")
        return []
    
    for txt_file in sorted(txt_files):
        raw_message_strings = split_messages_by_timestamp(str(txt_file))
        messages = parse_messages_from_strings(raw_message_strings)
        all_messages.extend(messages)
        print(f"Loaded {len(messages)} messages from {txt_file.name}")
    
    return all_messages


def extract_sender_messages(all_messages: list[Message], sender_name: str) -> list[Message]:
    """
    Extract all messages from a specific sender across all chat files.
    
    Args:
        all_messages: List of all Message objects from all files
        sender_name: Name of the sender to extract messages for
        
    Returns:
        List of Message objects from the specified sender, sorted chronologically
    """
    sender_messages = [msg for msg in all_messages if msg.sender.strip() == sender_name.strip()]
    # Sort by timestamp to maintain chronological order
    sender_messages.sort(key=lambda x: x.timestamp)
    return sender_messages


def get_all_unique_senders(all_messages: list[Message]) -> list[str]:
    """
    Get list of all unique senders in the chat data.
    
    Args:
        all_messages: List of all Message objects
        
    Returns:
        Sorted list of unique sender names
    """
    senders = set(msg.sender.strip() for msg in all_messages)
    return sorted(senders)

