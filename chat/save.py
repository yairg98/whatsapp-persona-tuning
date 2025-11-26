"""
Shared conversation saving functionality for chat modules.

Provides a unified format for saving both single and group chat conversations.
"""

import json
from pathlib import Path
from datetime import datetime

from config import CONVERSATIONS_DIR

# Ensure directory exists
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)


def save_conversation(
    messages: list,
    chat_type: str,
    persona: str | None = None,
    participants: list[str] | None = None,
    starter: str | None = None,
    filepath: Path | None = None
) -> Path:
    """
    Save a conversation to a JSON file with a unified format.
    
    Common fields for all conversation types:
    - type: "single_chat" or "group_chat"
    - timestamp: When the conversation was created
    - last_updated: When the file was last saved
    - total_messages: Number of messages in the conversation
    - messages: The conversation history
    
    Type-specific fields:
    - Single chat: persona (name of the persona being chatted with)
    - Group chat: participants (list of persona names)
    
    Args:
        messages: List of message dicts with role, content, and optionally name
        chat_type: Either "single_chat" or "group_chat"
        persona: For single chats, the persona name
        participants: For group chats, list of participant names
        starter: Deprecated, kept for backwards compatibility but not saved
        filepath: Optional existing filepath to overwrite (for auto-save)
        
    Returns:
        Path to the saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if filepath is None:
        # Generate filename based on chat type
        if chat_type == "single_chat" and persona:
            safe_name = persona.lower().replace(" ", "_")
            filename = f"{safe_name}_{timestamp}.json"
        else:
            filename = f"group_chat_{timestamp}.json"
        filepath = CONVERSATIONS_DIR / filename
    
    # Build conversation data with common fields
    conversation_data = {
        "type": chat_type,
        "timestamp": timestamp,
        "last_updated": datetime.now().isoformat(),
        "total_messages": len(messages),
        "messages": messages
    }
    
    # Add type-specific fields
    if chat_type == "single_chat":
        conversation_data["persona"] = persona
    elif chat_type == "group_chat":
        conversation_data["participants"] = participants or []
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    return filepath


def auto_save_and_notify(
    messages: list,
    chat_type: str,
    filepath: Path | None = None,
    **kwargs
) -> Path:
    """
    Save conversation and print confirmation message.
    
    Args:
        messages: List of message dicts
        chat_type: Either "single_chat" or "group_chat"
        filepath: Optional existing filepath to overwrite
        **kwargs: Additional arguments passed to save_conversation
                  (persona, participants, starter)
        
    Returns:
        Path to the saved file
    """
    filepath = save_conversation(messages, chat_type, filepath=filepath, **kwargs)
    print(f"\n✓ Auto-saved conversation to: {filepath}")
    return filepath

