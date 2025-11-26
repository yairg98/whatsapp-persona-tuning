"""
Formatting utilities for OpenAI fine-tuning API format.
"""

from processing.parser import Message
from processing.context import format_message_for_context


def format_openai_example(
    context_messages: list[Message],
    target_message: Message,
    sender_name: str,
    include_system_message: bool = True
) -> dict:
    """
    Format a training example in OpenAI fine-tuning format.
    
    Args:
        context_messages: List of preceding messages for context
        target_message: The target message from the sender (will be assistant response)
        sender_name: Name of the sender (for system message)
        include_system_message: Whether to include system message (default: True)
        
    Returns:
        Dictionary in OpenAI fine-tuning format with "messages" key
    """
    messages = []
    
    # Build system message (optional)
    if include_system_message:
        system_content = f"You are {sender_name}, responding in a chat conversation."
        messages.append({"role": "system", "content": system_content})
    
    # Build user content from context messages
    if context_messages:
        user_content = "\n".join(format_message_for_context(msg) for msg in context_messages)
    else:
        user_content = "(No previous context)"
    
    messages.append({"role": "user", "content": user_content})
    
    # Target message is the assistant response
    assistant_content = target_message.content
    messages.append({"role": "assistant", "content": assistant_content})
    
    return {"messages": messages}

