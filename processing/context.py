"""
Context window building functionality.

Builds context windows from preceding messages, respecting both token and message count limits.
"""

from processing.parser import Message
from processing.tokenizer import count_tokens


def format_message_for_context(message: Message) -> str:
    """
    Format a message for inclusion in context.
    
    Args:
        message: Message object to format
        
    Returns:
        Formatted string: "[Sender]: [Content]"
    """
    return f"{message.sender}: {message.content}"


def build_context_window(
    messages: list[Message],
    target_idx: int,
    max_tokens: int,
    max_messages: int
) -> tuple[list[Message], int]:
    """
    Build a context window from preceding messages.
    
    Collects preceding messages up to either the token limit or message count limit,
    whichever comes first.
    
    Args:
        messages: List of all messages (should be sorted chronologically)
        target_idx: Index of the target message (the one we're building context for)
        max_tokens: Maximum tokens allowed for context window
        max_messages: Maximum number of preceding messages to include
        
    Returns:
        Tuple of (list of context messages, total tokens used)
    """
    if target_idx == 0:
        return [], 0
    
    context_messages = []
    tokens_used = 0
    
    # Start from the message just before the target and work backwards
    start_idx = max(0, target_idx - max_messages)
    
    for i in range(target_idx - 1, start_idx - 1, -1):
        if i < 0:
            break
            
        msg = messages[i]
        formatted_msg = format_message_for_context(msg)
        msg_tokens = count_tokens(formatted_msg)
        
        # Check if adding this message would exceed token limit
        # Note: We need to account for the newline that will be added between messages
        newline_tokens = count_tokens("\n") if context_messages else 0
        total_tokens_if_added = tokens_used + newline_tokens + msg_tokens
        
        if total_tokens_if_added > max_tokens:
            break
        
        # Add message to context (prepend since we're going backwards)
        context_messages.insert(0, msg)
        tokens_used = total_tokens_if_added
        
        # Check message count limit
        if len(context_messages) >= max_messages:
            break
    
    # Final count of tokens including newlines between messages (for accurate reporting)
    if context_messages:
        context_text = "\n".join(format_message_for_context(msg) for msg in context_messages)
        tokens_used = count_tokens(context_text)
    
    return context_messages, tokens_used

