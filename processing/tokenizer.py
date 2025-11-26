"""
Token counting functionality using tiktoken for OpenAI models.
"""

import tiktoken


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in a text string using tiktoken.
    
    Args:
        text: Text string to count tokens for
        model: OpenAI model name to use for encoding (default: gpt-4o-mini)
        
    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))
