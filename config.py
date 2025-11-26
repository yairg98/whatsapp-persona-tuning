"""
Configuration for WhatsApp to OpenAI Fine-Tuning Pipeline

Adjust these parameters before running the pipeline to control dataset size and training behavior.
"""

from pathlib import Path

# ============================================================================
# Target Personas
# ============================================================================
# Names of people to create training datasets for (must match sender names in WhatsApp exports)
TARGET_SENDERS = [
    "Alice Smith",
    "Bob Johnson",
    "Carol Williams"
]

# ============================================================================
# File Paths
# ============================================================================
WHATSAPP_DIR = Path("data/whatsapp")
TRAINING_DIR = Path("data/training")
MODELS_DIR = Path("data/models")
CONVERSATIONS_DIR = Path("data/conversations")

# ============================================================================
# OpenAI Model Configuration
# ============================================================================
# Base model to fine-tune from
FINE_TUNE_BASE_MODEL = "gpt-4o-mini-2024-07-18"

# Cost per million tokens for training (USD) - used for cost estimation
TRAINING_COST_PER_MILLION_TOKENS = 5.00

# Token counting model (should match fine-tune base model for accuracy)
TOKEN_MODEL = "gpt-4o-mini"

# ============================================================================
# Fine-Tuning Hyperparameters
# ============================================================================
# Set to None to use OpenAI's auto-selected defaults (recommended for one-shot fine-tuning)
N_EPOCHS = None  # OpenAI auto-selects based on dataset size
BATCH_SIZE = None  # OpenAI auto-selects optimal batch size
LEARNING_RATE_MULTIPLIER = None  # OpenAI auto-selects optimal learning rate

# ============================================================================
# Chat Settings
# ============================================================================
# Temperature for chat completions (lower = more consistent, higher = more creative)
CHAT_TEMPERATURE = 0.5

# Maximum tokens for chat responses
CHAT_MAX_TOKENS = 500

# ============================================================================
# Context Window Settings
# ============================================================================
# Maximum tokens for context window (preceding messages)
MAX_CONTEXT_TOKENS = 250

# Maximum number of preceding messages to include in context
MAX_CONTEXT_MESSAGES = 5

# ============================================================================
# Data Filtering Options
# ============================================================================
# Whether to include system message in each training example
# If False, omits system message to save tokens (model learns style from assistant responses)
# If True, includes system message for explicit role definition (standard OpenAI format)
INCLUDE_SYSTEM_MESSAGE = False

# Minimum length (in characters) for assistant message to be included in training
# Set to 5 to filter out ultra-short messages like "k", "ok", "yeah"
# Set to 0 to include all messages
MIN_ASSISTANT_LENGTH = 0

# Filter out WhatsApp encryption notices and other boilerplate messages
FILTER_ENCRYPTION_NOTICES = True

# Filter out training examples where assistant content is only a media omission
# (e.g., "image omitted", "audio omitted") - these aren't useful for training
FILTER_MEDIA_ONLY_TARGETS = True

# Sampling rate to reduce dataset size while maintaining diversity
# e.g. - Set to 2 to include every 2nd message (50% reduction/cost-savings)
SAMPLE_RATE = 1

# Prevent sample overlap: if enabled, skips creating training examples for messages
# that were already included in the context of another training example
PREVENT_SAMPLE_OVERLAP = True


# ============================================================================
# Helper Functions
# ============================================================================
def get_training_filename(sender_name: str) -> str:
    """
    Generate training filename from sender name.
    
    Args:
        sender_name: Full name of sender
        
    Returns:
        Filename-safe training file name (e.g., "alice_smith_training.jsonl")
    """
    return f"{sender_name.lower().replace(' ', '_')}_training.jsonl"


def get_all_training_files() -> dict:
    """
    Generate PERSON_FILES mapping from TARGET_SENDERS.
    
    Returns:
        Dictionary mapping filename to person name
        e.g., {"alice_smith_training.jsonl": "Alice Smith", ...}
    """
    return {get_training_filename(name): name for name in TARGET_SENDERS}
