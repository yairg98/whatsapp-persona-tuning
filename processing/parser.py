"""
WhatsApp chat export parser.

Handles parsing of WhatsApp .txt export files into structured Message objects.
"""

import re
from datetime import datetime
from dataclasses import dataclass

# ============================================================================
# Constants
# ============================================================================

# Regex pattern to match WhatsApp timestamp line: [M/D/YY, H:MM:SS AM/PM]
TIMESTAMP_PATTERN = re.compile(r'^\[\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}:\d{2}\s*[AP]M\]')

# Regex pattern to extract timestamp, sender, and content from message line
MESSAGE_PATTERN = re.compile(
    r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}:\d{2}\s*[AP]M)\]\s*([^:]+):\s*(.*)'
)

# Special Unicode character that appears at start of some messages (RTL mark)
RTL_MARK = '\u200e'


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Message:
    timestamp: datetime
    sender: str
    content: str
    
    def __post_init__(self):
        # Clean up content - remove leading special characters
        self.content = self.content.strip()
        if self.content.startswith(RTL_MARK):
            self.content = self.content[1:].strip()


# ============================================================================
# Helper Functions
# ============================================================================

def parse_timestamp(date_str: str, time_str: str) -> datetime:
    """Parse date and time strings into datetime object."""
    # Handle 2-digit year (assume 2000s)
    parts = date_str.split('/')
    month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
    
    if year < 100:
        year += 2000 if year < 50 else 1900
    
    # Parse time: "10:04:26 PM"
    time_parts = time_str.replace(' ', '').upper()
    is_pm = 'PM' in time_parts
    time_clean = time_parts.replace('AM', '').replace('PM', '')
    hour, minute, second = map(int, time_clean.split(':'))
    
    if is_pm and hour != 12:
        hour += 12
    elif not is_pm and hour == 12:
        hour = 0
    
    return datetime(year, month, day, hour, minute, second)


# ============================================================================
# Step 1: Split file into raw message strings
# ============================================================================

def split_messages_by_timestamp(filepath: str) -> list[str]:
    """
    First pass: Split file into individual message strings.
    Each string starts with a timestamp line.
    
    Returns: List of raw message strings (multiline)
    """
    message_strings = []
    current_message = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip invisible Unicode characters (like left-to-right mark \u200e) from start of line
            # This is necessary because WhatsApp sometimes adds these before timestamps
            line_stripped = line.lstrip('\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069')
            
            # Check if this line starts a new message (check both original and stripped)
            if TIMESTAMP_PATTERN.match(line) or TIMESTAMP_PATTERN.match(line_stripped):
                # Save previous message if it exists
                if current_message:
                    message_strings.append(''.join(current_message))
                # Start new message (use stripped line to remove invisible chars)
                current_message = [line_stripped]
            else:
                # Continuation of current message
                current_message.append(line)
        
        # Don't forget the last message
        if current_message:
            message_strings.append(''.join(current_message))
    
    return message_strings


# ============================================================================
# Step 2: Parse message strings into Message objects
# ============================================================================

def parse_message_string(msg_string: str) -> Message | None:
    """
    Second pass: Parse a single message string into a Message object.
    
    Args:
        msg_string: Raw message string starting with timestamp
        
    Returns:
        Message object or None if parsing fails
    """
    lines = msg_string.rstrip().split('\n')
    if not lines:
        return None
    
    # Match the first line (timestamp line)
    match = MESSAGE_PATTERN.match(lines[0])
    if not match:
        return None
    
    date_str, time_str, sender, first_line_content = match.groups()
    
    # Parse timestamp
    timestamp = parse_timestamp(date_str, time_str)
    
    # Combine all content lines
    content_lines = [first_line_content] if first_line_content.strip() else []
    content_lines.extend(lines[1:])
    content = '\n'.join(content_lines).strip()
    
    # Clean up content
    if content.startswith(RTL_MARK):
        content = content[1:].strip()
    
    return Message(
        timestamp=timestamp,
        sender=sender.strip(),
        content=content
    )


def parse_messages_from_strings(message_strings: list[str]) -> list[Message]:
    """
    Parse a list of message strings into Message objects.
    """
    messages = []
    for msg_str in message_strings:
        msg = parse_message_string(msg_str)
        if msg:
            messages.append(msg)
    return messages

