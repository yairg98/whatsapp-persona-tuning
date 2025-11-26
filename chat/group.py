"""
Interactive group chat with multiple persona models.

Allows personas to have a conversation, with random selection of who speaks next
and how many messages they send (1-3). User can intervene at any point or let
the conversation flow naturally.

IMPORTANT: Each time a model responds, it receives the FULL conversation history
from the start, ensuring proper context and natural conversation flow.

Usage:
    python -m chat.group
"""

import os
import json
import random
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from config import MODELS_DIR, CHAT_TEMPERATURE, CHAT_MAX_TOKENS
from chat.prompts import get_group_chat_prompt
from chat.save import save_conversation, auto_save_and_notify

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. "
        "Please create a .env file from .env.example and add your API key."
    )

client = OpenAI(api_key=api_key)

MODELS_FILE = MODELS_DIR / "model_ids.json"

# Conversation starter prompts for lively discussion
CONVERSATION_STARTERS = [
    "What's the most overrated restaurant in the city and why?",
    "Is it weird to wear sunglasses indoors? Discuss.",
    "What's a hill you're willing to die on?",
    "If you could ban one thing from social media, what would it be and why?",
    "What's the best/worst date idea someone has suggested to you?",
    "What's something everyone pretends to like but actually hates?",
    "What's the most ridiculous argument you've gotten into recently?",
    "Is it acceptable to wear pajamas in public? Let's debate this.",
    "What's a movie/show everyone loves that you think is terrible?",
    "What's the worst piece of advice you've ever received?"
]


def load_model_ids():
    """Load model IDs from saved file."""
    if not MODELS_FILE.exists():
        print(f"Error: Model IDs file not found: {MODELS_FILE}")
        print("Please run 'python -m training.fine_tune' first to create the models.")
        return {}
    
    with open(MODELS_FILE, 'r') as f:
        return json.load(f)


def get_available_personas() -> list:
    """
    Load available personas from model_ids.json (source of truth).
    
    Returns:
        List of persona names that have fine-tuned models available
    """
    model_ids = load_model_ids()
    return list(model_ids.keys())


def format_group_chat_messages(conversation_history: list, current_persona: str) -> list:
    """
    Format conversation history for a specific persona, including system prompt.
    
    In a group chat, each persona sees all previous messages from all participants.
    We format this as a continuous conversation where each message includes the sender's name.
    This ensures the model has full context of the entire conversation and knows who said what.
    
    Args:
        conversation_history: List of messages in format [{"role": "user/assistant", "name": "Person Name", "content": "..."}, ...]
        current_persona: Name of the persona responding
    
    Returns:
        Formatted messages list for OpenAI API with full conversation context
    """
    system_prompt = get_group_chat_prompt(current_persona)
    messages = [{"role": "system", "content": system_prompt}]
    
    # Format conversation history as a group chat
    # Each message shows who said it, so the model understands the group context
    # This includes ALL messages from the start of the conversation
    conversation_text = []
    for msg in conversation_history:
        name = msg.get("name", "Someone")
        content = msg.get("content", "")
        # Format: "Name: message content"
        conversation_text.append(f"{name}: {content}")
    
    # Determine who spoke last to help the model understand context
    last_speaker = None
    if conversation_history:
        last_speaker = conversation_history[-1].get("name", "Someone")
    
    # Combine all messages into a single user message that represents the full group chat
    # This ensures the model sees the complete conversation context
    if conversation_text:
        full_conversation = "\n".join(conversation_text)
        
        # Build context-aware prompt
        if last_speaker == current_persona:
            # If the current persona just spoke, they should continue their thought or respond to others
            prompt = (
                f"Here is the full group chat conversation so far:\n\n{full_conversation}\n\n"
                f"You ({current_persona}) just sent a message. You can continue your thought, add to what you said, "
                f"or respond to what others said earlier. Do NOT ask yourself questions or respond to yourself as if you're a different person. "
                f"Remember: you are {current_persona} and you can see your own previous messages in the conversation above."
            )
        else:
            # If someone else just spoke, respond to them or the conversation
            prompt = (
                f"Here is the full group chat conversation so far:\n\n{full_conversation}\n\n"
                f"{last_speaker} just sent a message. What do you say next as {current_persona}? "
                f"Respond naturally to what others have said, building on the conversation. "
                f"You can respond to {last_speaker}'s message, respond to someone else, or add your own thoughts."
            )
        
        messages.append({
            "role": "user",
            "content": prompt
        })
    else:
        # If no history yet, just prompt for initial response
        messages.append({
            "role": "user",
            "content": f"Respond as {current_persona} to start the conversation."
        })
    
    return messages


def get_persona_response(persona_name: str, model_id: str, conversation_history: list) -> str:
    """
    Get a response from a persona based on the full conversation history.
    
    This function ensures that every API call includes the complete conversation
    context from the start, so the model can respond appropriately to the ongoing discussion.
    
    Args:
        persona_name: Name of the persona responding
        model_id: OpenAI model ID for this persona
        conversation_history: Complete list of all messages in the conversation so far
    
    Returns:
        Response text from the persona
    """
    # Format messages with full conversation context
    # This includes ALL previous messages from all participants
    messages = format_group_chat_messages(conversation_history, persona_name)
    
    # Make API call with full context
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=CHAT_TEMPERATURE,
        max_tokens=CHAT_MAX_TOKENS
    )
    
    return response.choices[0].message.content


def print_conversation_history(history: list):
    """Print conversation history in a readable format."""
    print("\n" + "="*80)
    print("CONVERSATION SO FAR:")
    print("="*80)
    for msg in history:
        name = msg.get("name", "You")
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "assistant":
            print(f"\n{name}: {content}")
        else:
            print(f"\nYou: {content}")
    print("="*80 + "\n")


def main():
    """Main group chat interface."""
    print("=" * 80)
    print("Group Chat with Personas")
    print("=" * 80)
    
    model_ids = load_model_ids()
    
    # Get available personas dynamically from model_ids.json
    available_personas = list(model_ids.keys())
    
    if not available_personas:
        print("Error: No fine-tuned models found.")
        print("Please run 'python -m training.fine_tune' first to create the models.")
        return
    
    print(f"\nAvailable personas: {', '.join(available_personas)}")
    
    # Select conversation starter
    print("\nSelect a conversation starter:")
    print("-" * 80)
    for i, starter in enumerate(CONVERSATION_STARTERS, 1):
        print(f"  {i}. {starter}")
    print(f"  {len(CONVERSATION_STARTERS) + 1}. Custom prompt")
    print()
    
    choice = input("Enter number (or press Enter for random): ").strip()
    
    if choice == "":
        starter = random.choice(CONVERSATION_STARTERS)
    elif choice.isdigit() and 1 <= int(choice) <= len(CONVERSATION_STARTERS):
        starter = CONVERSATION_STARTERS[int(choice) - 1]
    elif choice.isdigit() and int(choice) == len(CONVERSATION_STARTERS) + 1:
        starter = input("Enter your custom prompt: ").strip()
    else:
        starter = random.choice(CONVERSATION_STARTERS)
        print(f"Invalid choice. Using random starter: {starter}")
    
    print(f"\n{'='*80}")
    print(f"Conversation Starter: {starter}")
    print(f"{'='*80}\n")
    
    # Initialize conversation
    conversation_history = [
        {"role": "user", "name": "You", "content": starter}
    ]
    
    print(f"You: {starter}\n")
    
    turn_count = 0
    conversation_filepath = None  # Will be set on first save
    
    while True:
        # Get user choice for next action
        print("\n" + "-"*80)
        print("What would you like to do?")
        print("  1. Continue with random person (auto)")
        print("  2. Select next responder manually")
        print("  3. Add your own message")
        print("  4. View conversation history")
        print("  5. Save conversation (auto-saves after each turn)")
        print("  6. Save and exit")
        print("  7. Exit without saving")
        print("-"*80)
        
        choice = input("Enter choice (1-7): ").strip()
        
        if choice == "1":
            # Random person responds with 1-3 messages
            next_persona = random.choice(available_personas)
            num_messages = random.randint(1, 3)
            
            print(f"\n{next_persona} is responding ({num_messages} message{'s' if num_messages > 1 else ''})...")
            
            model_id = model_ids[next_persona]
            
            for i in range(num_messages):
                # Each API call includes full conversation history for proper context
                response = get_persona_response(next_persona, model_id, conversation_history)
                print(f"\n{next_persona}: {response}")
                
                conversation_history.append({
                    "role": "assistant",
                    "name": next_persona,
                    "content": response
                })
                turn_count += 1
            
            # Auto-save after each turn
            conversation_filepath = auto_save_and_notify(
                conversation_history, "group_chat",
                filepath=conversation_filepath,
                participants=available_personas,
                starter=starter
            )
        
        elif choice == "2":
            # Manual selection
            print("\nSelect next responder:")
            for i, persona in enumerate(available_personas, 1):
                print(f"  {i}. {persona}")
            
            persona_choice = input("Enter number: ").strip()
            
            if persona_choice.isdigit() and 1 <= int(persona_choice) <= len(available_personas):
                next_persona = available_personas[int(persona_choice) - 1]
                
                num_choice = input("How many messages? (1-3, or press Enter for random): ").strip()
                if num_choice.isdigit() and 1 <= int(num_choice) <= 3:
                    num_messages = int(num_choice)
                else:
                    num_messages = random.randint(1, 3)
                
                print(f"\n{next_persona} is responding ({num_messages} message{'s' if num_messages > 1 else ''})...")
                
                model_id = model_ids[next_persona]
                
                for i in range(num_messages):
                    response = get_persona_response(next_persona, model_id, conversation_history)
                    print(f"\n{next_persona}: {response}")
                    
                    conversation_history.append({
                        "role": "assistant",
                        "name": next_persona,
                        "content": response
                    })
                    turn_count += 1
                
                # Auto-save after each turn
                conversation_filepath = auto_save_and_notify(
                    conversation_history, "group_chat",
                    filepath=conversation_filepath,
                    participants=available_personas,
                    starter=starter
                )
            else:
                print("Invalid selection.")
        
        elif choice == "3":
            # User adds message
            user_message = input("\nYour message: ").strip()
            if user_message:
                conversation_history.append({
                    "role": "user",
                    "name": "You",
                    "content": user_message
                })
                print(f"\nYou: {user_message}")
                turn_count += 1
                
                # Auto-save after user message
                conversation_filepath = auto_save_and_notify(
                    conversation_history, "group_chat",
                    filepath=conversation_filepath,
                    participants=available_personas,
                    starter=starter
                )
        
        elif choice == "4":
            # View conversation history
            print_conversation_history(conversation_history)
        
        elif choice == "5":
            # Manual save (conversation is already auto-saved, but confirm)
            if conversation_filepath:
                print(f"\n✓ Conversation is saved at: {conversation_filepath}")
            else:
                conversation_filepath = save_conversation(
                    conversation_history, "group_chat",
                    participants=available_personas,
                    starter=starter
                )
                print(f"\n✓ Conversation saved to: {conversation_filepath}")
            print(f"Total messages: {turn_count}")
        
        elif choice == "6":
            # Save and exit
            conversation_filepath = save_conversation(
                conversation_history, "group_chat",
                filepath=conversation_filepath,
                participants=available_personas,
                starter=starter
            )
            print(f"\n✓ Conversation saved to: {conversation_filepath}")
            print(f"Total messages: {turn_count}")
            break
        
        elif choice == "7":
            # Exit without saving
            print("\nExiting without saving...")
            if conversation_filepath:
                print(f"(Note: Last auto-save was at: {conversation_filepath})")
            break
        
        else:
            print("Invalid choice. Please enter 1-7.")


if __name__ == "__main__":
    main()

