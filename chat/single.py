"""
Interactive CLI to chat with fine-tuned persona models.

Maintains conversation history and saves each conversation to a file.

Usage:
    python -m chat.single
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from config import MODELS_DIR, CHAT_TEMPERATURE, CHAT_MAX_TOKENS
from chat.prompts import get_single_chat_prompt
from chat.save import save_conversation

# Load environment variables from .env file
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


def load_model_ids():
    """Load model IDs from saved file."""
    if not MODELS_FILE.exists():
        print(f"Error: Model IDs file not found: {MODELS_FILE}")
        print("Please run 'python -m training.fine_tune' first to create the models.")
        return {}
    
    with open(MODELS_FILE, 'r') as f:
        return json.load(f)


def chat_with_persona(person_name: str):
    """
    Chat with a specific persona's fine-tuned model.
    
    Args:
        person_name: Name of the person whose model to use
    """
    model_ids = load_model_ids()
    
    if not model_ids:
        print("No fine-tuned models found. Please run 'python -m training.fine_tune' first.")
        return
    
    if person_name not in model_ids:
        print(f"Error: No fine-tuned model found for '{person_name}'")
        print(f"\nAvailable personas:")
        for name in model_ids.keys():
            print(f"  - {name}")
        return
    
    model_id = model_ids[person_name]
    
    # Get system prompt for single chat context
    system_prompt = get_single_chat_prompt(person_name)
    
    # Initialize conversation with system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    print(f"\n{'='*60}")
    print(f"Chatting with {person_name}")
    print(f"Model ID: {model_id[:30]}...")
    print(f"{'='*60}")
    print("\nCommands:")
    print("  - Type your message and press Enter to chat")
    print("  - Type '/quit' or '/exit' to end conversation")
    print("  - Type '/clear' to clear conversation history")
    print("  - Type '/save' to save conversation and start new one")
    print()
    
    conversation_count = 1
    
    while True:
        user_input = input(f"You: ").strip()
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() in ['/quit', '/exit']:
            # Save conversation without system message (for cleaner saved conversations)
            if len(messages) > 1:  # More than just system message
                conversation_messages = [m for m in messages if m["role"] != "system"]
                filepath = save_conversation(
                    conversation_messages, "single_chat", persona=person_name
                )
                print(f"\n✓ Conversation saved to: {filepath}")
            print("\nGoodbye!")
            break
        
        if user_input.lower() == '/clear':
            if len(messages) > 1:
                conversation_messages = [m for m in messages if m["role"] != "system"]
                filepath = save_conversation(
                    conversation_messages, "single_chat", persona=person_name
                )
                print(f"✓ Previous conversation saved to: {filepath}")
            # Reset to just system message
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            conversation_count += 1
            print("Conversation cleared. Starting new conversation.\n")
            continue
        
        if user_input.lower() == '/save':
            if len(messages) > 1:
                conversation_messages = [m for m in messages if m["role"] != "system"]
                filepath = save_conversation(
                    conversation_messages, "single_chat", persona=person_name
                )
                print(f"✓ Conversation saved to: {filepath}")
                # Reset to just system message
                messages = [
                    {"role": "system", "content": system_prompt}
                ]
                conversation_count += 1
                print("Starting new conversation.\n")
            else:
                print("No conversation to save.\n")
            continue
        
        # Add user message to conversation history
        messages.append({"role": "user", "content": user_input})
        
        try:
            # Get response from model
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=CHAT_TEMPERATURE,
                max_tokens=CHAT_MAX_TOKENS
            )
            
            assistant_response = response.choices[0].message.content
            print(f"{person_name}: {assistant_response}\n")
            
            # Add assistant response to conversation history
            messages.append({"role": "assistant", "content": assistant_response})
            
        except Exception as e:
            print(f"Error: {e}\n")
            # Remove the user message if there was an error
            messages.pop()


def select_persona():
    """Interactive persona selection."""
    model_ids = load_model_ids()
    
    if not model_ids:
        print("No fine-tuned models found. Please run 'python -m training.fine_tune' first.")
        return None
    
    print("\nAvailable personas:")
    print("-" * 40)
    personas = list(model_ids.keys())
    for i, name in enumerate(personas, 1):
        print(f"  {i}. {name}")
    print()
    
    while True:
        choice = input("Select a persona (number or name, or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            return None
        
        # Handle number selection
        if choice.isdigit() and 1 <= int(choice) <= len(personas):
            return personas[int(choice) - 1]
        
        # Handle name selection
        if choice in personas:
            return choice
        
        print(f"Invalid selection: '{choice}'. Please try again.")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Persona Chat Interface")
    print("=" * 60)
    
    # Check if model IDs file exists
    if not MODELS_FILE.exists():
        print(f"\nNo fine-tuned models found.")
        print(f"Please run 'python -m training.fine_tune' first to create the models.")
        return
    
    # Select persona
    person_name = select_persona()
    
    if person_name:
        # Start chat
        chat_with_persona(person_name)


if __name__ == "__main__":
    main()

