#!/usr/bin/env python3
"""
Convenience runner for WhatsApp persona fine-tuning pipeline.

This script provides a simple interface to run all the main commands
without needing to remember the full module paths.

Usage:
    python run.py <command> [options]

Commands:
    validate    - Validate WhatsApp export files
    process     - Generate training data from WhatsApp chats
    cost        - Estimate fine-tuning costs
    train       - Fine-tune models (use --dry-run for preview)
    chat        - Chat with a single persona
    group       - Start a group chat with all personas
    help        - Show this help message

Examples:
    python run.py validate
    python run.py validate filename.txt
    python run.py validate file1.txt file2.txt
    python run.py process
    python run.py process filename.txt
    python run.py process file1.txt file2.txt
    python run.py cost
    python run.py train --dry-run
    python run.py train "Alice Smith"
    python run.py chat
    python run.py group
"""

import sys
import subprocess


def show_help():
    """Display help message."""
    print(__doc__)


def run_module(module_path: str, extra_args: list = None):
    """Run a Python module with optional extra arguments."""
    cmd = [sys.executable, "-m", module_path]
    if extra_args:
        cmd.extend(extra_args)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)
    
    command = sys.argv[1].lower()
    extra_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    commands = {
        "validate": "processing.validate",
        "process": "processing.generate",
        "cost": "training.estimate_cost",
        "train": "training.fine_tune",
        "chat": "chat.single",
        "group": "chat.group",
        "help": None,
        "-h": None,
        "--help": None,
    }
    
    if command in ["help", "-h", "--help"]:
        show_help()
        sys.exit(0)
    
    if command not in commands:
        print(f"Unknown command: {command}")
        print("\nAvailable commands: validate, process, cost, train, chat, group, help")
        print("\nRun 'python run.py help' for more information.")
        sys.exit(1)
    
    module_path = commands[command]
    run_module(module_path, extra_args)


if __name__ == "__main__":
    main()

