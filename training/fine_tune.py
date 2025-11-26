"""
Fine-tune OpenAI models for each person's training dataset.

This script:
1. Uploads each JSONL training file to OpenAI
2. Creates a fine-tuning job for each person
3. Monitors job progress until completion
4. Saves model IDs to data/models/model_ids.json for later use

Usage:
    python -m training.fine_tune                    # Fine-tune all personas
    python -m training.fine_tune --dry-run          # Show what would be done without making API calls
    python -m training.fine_tune "Alice Smith"      # Fine-tune a single persona
    python -m training.fine_tune --dry-run "Alice"  # Dry run for a single persona
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from config import (
    FINE_TUNE_BASE_MODEL,
    N_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE_MULTIPLIER,
    TRAINING_DIR,
    MODELS_DIR,
    TRAINING_COST_PER_MILLION_TOKENS,
    get_all_training_files
)
from processing.tokenizer import count_tokens

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

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Generate PERSON_FILES mapping from config
PERSON_FILES = get_all_training_files()

MODELS_FILE = MODELS_DIR / "model_ids.json"
JOBS_FILE = MODELS_DIR / "job_ids.json"  # Track in-progress jobs


def load_existing_models():
    """Load previously saved model IDs."""
    if MODELS_FILE.exists():
        with open(MODELS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_models(model_ids: dict):
    """Save model IDs to file."""
    with open(MODELS_FILE, 'w') as f:
        json.dump(model_ids, f, indent=2)
    print(f"\nModel IDs saved to: {MODELS_FILE}")


def load_job_ids():
    """Load in-progress job IDs from file."""
    if JOBS_FILE.exists():
        with open(JOBS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_job_ids(job_ids: dict):
    """Save job IDs to file."""
    with open(JOBS_FILE, 'w') as f:
        json.dump(job_ids, f, indent=2)


def check_existing_job(job_id: str) -> tuple:
    """
    Check if a job exists and its current status.
    
    Returns:
        Tuple of (status, job_object, model_id_or_none)
        If job not found, returns (None, None, None)
    """
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        model_id = None
        if hasattr(job, 'fine_tuned_model') and job.fine_tuned_model:
            model_id = job.fine_tuned_model
        return job.status, job, model_id
    except Exception:
        return None, None, None


def count_training_file_tokens(filepath: Path) -> int:
    """
    Count total tokens in a training JSONL file.
    
    Args:
        filepath: Path to the training JSONL file
        
    Returns:
        Total number of tokens in the file
    """
    total_tokens = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            # Count tokens the same way as OpenAI does (entire JSON string)
            example_json = json.dumps(example, ensure_ascii=False)
            total_tokens += count_tokens(example_json)
    
    return total_tokens


def upload_file(filepath: Path) -> str:
    """Upload training file to OpenAI and return file ID."""
    print(f"Uploading {filepath.name}...")
    with open(filepath, 'rb') as f:
        file = client.files.create(file=f, purpose='fine-tune')
    print(f"  ✓ Uploaded! File ID: {file.id}")
    return file.id


def create_fine_tuning_job(file_id: str, person_name: str, num_examples: int = None) -> tuple:
    """
    Create a fine-tuning job and return job ID and job object.
    
    Args:
        file_id: OpenAI file ID for the training data
        person_name: Name of the person being fine-tuned
        num_examples: Number of training examples (for logging/recommendations)
        
    Returns:
        Tuple of (job_id, job_object) where job_object contains n_epochs and other info
    """
    print(f"Creating fine-tuning job for {person_name}...")
    
    # Build hyperparameters dict (only include if not None/default)
    hyperparameters = {}
    
    if N_EPOCHS is not None:
        hyperparameters["n_epochs"] = N_EPOCHS
    
    if BATCH_SIZE is not None:
        hyperparameters["batch_size"] = BATCH_SIZE
    
    if LEARNING_RATE_MULTIPLIER is not None:
        hyperparameters["learning_rate_multiplier"] = LEARNING_RATE_MULTIPLIER
    
    # Print hyperparameters being used
    print(f"  Base model: {FINE_TUNE_BASE_MODEL}")
    if hyperparameters:
        print(f"  Hyperparameters: {hyperparameters}")
    else:
        print(f"  Hyperparameters: Using OpenAI's auto-selected defaults (recommended)")
        print(f"    → OpenAI will optimize epochs, batch size, and learning rate")
        print(f"    → This is the safest choice for one-shot fine-tuning")
    
    # Provide dataset-specific information
    if num_examples:
        print(f"  Training examples: {num_examples:,}")
        if N_EPOCHS is None:
            print(f"  → OpenAI will auto-select optimal epochs based on dataset size")
        else:
            if num_examples < 3000:
                print(f"  → Small dataset: 3 epochs typically recommended")
            elif num_examples < 10000:
                print(f"  → Medium dataset: 2 epochs typically recommended")
            else:
                print(f"  → Large dataset: 1-2 epochs typically recommended")
    
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=FINE_TUNE_BASE_MODEL,
        hyperparameters=hyperparameters if hyperparameters else None
    )
    print(f"  ✓ Created! Job ID: {job.id}")
    return job.id, job


def monitor_job(job_id: str, person_name: str, total_tokens: int = None, n_epochs: int = None) -> str:
    """
    Monitor fine-tuning job until completion with detailed status updates.
    
    Args:
        job_id: OpenAI fine-tuning job ID
        person_name: Name of the person being fine-tuned
        total_tokens: Total tokens that will be trained (file_tokens × epochs)
        n_epochs: Number of epochs being used
    
    Returns:
        Model ID if successful, None if failed
    """
    print(f"\nMonitoring fine-tuning job for {person_name}...")
    if total_tokens is not None and n_epochs is not None:
        # Ensure n_epochs is int for formatting
        try:
            n_epochs_int = int(n_epochs)
            print(f"  Total tokens to train: {total_tokens:,} ({n_epochs_int} epochs)")
        except (ValueError, TypeError):
            print(f"  Total tokens to train: {total_tokens:,}")
    print("  (This may take 10-30 minutes. Checking every minute...)\n")
    
    start_time = time.time()
    last_status = None
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        
        # Calculate elapsed time
        elapsed = int(time.time() - start_time)
        elapsed_str = f"{elapsed // 60}m {elapsed % 60}s"
        
        # Build status message
        status_parts = [f"Status: {status.upper()}"]
        status_parts.append(f"Elapsed: {elapsed_str}")
        
        # Add trained tokens and percentage if available
        if hasattr(job, 'trained_tokens') and job.trained_tokens:
            trained = job.trained_tokens
            status_parts.append(f"Trained: {trained:,} tokens")
            
            # Calculate percentage if we know total tokens
            if total_tokens and total_tokens > 0:
                percentage = (trained / total_tokens) * 100
                status_parts.append(f"Progress: {percentage:.1f}%")
        
        # Add epochs information
        if n_epochs:
            status_parts.append(f"Epochs: {n_epochs}")
        elif hasattr(job, 'n_epochs') and job.n_epochs:
            status_parts.append(f"Epochs: {job.n_epochs}")
        
        # Add job creation time if available
        if hasattr(job, 'created_at') and job.created_at:
            created_str = datetime.fromtimestamp(job.created_at).strftime("%H:%M:%S")
            status_parts.append(f"Started: {created_str}")
        
        # Format and print status (clear previous line with \r)
        status_line = "  " + " | ".join(status_parts)
        print(f"\r{status_line}", end="", flush=True)
        
        # Only print newline if status changed
        if status != last_status:
            print()  # New line when status changes
            last_status = status
        
        # Handle completion
        if status == "succeeded":
            print(f"\n  ✓ Fine-tuning complete!")
            if hasattr(job, 'finished_at') and job.finished_at and hasattr(job, 'created_at') and job.created_at:
                total_time = int(job.finished_at - job.created_at)
                print(f"  Total time: {total_time // 60}m {total_time % 60}s")
            if hasattr(job, 'trained_tokens') and job.trained_tokens:
                print(f"  Total tokens trained: {job.trained_tokens:,}")
                if total_tokens and total_tokens > 0:
                    percentage = (job.trained_tokens / total_tokens) * 100
                    print(f"  Completion: {percentage:.1f}%")
            if hasattr(job, 'fine_tuned_model') and job.fine_tuned_model:
                print(f"  Model ID: {job.fine_tuned_model}")
                return job.fine_tuned_model
            else:
                print(f"  ⚠ Warning: Job succeeded but no model ID found")
                return None
        
        # Handle failure
        elif status == "failed":
            print(f"\n  ✗ Fine-tuning failed!")
            if hasattr(job, 'error') and job.error:
                error_msg = job.error.get('message', str(job.error)) if isinstance(job.error, dict) else str(job.error)
                print(f"  Error: {error_msg}")
            return None
        
        # Continue monitoring
        elif status in ["validating_files", "queued", "running"]:
            time.sleep(60)  # Wait 1 minute before checking again
        else:
            print(f"\n  Unknown status: {status}")
            time.sleep(60)


def dry_run_summary(person_files_to_process: dict):
    """
    Show what would be done without making API calls.
    
    Args:
        person_files_to_process: Dictionary of filename -> person name to process
    """
    print("=" * 60)
    print("DRY RUN - No API calls will be made")
    print("=" * 60)
    print()
    
    total_tokens = 0
    total_examples = 0
    
    print(f"{'Persona':<25} {'File':<35} {'Examples':>10} {'Tokens':>12}")
    print("-" * 85)
    
    for filename, person_name in person_files_to_process.items():
        filepath = TRAINING_DIR / filename
        
        if not filepath.exists():
            print(f"{person_name:<25} {'(file not found)':<35}")
            continue
        
        # Count examples and tokens
        examples = 0
        tokens = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                examples += 1
                example = json.loads(line)
                example_json = json.dumps(example, ensure_ascii=False)
                tokens += count_tokens(example_json)
        
        print(f"{person_name:<25} {filename:<35} {examples:>10,} {tokens:>12,}")
        total_tokens += tokens
        total_examples += examples
    
    print("-" * 85)
    print(f"{'TOTAL':<25} {'':<35} {total_examples:>10,} {total_tokens:>12,}")
    print()
    
    # Cost estimates
    print("Estimated Training Costs:")
    print("-" * 40)
    for epochs in [1, 2, 3]:
        cost = (total_tokens * epochs / 1_000_000) * TRAINING_COST_PER_MILLION_TOKENS
        print(f"  {epochs} epoch{'s' if epochs > 1 else ' '}: ${cost:.2f}")
    print()
    print(f"Base model: {FINE_TUNE_BASE_MODEL}")
    print(f"Cost per million tokens: ${TRAINING_COST_PER_MILLION_TOKENS:.2f}")
    print()
    print("To proceed with fine-tuning, run without --dry-run flag.")


def main():
    """Fine-tune models for all people or a single person if specified."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fine-tune OpenAI models for persona chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m training.fine_tune                    # Fine-tune all personas
  python -m training.fine_tune --dry-run          # Preview without API calls
  python -m training.fine_tune "Alice Smith"      # Fine-tune single persona
  python -m training.fine_tune --dry-run "Alice"  # Dry run for single persona
        """
    )
    parser.add_argument(
        "persona",
        nargs="?",
        help="Name of specific persona to fine-tune (optional)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making API calls"
    )
    
    args = parser.parse_args()
    requested_person = args.persona
    dry_run = args.dry_run
    
    if requested_person:
        # Validate that the requested person exists
        if requested_person not in PERSON_FILES.values():
            print(f"Error: '{requested_person}' not found in personas.")
            print(f"\nAvailable personas:")
            for name in PERSON_FILES.values():
                print(f"  - {name}")
            return
        
        # Filter to only the requested person
        person_files_to_process = {
            filename: name for filename, name in PERSON_FILES.items()
            if name == requested_person
        }
    else:
        # Process all people
        person_files_to_process = PERSON_FILES
    
    # Handle dry run
    if dry_run:
        dry_run_summary(person_files_to_process)
        return
    
    # Normal execution
    if requested_person:
        print("=" * 60)
        print(f"OpenAI Fine-Tuning: Creating Model for {requested_person}")
        print("=" * 60)
        print()
    else:
        print("=" * 60)
        print("OpenAI Fine-Tuning: Creating Models for Each Persona")
        print("=" * 60)
        print()
    
    # Load existing models and jobs
    existing_models = load_existing_models()
    existing_jobs = load_job_ids()
    
    # Track all models (existing + new)
    all_model_ids = existing_models.copy()
    active_jobs = existing_jobs.copy()
    
    # Fine-tune models for each person
    for filename, person_name in person_files_to_process.items():
        filepath = TRAINING_DIR / filename
        
        # Skip if model already exists
        if person_name in existing_models:
            print(f"\n{person_name}: Model already exists")
            print(f"  Model ID: {existing_models[person_name]}")
            # Remove from active jobs if present (job completed)
            if person_name in active_jobs:
                del active_jobs[person_name]
                save_job_ids(active_jobs)
            continue
        
        if not filepath.exists():
            print(f"\n⚠ Skipping {person_name} - training file not found: {filepath}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {person_name}")
        print(f"{'='*60}")
        
        try:
            # Check if there's an existing job for this person
            job_id = None
            job_obj = None
            resume_job = False
            
            if person_name in existing_jobs:
                existing_job_id = existing_jobs[person_name]
                print(f"  Checking for existing job: {existing_job_id}")
                try:
                    status, job, model_id = check_existing_job(existing_job_id)
                    
                    if status == "succeeded":
                        # Job completed, save the model ID
                        print(f"  ✓ Found completed job!")
                        if model_id:
                            all_model_ids[person_name] = model_id
                            save_models(all_model_ids)
                            del active_jobs[person_name]
                            save_job_ids(active_jobs)
                            print(f"  Model ID: {model_id}")
                            print(f"\nCompleted: {person_name}")
                            continue
                    elif status in ["validating_files", "queued", "running"]:
                        # Job still in progress, resume monitoring
                        print(f"  ✓ Found in-progress job! Status: {status}")
                        print(f"  Resuming monitoring...")
                        job_id = existing_job_id
                        job_obj = job
                        resume_job = True
                    elif status == "failed":
                        # Job failed, create a new one
                        print(f"  Previous job failed. Creating new job...")
                        del active_jobs[person_name]
                        save_job_ids(active_jobs)
                    elif status is None:
                        # Job not found (deleted?), create a new one
                        print(f"  Previous job not found. Creating new job...")
                        del active_jobs[person_name]
                        save_job_ids(active_jobs)
                except Exception as e:
                    # Error checking existing job, log and create new one
                    print(f"  Error checking existing job: {e}")
                    print(f"  Creating new job...")
                    del active_jobs[person_name]
                    save_job_ids(active_jobs)
            
            # If not resuming, create a new job
            file_tokens = None
            if not resume_job:
                # First check OpenAI directly for any active jobs (before creating)
                try:
                    recent_jobs = client.fine_tuning.jobs.list(limit=50)
                    training_filename = None
                    for fn, pn in PERSON_FILES.items():
                        if pn == person_name:
                            training_filename = fn
                            break
                    
                    if training_filename:
                        for job in recent_jobs.data:
                            # Check if job is for this person and still active
                            if (hasattr(job, 'training_file') and 
                                job.status in ["validating_files", "queued", "running"]):
                                # Get filename from file object (might be nested)
                                job_filename = None
                                if hasattr(job.training_file, 'filename'):
                                    job_filename = job.training_file.filename
                                elif isinstance(job.training_file, str):
                                    # If it's just a file ID, we'd need to fetch the file details
                                    # For now, skip this check if it's just an ID
                                    pass
                                
                                if job_filename == training_filename:
                                    # Found an active job for this person
                                    print(f"  ⚠ Found active job already in progress: {job.id}")
                                    print(f"  Status: {job.status}")
                                    print(f"  Skipping creation - will resume monitoring...")
                                    job_id = job.id
                                    job_obj = job
                                    resume_job = True
                                    # Save to tracking file
                                    active_jobs[person_name] = job_id
                                    save_job_ids(active_jobs)
                                    break
                except Exception as e:
                    # If we can't check OpenAI jobs, continue with creating new job
                    print(f"  Note: Could not check OpenAI for existing jobs: {e}")
                
                # Only create if we're not resuming
                if not resume_job:
                    # Count training examples for guidance
                    num_examples = 0
                    if filepath.exists():
                        with open(filepath, 'r') as f:
                            num_examples = sum(1 for _ in f)
                    
                    # Count training file tokens
                    print(f"  Counting tokens in training file...")
                    file_tokens = count_training_file_tokens(filepath)
                    # Ensure file_tokens is an int
                    try:
                        file_tokens = int(file_tokens) if file_tokens is not None else None
                    except (ValueError, TypeError):
                        file_tokens = None
                    if file_tokens is not None:
                        print(f"  Training file contains: {file_tokens:,} tokens")
                    else:
                        print(f"  Training file contains: {file_tokens} tokens (could not count)")
                    
                    # Upload file
                    file_id = upload_file(filepath)
                    
                    # Create fine-tuning job (now returns job_id and job object)
                    job_id, job_obj = create_fine_tuning_job(file_id, person_name, num_examples)
                    
                    # Save job ID for resumption
                    active_jobs[person_name] = job_id
                    save_job_ids(active_jobs)
            else:
                # For resumed jobs, count tokens now (needed for progress percentage)
                print(f"  Counting tokens in training file...")
                file_tokens = count_training_file_tokens(filepath)
                # Ensure file_tokens is an int for safe formatting
                if file_tokens is not None:
                    try:
                        file_tokens = int(file_tokens)
                        print(f"  Training file contains: {file_tokens:,} tokens")
                    except (ValueError, TypeError):
                        print(f"  Training file contains: {file_tokens} tokens (could not format)")
                else:
                    print(f"  Warning: Could not count tokens in training file")
            
            # Get number of epochs from job object or config (same for new and resumed jobs)
            # Note: For newly created jobs, hyperparameters might not be populated yet
            n_epochs = None
            if N_EPOCHS is not None:
                n_epochs = int(N_EPOCHS)  # Ensure it's an integer
            elif job_obj and hasattr(job_obj, 'hyperparameters') and job_obj.hyperparameters:
                # Try accessing as attribute (OpenAI object) or dict key
                try:
                    if hasattr(job_obj.hyperparameters, 'n_epochs'):
                        n_epochs = job_obj.hyperparameters.n_epochs
                    elif isinstance(job_obj.hyperparameters, dict) and 'n_epochs' in job_obj.hyperparameters:
                        n_epochs = job_obj.hyperparameters['n_epochs']
                except (AttributeError, KeyError, TypeError):
                    n_epochs = None
            elif job_obj and hasattr(job_obj, 'n_epochs'):
                try:
                    n_epochs = job_obj.n_epochs
                except (AttributeError, TypeError):
                    n_epochs = None
            
            # Convert n_epochs to int if it exists (API might return string or None)
            if n_epochs is not None:
                try:
                    n_epochs = int(n_epochs)
                except (ValueError, TypeError):
                    n_epochs = None
            
            # Calculate total tokens (file_tokens × epochs)
            total_tokens = None
            # Ensure file_tokens is an int
            if file_tokens is not None:
                try:
                    file_tokens = int(file_tokens)
                except (ValueError, TypeError):
                    file_tokens = None
            
            if file_tokens is not None and n_epochs is not None:
                try:
                    total_tokens = file_tokens * n_epochs
                    # Use safe formatting - ensure all values are ints before using :,
                    if isinstance(total_tokens, int) and isinstance(n_epochs, int) and isinstance(file_tokens, int):
                        print(f"  Total tokens to train: {total_tokens:,} ({n_epochs} epochs × {file_tokens:,} tokens)")
                    else:
                        # Fallback without comma formatting
                        print(f"  Total tokens to train: {total_tokens} ({n_epochs} epochs × {file_tokens} tokens)")
                except (TypeError, ValueError) as e:
                    print(f"  Note: Could not calculate total tokens: {e}")
            elif file_tokens is not None:
                # If epochs unknown, we can't calculate total, but still show progress
                print(f"  Note: Epochs not yet determined (OpenAI will auto-select)")
            
            # Monitor job with total tokens info
            model_id = monitor_job(job_id, person_name, total_tokens, n_epochs)
            
            if model_id:
                all_model_ids[person_name] = model_id
                # Save incrementally after each successful job
                save_models(all_model_ids)
                # Remove from active jobs (job completed)
                if person_name in active_jobs:
                    del active_jobs[person_name]
                    save_job_ids(active_jobs)
            
            print(f"\nCompleted: {person_name}")
            
        except Exception as e:
            print(f"\n✗ Error processing {person_name}: {e}")
            continue
    
    # Final save
    save_models(all_model_ids)
    
    print(f"\n{'='*60}")
    print("Fine-Tuning Summary")
    print(f"{'='*60}")
    print(f"Total models available: {len(all_model_ids)}")
    for name, model_id in all_model_ids.items():
        print(f"  {name}: {model_id}")
    print()
    print("✓ Fine-tuning complete! Models are ready to use.")
    print(f"Run 'python -m chat.single' to start chatting with a persona.")


if __name__ == "__main__":
    main()

