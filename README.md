# WhatsApp Persona Fine-Tuning

Fine-tune OpenAI GPT models to mimic real people's communication styles using WhatsApp chat exports. This project provides a complete pipeline from raw chat data to interactive conversations with personalized AI personas.

## Project Context

This project was originally created for a [PowerPoint night](https://www.reddit.com/r/powerpoint/comments/1ep3lsb/what_is_a_powerpoint_night/) presentation, where friends take turns presenting on lighthearted or interesting topics. Beyond the fun premise of "replacing my friends with AI simulations," it served as an exploration into lightweight LLM fine-tuning techniques—experimenting with data preparation, prompt engineering, and creating believable conversational AI personas from real messaging data. The [full presentation PDF](Replacing-My-Friends-With-Simulations.pdf) is available in this repository. I highly recommend giving this a try; it's a fun, budget-friendly way to explore LLM fine-tuning and entertain friends!

## Features

- **WhatsApp Parser**: Robust parsing of WhatsApp chat exports with support for multi-line messages and Unicode handling
- **Training Data Generator**: Converts conversations into OpenAI fine-tuning format with configurable context windows
- **Automated Fine-Tuning**: Uploads training data and manages fine-tuning jobs with progress monitoring
- **Interactive Chat**: CLI interface for one-on-one conversations with fine-tuned personas
- **Group Chat**: Multi-persona conversations where AI personas chat with each other
- **Cost Estimation**: Estimate training costs before committing to fine-tuning

## Prerequisites

- Python 3.10+
- OpenAI API key with fine-tuning access
- WhatsApp chat export files (.txt format)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yairg98/whatsapp-persona-tuning.git
cd whatsapp-persona-tuning
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Quick Start

### 1. Export WhatsApp Chats

Export your WhatsApp conversations:
- Open WhatsApp on your phone
- Go to a chat > Menu > More > Export chat
- Choose "Without Media"
- Save the .txt file to `data/whatsapp/`

### 2. Configure Target Personas

Edit `config.py` to set the names of people you want to create personas for:

```python
TARGET_SENDERS = [
    "Alice Smith",
    "Bob Johnson",
    "Carol Williams"
]
```

Names must match exactly as they appear in the WhatsApp exports.

### 3. Validate WhatsApp Files

```bash
# Validate all files
python run.py validate

# Validate specific file(s)
python run.py validate filename.txt
python run.py validate file1.txt file2.txt
```

This checks that your WhatsApp exports are in the correct format and shows which target senders were found.

### 4. Generate Training Data

```bash
# Process all files
python run.py process

# Process specific file(s)
python run.py process filename.txt
python run.py process file1.txt file2.txt
```

This will:
- Parse WhatsApp files (all files or specified ones)
- Extract messages from target senders
- Generate JSONL training files in `data/training/`

### 5. Estimate Costs (Optional)

```bash
python run.py cost
```

Review the estimated fine-tuning costs before proceeding.

### 6. Fine-Tune Models

```bash
# Preview what would be done (no API calls)
python run.py train --dry-run

# Actually fine-tune all personas
python run.py train

# Fine-tune a specific persona
python run.py train "Alice Smith"
```

This will:
- Upload training files to OpenAI
- Create fine-tuning jobs for each persona
- Monitor progress until completion
- Save model IDs to `data/models/model_ids.json`

### 7. Chat with Personas

```bash
# One-on-one chat
python run.py chat

# Group chat with all personas
python run.py group
```

### 8. Customize System Prompts (Optional)

After chatting with your personas, you may want to refine their behavior. System prompts control how the personas respond and can be tweaked without retraining.

Edit `chat/prompts.py` to customize:

- **IDENTITY**: Core persona definition and role-playing instructions
- **ANTI_ASSISTANT**: Instructions to avoid AI assistant-like behavior
- **ENGAGEMENT**: How engaging and detailed responses should be
- **SPECIFICITY**: Level of concrete details vs. vague generalities
- **STORYTELLING**: How vivid and descriptive stories should be
- **SINGLE_CHAT_FLOW** / **GROUP_CHAT_FLOW**: Conversation flow guidance

**Example tweaks:**
- Make responses more concise: modify `ENGAGEMENT` to reduce verbosity
- Increase personality quirks: add instructions to `IDENTITY`
- Change conversation style: adjust `SINGLE_CHAT_FLOW` or `GROUP_CHAT_FLOW`

Changes take effect immediately - just restart your chat session. No retraining required!

## Project Structure

```
whatsapp-persona-tuning/
├── run.py                    # Convenience command runner
├── config.py                 # Configuration settings
│
├── processing/               # Data processing stage
│   ├── generate.py           # Main data processing pipeline
│   ├── validate.py           # WhatsApp format validation
│   ├── parser.py             # Message parsing utilities
│   ├── context.py            # Context window building
│   ├── formatter.py          # OpenAI format conversion
│   ├── tokenizer.py          # Token counting
│   └── utils.py              # Helper functions
│
├── training/                 # Model training stage
│   ├── fine_tune.py          # Fine-tuning automation
│   └── estimate_cost.py      # Training cost estimation
│
├── chat/                     # Chat interaction stage
│   ├── single.py             # One-on-one chat interface
│   ├── group.py              # Multi-persona group chat
│   ├── prompts.py            # System prompt templates
│   └── save.py               # Unified conversation saving
│
└── data/
    ├── whatsapp/             # WhatsApp chat exports
    ├── training/             # Generated training files
    ├── conversations/        # Saved chat logs
    └── models/               # Model IDs and job tracking
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `python run.py validate` | Validate all WhatsApp export files |
| `python run.py validate filename.txt` | Validate specific file(s) |
| `python run.py process` | Generate training data from all WhatsApp chats |
| `python run.py process filename.txt` | Generate training data from specific file(s) |
| `python run.py cost` | Estimate fine-tuning costs |
| `python run.py train` | Fine-tune models for all personas |
| `python run.py train --dry-run` | Preview fine-tuning without API calls |
| `python run.py train "Name"` | Fine-tune a specific persona |
| `python run.py chat` | Chat with a single persona |
| `python run.py group` | Start a group chat with all personas |

Alternatively, you can run modules directly:
```bash
python -m processing.validate                    # Validate all files
python -m processing.validate filename.txt      # Validate specific file(s)
python -m processing.generate                    # Process all files
python -m processing.generate filename.txt      # Process specific file(s)
python -m training.estimate_cost
python -m training.fine_tune
python -m chat.single
python -m chat.group
```

## Configuration

Key settings in `config.py`:

| Setting | Description | Default |
|---------|-------------|---------|
| `TARGET_SENDERS` | List of persona names to train | 3 examples |
| `MAX_CONTEXT_TOKENS` | Max tokens for context window | 250 |
| `MAX_CONTEXT_MESSAGES` | Max preceding messages | 5 |
| `FINE_TUNE_BASE_MODEL` | Base model to fine-tune | gpt-4o-mini-2024-07-18 |
| `TRAINING_COST_PER_MILLION_TOKENS` | Cost for estimation | $5.00 |
| `FILTER_ENCRYPTION_NOTICES` | Remove WhatsApp notices | True |
| `PREVENT_SAMPLE_OVERLAP` | Avoid data leakage | True |

## Cost Considerations

Fine-tuning costs depend on:
- Number of training examples
- Tokens per example
- Number of epochs (auto-selected by OpenAI)

Use the cost estimation script before fine-tuning:
```bash
python run.py cost
```

Typical costs, given default configuration:
- ~10,000 examples: ~$5

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
