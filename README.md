# Thoughtful AI Support Agent

A conversational support agent that answers questions about Thoughtful AI's healthcare automation products using semantic matching with LLM fallback.

## How it works

1. User input is matched against a knowledge base using TF-IDF cosine similarity
2. If a match is found above the confidence threshold, the predefined answer is returned
3. If no match is found, the query is forwarded to Claude for a contextual response
4. If no API key is configured, a helpful fallback message guides the user toward supported topics

## Setup

```bash
pip install -r requirements.txt
```

Optionally set your Anthropic API key for LLM fallback:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

## Run

```bash
python agent.py
```

Opens a Gradio chat interface at `http://localhost:7860`.

## Test

```bash
pytest tests/ -v
```

## Stack

- **Gradio** - Chat UI
- **scikit-learn** - TF-IDF vectorization + cosine similarity
- **Anthropic Claude** - LLM fallback for out-of-scope questions
