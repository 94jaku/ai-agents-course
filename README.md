# AI Agents Course

## Setup

1. Copy `.env.example` to `.env` and add your API keys
2. Use Node.js 24.0.0 (via asdf, nvm, or fnm)

## Run

```bash
make task-1
```

Dependencies install automatically on first run.

### Task 3

```bash
cd task-3
cp .env.example .env        # add OPENAI_API_KEY and TAVILY_API_KEY
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python graph.py
```

## Tasks

- [task-1](task-1/) - LLM Tool Calling
- [task-2](task-2/) - RAG with n8n, Supabase Vector Store & OpenAI Embeddings
- [task-3](task-3/) - Multi-Agent LangGraph: Tech News Researcher & Reviewer with feedback loop
