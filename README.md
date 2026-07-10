# Message History Chatbot

A conversational chatbot built with **LangChain** and **Groq's Llama 3.3 70B** model that remembers conversation context across multiple turns using session-based chat history. The project demonstrates how to build stateful conversations, manage multi-language responses, and control context window size through message trimming.

## Overview

Standard LLM calls are stateless — each request is treated independently, with no memory of prior exchanges. This project solves that by wrapping a Groq-hosted LLM with LangChain's `RunnableWithMessageHistory`, allowing the chatbot to recall earlier messages within a session (e.g., remembering a user's name) while keeping separate sessions fully isolated from one another.

The notebook walks through the concept in stages: starting from a stateless model call, adding session-based memory, introducing prompt templates with dynamic variables (like response language), and finally managing context length by trimming older messages so the conversation stays within the model's token limits.

## Features

- **Stateless vs. stateful comparison** — demonstrates why raw LLM calls forget context and how message history fixes that
- **Session-based memory** — each `session_id` maintains its own independent chat history via an in-memory store
- **Multi-language responses** — a `ChatPromptTemplate` with a `{language}` variable lets the assistant respond in the user's requested language (e.g., Hindi, English)
- **Context window management** — uses `trim_messages` to cap conversation history by token count, keeping recent messages while dropping older ones
- **Composable chains** — built with LangChain Expression Language (LCEL), piping prompts, trimmers, and the model together

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — Llama 3.3 70B Versatile |
| Orchestration | LangChain (LCEL, `RunnableWithMessageHistory`) |
| Environment management | `python-dotenv` |
| Notebook | Jupyter (`ipykernel`) |

> **Note:** `requirements.txt` also includes packages for related components in this project family (FastAPI, LangServe, Streamlit, Chroma, HuggingFace) that aren't used directly in this notebook but support adjacent parts of the workflow, such as serving the chain as an API or building a UI on top of it.

## Project Structure

```
Message-History-Chatbot/
├── Message_History_Chatbot.ipynb   # Main notebook with all chatbot logic
├── requirements.txt                 # Python dependencies
├── .env                              # API keys (not committed)
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/keys)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/Message-History-Chatbot.git
   cd Message-History-Chatbot
   ```

2. Create and activate a virtual environment (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Add your Groq API key to a `.env` file in the project root
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Running the Notebook

```bash
jupyter notebook Message_History_Chatbot.ipynb
```

Run the cells in order to see the progression from a stateless model call to a full session-aware, language-adaptive, context-trimmed chatbot.

## How It Works

1. **Model setup** — `ChatGroq` is initialized with the Llama 3.3 70B model and the Groq API key.
2. **Session store** — a simple Python dictionary (`store`) maps `session_id` values to `ChatMessageHistory` objects, created on demand via `get_session_history()`.
3. **`RunnableWithMessageHistory`** — wraps the model (or chain) so every `.invoke()` call automatically reads from and appends to the correct session's history based on the `session_id` in the `config`.
4. **Prompt templates** — a `ChatPromptTemplate` combines a system instruction with a `MessagesPlaceholder`, later extended with a `{language}` variable so the assistant can reply in a specified language.
5. **Trimming** — `trim_messages` keeps only the most recent messages (by token count) before they're passed to the model, preventing the context window from growing unbounded in long conversations.

## Example Usage

```python
config = {"configurable": {"session_id": "chat1"}}

with_message_history.invoke(
    [HumanMessage(content="Hi, my name is XYZ")],
    config=config
)

with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config
)
# → "Your name is XYZ."
```

Switching to a new `session_id` starts a completely fresh conversation with no memory of the previous session.

## Future Improvements

- Replace the in-memory `store` with a persistent backend (e.g., Redis, SQLite) so history survives restarts
- Migrate from the deprecated `RunnableWithMessageHistory` to LangGraph's built-in persistence, as recommended by LangChain
- Wrap the chain in a FastAPI + LangServe endpoint for API access
- Add a Streamlit front end for an interactive chat interface
- Add automated tests for trimming and session-isolation behavior

## License

This project is open source and available under the [MIT License](LICENSE).
