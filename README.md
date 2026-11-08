# Sage — LLM Chatbot with Context Window Management

A multi-turn chatbot backed by a real LLM API (**Google Gemini** or **OpenAI**, your choice), with a Flask backend,
a clean web UI, a terminal UI, and two context-management strategies.

## Run it in VS Code (5 minutes)

1. **Open the folder** in VS Code (`File → Open Folder → llm-chatbot`), then open the terminal (`Ctrl+` `` ` ``).
2. **Create a virtual environment and install packages**

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   macOS / Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   (If VS Code asks to select the interpreter, pick the one inside `.venv`.)
3. **Add your API key**
   - Copy `.env.example` to a new file named `.env`.
   - Gemini (free): get a key at <https://aistudio.google.com/apikey> and paste it after `GEMINI_API_KEY=`.
   - Or OpenAI: set `LLM_PROVIDER=openai` and fill `OPENAI_API_KEY=`.
4. **Start the web app**
   ```bash
   python app.py
   ```
   Open <http://127.0.0.1:5000>.
5. **Or use the terminal version**
   ```bash
   python cli.py
   ```
   Commands: `/reset`, `/history`, `/quit`.

> No key yet? Run with `LLM_PROVIDER=mock` in `.env` to try the whole app offline (replies are canned echoes).

If you get a "model not found" message, Google/OpenAI may have retired the default model — change `GEMINI_MODEL`
or `OPENAI_MODEL` in `.env`.

## How each requirement is met

| Requirement | Where |
|---|---|
| Backend endpoint → LLM → response | `POST /api/chat` in `app.py`; provider adapters in `providers.py` |
| Multi-turn history per conversation | `ConversationStore` / `Conversation.messages` in `chat_core.py` (keyed by `conversation_id`) |
| Context window management | `ChatService._compact()` in `chat_core.py` — **summary** (default) or **truncate**, set via `CONTEXT_STRATEGY` |
| System prompt with a defined role | `SYSTEM_PROMPT` in `config.py` (Sage, an interview-prep mentor) |
| Error handling | `classify()` in `providers.py` maps timeouts, rate limits, bad keys, network and empty replies to friendly messages; retries on 429/network; the UI shows a banner with a **Retry** button; failed turns are rolled back so history is never corrupted |
| Keys in env vars, never committed | `.env` loaded by `python-dotenv`; `.env` is in `.gitignore`; only `.env.example` ships |

## How context management works

Chat APIs are stateless: every request must include the history. To stop that growing forever:

- Once a conversation has more than `MAX_HISTORY_MESSAGES` (12) messages, everything except the last
  `KEEP_RECENT_MESSAGES` (6) is removed from the history.
- **`summary` strategy:** the removed messages are first sent to the model with a summarising prompt, and the result
  is stored as a *running summary* that gets prepended to the system prompt on every later request. Facts like your
  name or goals survive even after the raw messages are gone. If the summarising call fails, it falls back to truncation.
- **`truncate` strategy:** the removed messages are simply dropped (sliding window) — simpler, cheaper, but forgets.

The UI footer shows the current context size, and whether a summary is active, so you can watch it happen.
To see it quickly, set `MAX_HISTORY_MESSAGES=6` and `KEEP_RECENT_MESSAGES=2` in `.env`.

## Project layout

```
app.py            Flask server + API routes
chat_core.py      Conversation store + context management
providers.py      Gemini / OpenAI / Mock adapters + error classification
config.py         All settings from environment variables; the system prompt
cli.py            Terminal chat
static/index.html Web UI
tests/            13 offline tests (run: python -m pytest)
.env.example      Template for your keys and settings
```

## API

```
POST   /api/chat                  {"conversation_id": "...optional", "message": "hi"}
                                  → {"conversation_id", "reply", "context": {...}}
DELETE /api/conversations/<id>    clears that conversation
GET    /api/health
```

Errors return JSON like `{"error": "Rate limit reached…", "kind": "rate_limit"}` with an appropriate HTTP status
(400 bad input, 429 rate limit, 502 API error, 503 network, 504 timeout).

## Notes & ideas to extend

- History lives in server memory, so it resets when the server restarts. To persist it, swap `ConversationStore`
  for SQLite or Redis — nothing else needs to change.
- Add streaming responses (Server-Sent Events) for a typing effect.
- Count real tokens (e.g. `tiktoken`) instead of messages for a more precise budget.
- Before pushing to GitHub run `git status` and confirm `.env` is not listed.
