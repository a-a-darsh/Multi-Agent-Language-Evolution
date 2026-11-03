# Language Evolution Simulator

Create a civilization, generate its language, and (eventually) simulate how
that language evolves. See `backend/README.md` for architecture and design
decisions, and `docs/what-went-wrong.md` for problems found along the way and
how they were fixed.

## Running locally

Ollama must be running (it starts with Windows; otherwise run `ollama serve`).

**One click (Windows):** double-click `start.bat` in the repo root. It installs
dependencies if needed, opens the backend and frontend in their own windows, and
opens http://localhost:3000 once the frontend is up. Close the two windows to stop.

Or manually, in two terminals:

```
cd backend
python -m uv run uvicorn app.main:app --reload --reload-dir app --timeout-graceful-shutdown 3 --port 8000
```

```
cd frontend
npm install
npx next dev -p 3000
```

Open http://localhost:3000. If the backend seems to ignore a code change, restart it:
on Windows, uvicorn's `--reload` occasionally logs "Reloading…" without starting the new process. The frontend proxies `/api/*` to the backend
(override with the `API_URL` env var). In PowerShell use `npx next dev -p 3000`
rather than `npm run dev -- --port 3000` (PowerShell swallows the `--`).

## What the app does (MVP 1)

1. **Setup:** pick the AI model (local Ollama, OpenAI, Anthropic, or OpenRouter).
2. **Questionnaire:** 5 random questions; anything skipped is decided by the AI.
3. **Generate language:** runs in the background with live progress:
   civilization and neighbors, then the design agent, then the dictionary agent.
4. **Language page:**
   - *Overview*: sounds, syllables, grammar, word-building affixes with examples
   - *Dictionary*: search by word or meaning, filter, see how each word is built,
     what it comes from, and its word family
   - *World*: the civilization, its neighbors, and their relationships
   - *Generation log*: every agent step, problem found, and model call
5. **My languages:** everything generated is saved (`backend/.local/langevo.sqlite3`).

The CLI can run the same pipeline: `python -m uv run python -m app.cli generate --example coastal_trading_sov`.
