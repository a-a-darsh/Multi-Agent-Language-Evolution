# Language Evolution Simulator

Create a civilization, generate its language, and (eventually) simulate how
that language evolves. See `backend/README.md` for architecture and design decisions.

## Running locally

Two terminals:

```
cd backend
python -m uv run uvicorn app.main:app --reload --port 8000
```

```
cd frontend
npm install
npx next dev -p 3000
```

Open http://localhost:3000. The frontend proxies `/api/*` to the backend
(override with the `API_URL` env var).

## What the UI does today

- Asks 5 random questions from the question bank ("Different questions" reshuffles).
- Every question can be skipped; skipped answers are decided by the generator.
- Generates the home civilization and 3-5 neighbors, with their two-way attitudes,
  relative power, separation, and contact.
- Validates custom words now, even though they're only used once language
  generation exists.

Language generation (phonology, dictionary) is not wired in yet.
