# Language Evolution Simulator - backend

## Running

```
python -m uv sync
python -m uv run pytest
```

Local LLM: Ollama with `hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q6_K`
(models stored in `K:\Ollama\models` via the `OLLAMA_MODELS` user env var).
Embeddings: `nomic-embed-text` in Ollama.

Try the design agent:

```
python -m uv run python -m app.cli design --example coastal_trading_sov   # uses the LLM saved in the UI
python -m uv run python -m app.cli design --example evidentiality --model <ollama model>
python -m uv run python -m app.cli generate --example coastal_trading_sov  # design + full dictionary
python -m uv run python -m app.cli graph                                   # Mermaid diagram of the workflow
```

Runs are saved to `backend/.local/runs/` with every prompt, reply, and validation issue.

## Status

- [x] Phase 1 - domain models, semantic inventory, question bank, example requests
- [x] Phase 2 - phonology engine: syllable parser/validator, seeded generators, root registry, user-word analysis
- [x] Phase 3 - morphological engine: word assembly, boundary repair, derivation, whole-lexicon validator
- [x] Phase 4 - Language Design Agent (LangGraph): design -> validate -> critique -> repair loop,
  recorded/replayable LLM calls
- [x] Phase 5 - morphological planner (LangGraph): concept selection, LLM plan, deterministic build,
  partial repair, per-derivation critic; full pipeline (`app.cli generate`)
  - Known limits with local Mistral Small: ~12-20 derivations per ~155 entries; the design agent
    sometimes names affixes after cultural themes rather than functions (agent, place...).
- [x] Phase 7 - persistence (SQLite) and API, background generation jobs with live progress
- [x] Phase 8 - frontend: questionnaire -> progress -> overview, dictionary explorer, world, generation log
- Moved to MVP 2: RAG. Decision: retrieval is only used for language *evolution* (historical precedent,
  contact outcomes, simulation memory); the founding language is generated without it.

## Layout

| Path | Purpose |
|---|---|
| `app/models/` | Pydantic domain models - the source of truth |
| `app/data/` | Question bank, semantic inventory, example requests |
| `app/questionnaire.py` | Random question selection; answers -> `GenerationRequest` |
| `app/validation/phonology_validator.py` | Syllable parser; decides what forms are legal |
| `app/generation/phoneme_generator.py` | Seeded RNG streams, syllable generation, spelling <-> phonemes |
| `app/generation/root_generator.py` | Root generation and the registry of reserved forms |
| `app/generation/user_vocabulary.py` | Infers what user words require of the phonology |
| `app/generation/morphological_engine.py` | Builds words from morphemes; the tool the Phase 5 planner will call |
| `app/validation/morphology_validator.py` | Whole-lexicon checks: compositionality, duplicates, derivation graph |
| `app/llm/chat_model.py` | Saved settings -> LangChain chat model (Ollama/OpenAI/Anthropic/OpenRouter) |
| `app/llm/structured.py` | Structured-output calls, call records, scripted and replay LLMs |
| `app/agents/design_schema.py` | The schema the design LLM fills in (field descriptions are prompt text) |
| `app/agents/design_validation.py` | Draft -> domain models, plus constraint and usability checks |
| `app/agents/language_designer.py` | The LangGraph workflow and prompts |
| `app/generation/concept_selection.py` | Core concepts + culture-matched concepts for the dictionary |
| `app/generation/dictionary_builder.py` | Executes a plan: roots, derivations, fallbacks, collision re-rolls |
| `app/agents/dictionary_schema.py` | Run-time (dynamic) schemas for plan, repair, and critic |
| `app/agents/morphological_planner.py` | The planner workflow (plan -> build -> critique -> repair) |
| `app/agents/pipeline.py` | Parent graph: design agent, then planner |
| `app/agents/runner.py` | Runs a graph with `stream()` to report each finished node (progress) |
| `app/persistence/` | SQLite tables and the repository (the only code that knows storage) |
| `app/services/generation_jobs.py` | Background generation jobs, one at a time |
| `app/services/dictionary_view.py` | Dictionary search, entry details, word families |
| `app/api/languages.py` | Generation jobs, languages, dictionary, world, and log endpoints |
| `app/cli.py` | Run agents from the command line against a real model |
| `app/generation/world_generator.py` | Fills missing civilization traits; generates neighbors and relationships |
| `app/data/world_generation.json` | All weights and rules for world generation (tunable without code changes) |
| `app/models/events.py` | Event templates, reusable effects, and recorded event history |
| `app/data/events.json` | Event and effect catalog |
| `app/simulation/events.py` | Validates slot bindings and applies an event's world effects |

## Design decisions

- **Validator is the single authority on legality.** The generator builds from
  legal syllables, but whole roots are always re-checked by the validator.
  A property test over random phonologies enforces that they agree.
- **Syllabification:** all parses are enumerated; a form is legal if any parse
  satisfies the positional restrictions. The preferred parse uses the
  maximal-onset principle (`aka` -> `a.ka`).
- **Supported syllables:** V, CV, VC, CVC. No clusters yet. Vowel-initial
  syllables are allowed; adjacent vowels (hiatus) are opt-in per language
  and inferred from user words.
- **Phoneme id = grapheme** in MVP 1 (strict one-letter-one-sound orthography).
  Forms store phoneme ids, so sound changes can later rewrite phonemes
  without touching spelling logic.
- **Reproducibility:** `derive_rng(seed, label)` gives each random step its
  own stream, so adding a new random step does not change existing output for
  the same seed. LLM outputs are not reproducible from the seed; they will be
  recorded and replayable instead (Phase 4).
- **User words are never altered** beyond trimming whitespace and lowercasing.
  Conflicts (two spellings for one meaning, one spelling for two meanings,
  unsupported letters or clusters) are reported, never resolved silently.
- **Questionnaire:** each session asks 5 random questions from the bank
  (required ones always included). Unasked questions mean "AI decides".
- **Initial language depends only on the home civilization.** Neighbors and
  events influence how the language *changes* (e.g. loanwords), never its
  founding state. The focal civilization is completed from its own RNG
  stream, so it is identical regardless of neighbor generation (tested).
- **Words can share meanings (polysemy).** A lexeme has a list of senses.
  Two routes: the user gives the same spelling to two meanings (one word,
  both senses), or the planner lists `extended_senses` (wealth = the word for
  zinc, where zinc is money). Code checks same part of speech and no
  borrowing of a borrowed word; the critic reviews shared words like
  derivations, and rejected ones get their own root. The dictionary shows
  every meaning, and searching any of them finds the word.
- **Era shapes vocabulary.** Six technology eras (Stone Age to Early Modern;
  `models/eras.py`). Concepts can be tagged with eras where they're
  characteristic (era matches count double in selection) and a `min_era`
  before which they don't exist, so a Bronze Age dictionary gets copper and
  chariots but never coins or gunpowder. Later eras keep earlier vocabulary.
  The era question is always asked; a written answer like "around 1500 BCE"
  sets both the founding year and the era. Neighbors share the era.
- **Neighbors are procedural, not LLM-generated.** 3-5 neighbors are drawn from
  seeded weighted tables (geography adjacency -> economy -> social organization
  -> traits). The LLM may later add narrative, but must not contradict these facts.
- **Attitudes are directional.** One `CivilizationRelationship` per pair holds
  shared facts (separation, contact intensity and types, relative power) plus
  `attitude_a_to_b` and `attitude_b_to_a`, so one-sided admiration is natural.
  Attitudes are sampled from base weights adjusted by trait/economy/power
  modifiers in the config. Only focal-neighbor pairs are generated for now.
- **Neighbor languages are sketches, filled lazily.** A neighbor gets a fixed
  sound system (its name is generated from it) and an empty word list. Donor
  words are created only when an event borrows them, and may reference an
  earlier donor word they build on (`base_word_id`), so e.g. "rider" can
  extend an already-borrowed "horse". The lazy-creation logic itself belongs
  to the future loanword system.
- **Events: template vs record.** An `EventTemplate` ("[HOME_CIV] declares war
  on [CIV1]") is reusable and lists effect ids from a shared effect catalog.
  An `EventRecord` is one occurrence: year, slot bindings by name
  (`{"CIV1": id}`, not positional), rendered description, and every
  before/after change, so history and etymologies can be explained.
  `apply_event` returns a new World; the old one is never mutated.
  World effects execute now; linguistic effects are recorded as pending
  until the loanword system exists. Choosing which event happens is the
  future Simulation Director's job. Every effect must involve HOME_CIV;
  events between neighbors are out of scope.
- **LLM choice is configured on first run.** Providers: local (Ollama),
  OpenAI, Anthropic, OpenRouter. The backend verifies the key/server and
  lists models from the provider's own API (nothing hardcoded). Settings
  live in `backend/.local/llm_settings.json` (git-ignored; override the path
  with `LANGEVO_SETTINGS_PATH`). The key is stored in plain text there, so
  treat that file like a password, and it is never returned to the frontend,
  which only sees a hint. Embeddings for RAG are a separate choice, to be
  made in Phase 6 (Anthropic has no embeddings API).
- **Design agent: generator + critic.** Code checks structure (`validate`);
  a second LLM role checks meaning (`critique`, temperature 0). Lessons from
  real runs that shaped it:
  - Fields with defaults are optional in the JSON schema, so the model skipped
    them. Fields needing a decision have no defaults.
  - "Keep everything else the same" plus the old JSON made repairs copy the
    old answer. Repair prompts list problems first, as instructions.
  - A critic asked "any problems?" rubber-stamped. It now reviews each affix
    (apply to an example, then classify) and code computes the verdict.
  - Open-ended critic opinions were mostly taste, and chasing them bloated
    designs (4 -> 12 features). Only narrow critic judgments trigger repair;
    the rest are reported as advisory warnings.
  - An unbounded string field looped for 9,600 words. All critic strings
    have max lengths and every call has a token cap.
  - Unambiguous fixes are made by code, not requested from the LLM: letters
    and syllable shapes the user's words need, and hyphens in affix forms.
  - Structural errors fail the run once the repair budget is spent; critic
    findings don't (the design is accepted with warnings), and a repair that
    breaks a valid design falls back to the previous version.
- **Generation runs as a background job.** `POST /api/generations` validates
  answers, then returns a job id at once; a single worker thread runs the
  pipeline and records each LangGraph node as it finishes; the UI polls.
  The language row and the job's "succeeded" status are written in one
  transaction, so a half-finished language is never visible. Jobs left
  running when the server restarts are marked `INTERRUPTED`.
- **Storage is document-style.** Languages are saved as JSON of the Pydantic
  `LanguageState` and `World`, plus indexed summary columns. Domain models
  never touch the ORM; per-entity tables can come later behind the repository.
- **The world and the language share one civilization.** The job generates
  the world first and passes its home civilization into the pipeline.
- **Words are computed, never typed.** `assemble(morphemes)` is a pure
  function; every stored form must equal what its morphemes produce
  (`FORM_MISMATCH` otherwise). Sound changes will rely on this.
- **Boundary repair is deterministic and minimal.** A language may declare a
  linking vowel (for consonant clashes) and/or linking consonant (for
  hiatus). The engine tries no repair first, then the fewest insertions, in
  a fixed order. Linking segments appear in `LexemeForm.segments` with no
  morpheme. Without a repair rule, illegal derivations are rejected.
- **Errors are written for an LLM reader.** Engine errors carry a code, a
  plain-English message, and details (e.g. available rules, the clashing
  pieces) because Phase 5's planner will receive them in its repair loop.
- **Derivation depth** is limited to 2 by default; every word form, root or
  derived, is reserved in the registry so homophones are caught.
- **Lexeme vs form:** a `Lexeme` holds identity and its ordered morphemes; a
  `LexemeForm` holds the realized phonemes for one `LanguageVersion`. Derived
  forms must be reconstructible from their morphemes.
