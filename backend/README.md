# Language Evolution Simulator - backend

## Running

```
python -m uv sync
python -m uv run pytest
```

Local LLM: Ollama with `hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q6_K`
(models stored in `K:\Ollama\models` via the `OLLAMA_MODELS` user env var).
Embeddings: `nomic-embed-text` in Ollama.

## Status

- [x] Phase 1 - domain models, semantic inventory, question bank, example requests
- [x] Phase 2 - phonology engine: syllable parser/validator, seeded generators, root registry, user-word analysis
- [x] Phase 3 - morphological engine: word assembly, boundary repair, derivation, whole-lexicon validator
- [ ] Phase 4 - Language Design Agent (LangGraph)
- [ ] Phase 5 - morphological planning
- [ ] Phase 6 - RAG
- [ ] Phase 7 - persistence and API
- [ ] Phase 8 - frontend

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
