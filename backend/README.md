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
python -m uv run python -m app.cli evolve --language Varish palatalization i_umlaut  # preview sound changes (saves nothing)
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
- [x] MVP 2, step 1 - sound-change engine (IPA sounds with features, clusters, insertion, exceptions,
  coined words, saved versions) and the Linguistic Evolution Agent (tool use; see below)
- Moved to MVP 2: RAG. Decision: retrieval is only used for language *evolution* (historical precedent,
  contact outcomes, simulation memory); the founding language is generated without it.

## Layout

| Path | Purpose |
|---|---|
| `app/models/` | Pydantic domain models - the source of truth |
| `app/data/` | Question bank, semantic inventory, example requests |
| `app/models/sounds.py` | Sound catalog: IPA ids, features, default spellings |
| `app/models/sound_change.py` | Sound-change rules as data: target, output, environment |
| `app/evolution/sound_change_engine.py` | Applies a rule to every word; preview with impact report; next version |
| `app/evolution/history.py` | Replays every word's history; the one source of truth for evolved forms |
| `app/evolution/lexicon.py` | Coins new roots and derivations in an evolved language |
| `app/api/evolution.py` | Sound-change preview/commit, new words, version history |
| `app/evolution/templates.py` | Named, attested changes (palatalization, umlaut, lenition, final vowel loss...) |
| `app/persistence/migrations.py` | Upgrades stored JSON from older model versions on load |
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
| `app/agents/evolution_agent.py` | The Linguistic Evolution Agent: a tool-using loop |
| `app/agents/evolution_tools.py` | Its tools: overview, find words, preview, submit |
| `app/llm/tool_calling.py` | Tool-calling LLM wrapper with recording, scripted replies, structured fallback |
| `app/evolution/specs.py` | How the API and the agent describe a step (template or rule, exceptions) |
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
- **Sound changes are regular rules, applied by the engine.** A rule
  (`A -> B / left _ right`) names the sounds by IPA or features and never
  lists words; the engine applies it to every word that meets the condition,
  all positions at once (a rule never feeds itself). Rules substitute or
  delete one sound at a time, so segmentation survives. A feature output
  ("make it voiced") changes only that feature; a sound with no such partner
  in the catalog (/q/ has no voiced counterpart) is left alone and reported.
  `apply_sound_change` returns a preview (changed words, new and lost sounds,
  new homophones, words that would become illegal) and, if every word stays
  legal, the next `LanguageVersion`; nothing is saved by the engine.
  - **Morphemes keep their founding forms; history is replayable.**
    `evolution/history.realize` rebuilds every word from scratch: founding
    assembly, then each version's changes in order (minus recorded
    exceptions), with words coined later built from their parts as they
    sounded then. The engine, the validator, and coinage all use it, so they
    can't drift apart (a property test interleaves all three). Earlier
    versions and their forms are kept in `LanguageState`.
  - **Steps.** Several rules can form one step (one new version), checked for
    legality only at the end: final vowel loss may create "ojenr" if a later
    rule in the same step inserts a vowel ("ojener").
  - **Insertion** (`insert`): epenthesis and prothesis. An inserted sound
    joins the segment of the sound before it (or after it, word-initially).
    Matcher features may list alternatives (`manner: [nasal, trill]`).
  - **Exceptions** (`SoundChangeException`): a word may skip a rule only with
    a stated reason (lexical diffusion, frequency, analogy), at most 5% of
    the dictionary per step, and only where skipping changes something.
  - **Linking sounds evolve** like other sounds (judged on their own); if
    they disappear, the version has no boundary repair.
  - **Sound rules describe the language, so they follow it.** After a change,
    the inventory drops sounds no word uses and adds new ones (default
    spelling, or one chosen per call); syllable shapes, hiatus, and position
    restrictions are updated to fit the new words, with each change reported.
  - **Homophones from mergers are allowed** (a warning, not an error) in
    evolved versions; founding languages still forbid them.
  - A new syllable shape is added only for a word the existing shapes can't
    parse, so a CVC language doesn't gain CCV for "ap.lo".
- **Coining words in an evolved language** (`evolution/lexicon.py`). A new
  root is written in today's spelling (or generated from today's sound
  system); a new derivation joins the base word's current form with the
  affix's current shape (the founding affix with every change since applied
  to it alone). Coined words record their version, and every later change
  applies to them. Coinages may not create homophones.
- **Saving.** `Repository.save_language_state` replaces the stored state but
  only if it is still at the version the caller started from (409 / stale
  error otherwise), so two writers can't overwrite each other's history.
  API: `GET /api/sound-change-templates`, `POST
  /api/languages/{id}/sound-changes/preview` and `.../sound-changes`
  (commit), `POST .../words`, `GET .../history`. CLI: `evolve ... --save`.
  Not yet: metathesis, and a frontend for evolution.
- **Linguistic Evolution Agent: the first agent that chooses its own actions.**
  The design and dictionary agents are workflows (code routes every step).
  This one is given a language, a span of years and optional historical
  context, and decides what to do with four tools: `get_language_overview`,
  `find_words`, `preview_sound_changes` (the engine's impact report: words
  changed, homophones, illegal words, reversals of earlier changes) and
  `submit_evolution`. The graph only runs the loop (`act` -> `use_tools`)
  and enforces a budget (16 model calls). The LLM chooses and tunes rules;
  the engine applies them and has the final say, so a submission is refused
  if any word would become illegal or a rule changes nothing, and the agent
  revises. Nothing is saved unless the caller asks (`--save`, `"save": true`).
  - Local models sometimes describe a tool call instead of making it, and
    Ollama ignores `tool_choice`. Fallbacks: a step written as JSON in the
    text is run as a preview (never a submission), and otherwise the next
    action is requested through structured output, which is
    schema-constrained. See `docs/what-went-wrong.md` #35-38.
  - Tool errors are written as instructions ("a rule does one thing ... use
    two rules in the same step"), because they are the agent's feedback.
  - CLI: `evolve-agent --language X --years 200 --context "..." [--save]`
    (records the run in `.local/runs/`). API: `POST
    /api/languages/{id}/evolve-agent` (synchronous, ~1 minute locally).
- **Syllabification:** all parses are enumerated; a form is legal if any parse
  satisfies the positional restrictions. The preferred parse uses the
  maximal-onset principle (`aka` -> `a.ka`).
- **Syllables and clusters:** up to three consonants on each side of the
  vowel (CCCVCCC). Clusters follow the sonority sequencing principle:
  consonants get more vowel-like toward the vowel and less after it
  (stop < affricate < fricative < nasal < liquid < glide), so pl-, -lp, kn-
  are legal and lp-, -pl, tt- are not. `allow_s_clusters` lets s-like sounds
  break that order next to a stop at the edge (st-, str-, -ks), and is on
  only when a user word or a sound change needs it. The design agent still
  chooses from V, CV, VC, CVC; cluster shapes reach a founding language
  through user words (added by code, like required sounds) or later through
  sound changes. Syllabification prefers the largest legal onset (a.plo, not
  ap.lo). Single-consonant slots draw random numbers exactly as before, so
  seeds reproduce the same roots; cluster slots sample only legal clusters.
  Vowel-initial syllables are allowed; adjacent vowels (hiatus) are opt-in
  per language and inferred from user words.
- **Phoneme id = IPA symbol; spelling is per language.** `models/sounds.py`
  is the universal catalog (~60 sounds) with phonological features (place,
  manner, voicing; height, backness, rounding, length, nasality) for
  sound-change rules to match on. Each language's `Phoneme.grapheme` is its
  own spelling, so /ʃ/ can be "sh" in one language and "š" in another, and a
  sound change rewrites ids without touching spelling. The catalog's default
  spellings keep the original alphabet unchanged; every sound added later
  (ñ, ť, ā, ã, ü...) uses one non-ASCII letter, so no old ASCII word re-reads
  as a new sound. The design agent still picks from the original alphabet;
  new sounds reach a founding language only through user words. Languages
  saved with letter ids are upgraded on load (`persistence/migrations.py`).
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
- **Parts of speech guide, they don't block.** An affix's "takes X, makes Y"
  is shown to the planner as typical usage; any affix may attach to any word,
  and the result takes its concept's part of speech. Meaning is judged by the
  critic instead (enforcing word classes blocked wave = water + movement).
- **Compounds, affixes, and roots.** A word can be built two ways: a
  compound of two words (`LexemeOrigin.COMPOUND`, seahorse = sea + horse) or
  a word + an affix. The planner uses an affix only when base + affix says
  everything important (fisher = fish + agent), a compound when a concept
  really combines two things, and an own root for basic vocabulary. Code
  refuses an affix on a word that already has it (no ma-ma-bimaf) and
  grammatical words (in, not) as compound parts; the design agent is told
  off for affix meanings that only name a word class ("noun formation").
  The critic names what the parts leave out, and a named gap rejects the
  word. Further rules (code, not prompts, after the critic approved
  black = night + one): basic-vocabulary (core) concepts are never compounds,
  though a fitting affix may derive them (bad = not + good); numbers,
  quantities and function words are never compound parts; an affix's result
  must be its output word class (buy can't be trade + person); a word may be a
  part of at most 8 compounds. A compound is in both parts' word families. At a compound seam, two
  meeting vowels merge (the first word's final vowel drops: mifi + esmisrac
  -> mifesmisrac); for affixes, elision is the last resort after linking
  sounds. Compounds evolve and can be coined in evolved languages like any
  other word (`evolution/lexicon.compound_word`, `POST .../words` with `parts`).
- **Unambiguous design fixes are made by code.** Besides what user words
  need, code adds letters that affixes are written with, drops restrictions
  on sounds the inventory lacks, and drops restrictions a user word breaks.
- **Derived-word target.** The create page sets 0-50% (default 20%). The
  planner is told the target; an `expand` step asks for more derivations if
  the build falls short (up to 2 rounds, only among words that still have
  their own root); the critic still rejects forced ones and the result
  reports a shortfall. 0% skips planning entirely.
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
