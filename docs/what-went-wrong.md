# What Went Wrong (and How It Was Fixed)

A log of times the Language Evolution Simulator didn't behave as intended, how each problem was
diagnosed, and what changed. Most of the interesting ones are LLM-agent failures found by running
the real pipeline (local Mistral Small 3.2 24B via Ollama) and reading the saved run records in
`backend/.local/runs/`.

Each entry: **Issue**, **How it was diagnosed**, **Fix**, and a one-line **Takeaway**.

---

## Part 1: Agent and LLM behavior

### 1. The model skipped a required feature, and the repair loop returned the same answer three times

- **Issue:** The user asked for evidentiality ("verbs show whether the speaker saw the event or heard
  about it"). The design agent never added it. The validator flagged `MISSING_FEATURE`, the repair
  loop ran its full budget, and all three attempts failed with the identical error.
- **How it was diagnosed:** Opened the run record and compared the three drafts. They were
  byte-for-byte identical: same reasoning, same affixes, `features: []`. Then checked the JSON schema
  the model received: `features` had a default (empty list), which Pydantic marks as *optional*. With
  Ollama's constrained decoding, the model can legally leave optional fields out, and it did.
- **Fix:** Removed defaults from every field that needs a deliberate decision (`features`,
  `restrictions`, `common_sounds`, linking segments). They can still be empty or null, but the model
  must write them out. Added a test asserting those fields are in the schema's `required` list.
- **Takeaway:** In structured output, the schema is part of the prompt. "Optional" means "skippable".

### 2. Repair prompts made the model copy its previous answer

- **Issue:** Even with the right error message, repairs came back unchanged (same issue as above).
- **How it was diagnosed:** Read the repair prompt actually sent. It showed the old JSON first, then
  the problems, then ended with "fix exactly these problems and keep everything else the same". The
  last instruction plus a complete answer to copy anchored the model on its old output.
- **Fix:** Restructured the repair message. The problems come first, numbered and phrased as
  instructions. The rejected design is shown only "for reference". The prompt says explicitly that
  the new answer must differ wherever a problem points. Error messages also name the field to change
  (e.g. "the `features` list is empty… add an entry for each request").
- **Takeaway:** In a repair loop, what you show the model (and in what order) can matter more than
  what you tell it.

### 3. Outputs passed every check but were low quality

- **Issue:** A design passed validation on the first try, but used every consonant in the alphabet,
  copied affixes from the user's own words (`-mar` from *mari*, `-sav` from *sava*), and listed
  "plural" as a word-building affix (it's grammar, not word formation).
- **How it was diagnosed:** Manual review of real outputs instead of trusting the green checkmark.
  The validators only checked *structure* (legal sounds, attachable affixes, user words preserved).
  None checked *meaning*.
- **Fix:** Two layers. Deterministic quality checks where code can decide: an inventory-size limit,
  and affixes that duplicate pieces of user words. An LLM **critic** node for what code can't judge
  (is an affix derivational or inflectional?). The schema was also tightened (max 16 consonants) and
  the prompt clarified the derivational/inflectional distinction.
- **Takeaway:** "Valid" and "good" are different properties and need different checks.

### 4. The critic rubber-stamped everything

- **Issue:** The first critic accepted a design whose "derivational" affixes were case endings and
  evidential markers, even though the critic's prompt named both as inflectional.
- **How it was diagnosed:** Read the critic's output. Its reasoning was generic ("the design has
  been reviewed and found to be acceptable… affixes are correctly categorized") and never mentioned a
  single affix. It took 3.5 seconds, too fast to have examined eight items.
- **Fix:** Replaced the open "list any problems" schema with a **per-item checklist**: one review per
  affix, where the model must first *apply* the affix to an example ("see → see [witnessed]") and only
  then classify it. Code computes the verdict from the classifications; the model never gives a
  holistic "looks fine". On the next run it correctly flagged all 8 inflectional affixes.
- **Takeaway:** Ask an LLM judge narrow, per-item questions and make it show its work. Compute the
  verdict in code.

### 5. One runaway string field took 211 seconds and broke the output

- **Issue:** After the critic fix, a critique call took 211 s (normally ~8 s) and failed to parse.
- **How it was diagnosed:** The parse error contained the raw completion. It was ~75 KB, and one
  `explanation` field held ~9,600 words. The model had fallen into a repetition loop inside a single
  JSON string, until the output was cut off and the JSON was left unterminated.
- **Fix:** `max_length` on every free-text field in critic schemas (which also bounds constrained
  decoding), a cap on how many findings it can return, and a per-call output token limit
  (`num_predict` / `max_tokens`) in the model factory.
- **Takeaway:** Every LLM call needs an output budget, and every free-text field needs a length limit.

### 6. A noisy critic made designs worse

- **Issue:** With the critic active, a design ballooned from 4 grammatical features to 12 across
  repairs, and some of the critic's complaints were simply wrong (e.g. it called mixed prefixes and
  suffixes "incoherent", and invented a claim about animacy systems and hunter-gatherers).
- **How it was diagnosed:** Traced the repair rounds in the record. Each round the critic produced
  new open-ended "grammar coherence" findings, and the designer responded by adding features to
  satisfy them. The per-affix classifications, by contrast, had been accurate.
- **Fix:** Split critic output into **actionable** findings (narrow, high-precision: per-affix
  classification, inventory) that trigger repair, and **advisory** findings (open-ended opinions)
  that are reported but never acted on. Also capped features at 6.
- **Takeaway:** Only let high-precision feedback drive an automated loop. Low-precision feedback
  belongs in a report for a human.

### 7. The model fixed a constraint the wrong way

- **Issue:** The user's word *mari* requires the letter `r`. The model left `r` out of the sound
  system. When told its affixes used a missing letter, it "fixed" them by removing `r` from the
  affixes instead of adding it to the inventory, and the run failed.
- **How it was diagnosed:** Followed the three attempts in order in the record. Each repair satisfied
  the most recent message locally while moving away from the real goal.
- **Fix:** Applied the plan's principle, *prefer deterministic repair when the fix is unambiguous*.
  If a user word needs a letter, syllable shape, or vowel adjacency, code adds it before validation
  and records an `ADJUSTED_FOR_USER_WORDS` note. The LLM is never asked to make a decision that has
  only one correct answer.
- **Takeaway:** Don't spend LLM calls (or risk) on decisions code can make with certainty.

### 8. A critic-requested repair could destroy a valid design

- **Issue (found in review):** The critic flags a matter of taste, the repair introduces a
  *structural* error, and the budget runs out, so the whole run fails even though a valid design
  existed one step earlier.
- **How it was diagnosed:** Walked through the graph's routing by hand after adding the critic node
  and asked "what happens if the last repair makes things worse?"
- **Fix:** The critique step stores the last structurally valid draft as a fallback. If the budget is
  exhausted on a broken draft, `finalize` uses the fallback and records `REPAIR_REVERTED`. A test
  covers this path.
- **Takeaway:** Taste-driven changes must never be able to turn success into failure.

### 9. A design gave the dictionary planner nothing to build with

- **Issue:** The first full pipeline run produced a dictionary with **1 derived word out of 156**.
- **How it was diagnosed:** The build log showed ~48 part-of-speech errors. Looking at the design,
  all five affixes changed word class (verb → noun, noun → adjective…). None built nouns from nouns,
  so "sailor = sea + agent" or "harbor = ship + place" were impossible, and most dictionary
  concepts are nouns.
- **Fix:** Added a design-time quality check (`NO_NOUN_FAMILIES`): at least one affix must build a
  noun from a noun. It triggers a repair at design time, before the planner ever runs.
- **Takeaway:** Validate each stage against what the *next* stage needs, not just against its own spec.

### 10. Making errors impossible in the schema destroyed the quality of the meaning choices

- **Issue:** To eliminate part-of-speech errors, the planner schema was changed to one list per affix,
  where each list only accepted compatible concepts. POS errors disappeared, but derivations became
  nonsense: "hair = eat + agent", "harbor = walk + agent", "oar = sleep + agent", plus 20 entries
  like "eat = eat".
- **How it was diagnosed:** Compared the two runs' records side by side. In the earlier flat format,
  the *semantic* choices were good (wave from sea, coin from trade, harbor from ship), and only the
  rule was wrong. In the per-rule format the model picked the affix first, then walked down the verb
  list mechanically (fight, walk, make, sleep, sit, stand…), filling slots.
- **Fix:** Reverted to the flat format, ordered the way the decision is naturally made: concept, then
  base, then rule. Moved the mechanical constraint into code: if the chosen rule doesn't fit and
  exactly one rule does, code swaps it in (`RULE_CORRECTED`). Otherwise the repair message explains
  which rules or base types would work. Also gave planning its own lower temperature.
- **Takeaway:** Over-constraining output structure can wreck reasoning. Let the model make the
  judgment and let code enforce the mechanics.

### 11. A repair message acted as a forced menu

- **Issue:** When "sailor = sail + agent" failed (this language's agent suffix only takes verbs), the
  repair produced "sailor = sail + collective" and "merchant = trade + collective", which are valid
  but meaningless.
- **How it was diagnosed:** Read the error text sent to the model: "Rules that can: ['collective',
  'diminutive', 'augmentative']". With only those options listed, it picked one.
- **Fix:** The message now says to use a listed rule only if its meaning truly fits, and otherwise to
  choose a different base or an independent root.
- **Takeaway:** Error messages to an LLM are prompts. If you list options, include the way out.

### 12. Accurate critic findings were being thrown away

- **Issue:** The dictionary critic correctly flagged "net = fish + place", "snake = fight + agent" and
  "year = sun + place", but they stayed in the dictionary because the repair budget had already been
  spent on other failures. Separately, when the budget ran out the critic was **skipped entirely**
  (a routing bug).
- **How it was diagnosed:** Checked the final word families against the critique log and found
  flagged items in the output. For the routing bug, a run showed no critique call at all; tracing
  `after_build` showed it routed straight to `finalize` when out of budget.
- **Fix:** The critic now always runs after the last build. Because per-derivation judgments proved
  accurate *and* an independent root is always a safe fallback, derivations still rejected at the end
  are demoted to roots and reported. This differs deliberately from the design critic (#6), whose
  open-ended findings stay advisory.
- **Takeaway:** How much to trust a critic depends on how narrow its question is and how safe the
  fallback is.

### 13. Same base + same rule = two meanings, one word

- **Issue:** "merchant = trade + collective" and "market = trade + collective" produced the identical
  word, so one failed with a generic duplicate-form error.
- **How it was diagnosed:** The `DUPLICATE_FORM` error said market's form "is already the word for
  merchant". Inspecting the plan showed both entries had the same base and rule.
- **Fix:** A plan-level check (`SAME_COMPOSITION`) catches this before building and explains it in
  the model's terms ("both are trade + collective, which would be the same word").
- **Takeaway:** Catch problems at the level where they can be explained most clearly.

### 14. Small, unambiguous formatting slips wasted whole repair rounds

- **Issue:** The model wrote affixes as `-ma` and `mo-`. The hyphen isn't a sound, so 8 affixes were
  rejected and a full repair round (~20 s) was spent removing hyphens.
- **How it was diagnosed:** All eight errors showed the same unknown letter: `['-']`.
- **Fix:** Hyphens are stripped before parsing (notation, not content). Test added.
- **Takeaway:** Normalize harmless formatting deterministically; save repairs for real decisions.

### 15. A quality check was too strict and caused false positives

- **Issue:** The "affix copies a user word" check flagged two-letter affixes like `-va` (shares a
  syllable with *sava*) and `-ri` (end of *mari*). In languages built from CV syllables, that overlap
  is unavoidable.
- **How it was diagnosed:** A test unexpectedly ran out of scripted LLM responses. Tracing showed the
  "clean" fixture draft itself was being flagged, because its `-ri` ends the test word *mari*.
- **Fix:** Only affixes of 3+ letters (e.g. `-mar` from *mari*) count as copying.
- **Takeaway:** Validators need tests for false positives, not just for catching problems.

---

## Part 2: Code and engineering issues

### 16. A validation constraint was silently ignored

- **Issue:** `steps: int = Field(ne=0)` was meant to forbid zero-step attitude shifts in events. It
  did nothing: Pydantic has no `ne` constraint, so `steps: 0` was accepted.
- **How it was diagnosed:** A deprecation warning in the test output: "Using extra keyword arguments
  on `Field` is deprecated… (Extra keys: 'ne')".
- **Fix:** A real `field_validator` plus a test that `steps: 0` is rejected.
- **Takeaway:** Read warnings. A no-op validator looks exactly like a working one until it matters.

### 17. User words could be silently overwritten

- **Issue (found in review):** If the questionnaire asked both "what's the word for water?" and
  "define your own words", and the user gave different spellings for *water*, the second answer
  would silently replace the first, violating the rule that user words are never altered.
- **How it was diagnosed:** Reviewing the answer-merging code against the spec's "never silently
  overwrite user input" requirement.
- **Fix:** Conflicts raise `CONFLICTING_CUSTOM_WORD` with both spellings so the user can choose.
- **Takeaway:** Turn spec invariants into explicit checks in the code.

### 18. UI state and saved answers could disagree

- **Issue (found in review):** The custom-vocabulary editor kept its rows in local component state.
  After "Back to my answers", the rows reset to empty while the saved answers still contained the
  words, so the screen showed nothing but would submit hidden words.
- **How it was diagnosed:** Reasoned through the component lifecycle: going back remounts the form,
  and the editor's state wasn't initialized from the saved value.
- **Fix:** The editor initializes its rows from the saved answer.
- **Takeaway:** Two sources of truth for the same data will eventually disagree.

### 19. Embedding models offered as chat models; empty server URL

- **Issue:** The setup screen listed `nomic-embed-text` (an embedding model, which can't chat) as a
  chat-model choice. Separately, leaving the Ollama URL blank produced a request to an invalid URL.
- **How it was diagnosed:** The embedding model showed up in a live test against the real Ollama
  install. The URL issue was found reviewing the request-building code.
- **Fix:** Filter embedding models out of the chat list; default the local URL when blank.
- **Takeaway:** Test against the real service, not only mocks.

### 20. API key handling

- **Issue (design risk, prevented):** The LLM setup screen accepts API keys. A naive implementation
  would echo the key back to the browser or commit it to git.
- **How it was handled:** The key is stored only in a git-ignored local file, as a `SecretStr` (so it
  doesn't leak into logs), and is never returned by the API. The UI only sees a hint like `sk-…abcd`.
  A test asserts the key appears in no API response.
- **Takeaway:** Decide how secrets flow before writing the endpoint, and test the "never" cases.

### 21. Tooling and environment snags

- **Installer hang:** Ollama's silent installer appeared to hang for 10+ minutes. Ollama had in fact
  installed fine. PowerShell's `Start-Process -Wait` waits for *child* processes too, and the
  installer launches the long-running tray app. Fixed by checking the installed binary and stopping
  the wait.
- **`npm run dev -- --port 3000` failed in PowerShell:** PowerShell consumed the `--`, so Next.js read
  `3000` as a project directory. Fixed by calling `npx next dev -p 3000` directly.
- **Models on the wrong drive:** Moved model storage to a larger drive by setting `OLLAMA_MODELS`
  *before* installing, then verified with a small model pull that blobs landed on `K:`.

---

## Part 3: Integrating the agents into the app

### 22. Fixing structure used up the budget for fixing quality

- **Issue:** The first language generated through the GUI "succeeded", but every one of its
  word-building affixes was a grammatical ending (case markers and tense prefixes), and the dictionary
  agent then built words like "fisher = fish + genitive".
- **How it was diagnosed:** The job's step trace (recorded from LangGraph's stream for the progress
  screen) read `validate > repair > validate > repair > validate > critique > finalize`. Two structural
  repairs had used the *shared* budget of 2, so when the critic correctly flagged the inflectional
  affixes, there was no repair left and they were accepted as warnings.
- **Fix:** Separate budgets for structural repairs and quality (critic) repairs. The repair node
  decides which counter to charge based on whether the problems are hard errors. A test reproduces
  the exact sequence: 2 structural repairs, then the critic's finding still gets its own repair. The
  next GUI run showed `critique > repair`, and the affixes became agent, instrument, place and
  diminutive.
- **Takeaway:** Retry budgets are a design decision. Different failure types may need separate ones.

### 23. The model returned empty strings for required fields

- **Issue:** In the dictionary view, suffixes showed no meaning (`-as ()`). The model had returned
  `meaning: ""` for every affix.
- **How it was diagnosed:** Queried the stored language through the API: all affix `meaning` fields
  were empty strings. "Required" in the schema only forces the key to exist, not to have content.
- **Fix:** `min_length` on affix names and meanings, so constrained decoding can't emit an empty
  string, plus a display fallback to the affix name for older data. The first limit (5 characters)
  broke the test fixtures, because "not" is a valid meaning for a negative prefix. Lowered to 3.
- **Takeaway:** Required isn't the same as non-empty. Constrain content, not just presence.

### 24. Informational notes drowned out real warnings

- **Issue:** One language listed 25 generation notes; 19 of them were "rule corrected automatically",
  which hid the 6 notes about rejected derivations.
- **How it was diagnosed:** Looking at the Overview tab's note list in the API output.
- **Fix:** Group notes by kind with plain-language labels and counts, collapsed by default.
- **Takeaway:** Observability output needs design too. Volume isn't the same as information.

### 25. The world and the language had different copies of the same civilization

- **Issue (found in design):** The world generator and the design agent each built the home
  civilization from the same seed. The contents matched, but the IDs (random UUIDs) didn't, so a saved
  language would have referred to a civilization that wasn't in its saved world.
- **How it was diagnosed:** Planning the storage model: the World tab needs to join the language's
  civilization to the world's, and only content (not identity) was shared.
- **Fix:** The job generates the world first and passes its civilization object into the pipeline.
  A test asserts the IDs are identical end to end.
- **Takeaway:** "Equal" and "the same object" are different, and persistence makes the difference matter.

### 26. Making a schema stricter broke every language saved before the change

- **Issue:** Opening an older language in the GUI failed with "The server responded with 500".
  Languages generated later opened fine.
- **How it was diagnosed:** Checked each endpoint per language. Only the older language failed, and
  only on the two endpoints that read its generation record; its dictionary and world loaded fine.
  The server log showed `ValidationError … meaning: String should have at least 3 characters
  [input_value='']`. The fix for #23 (a minimum length on affix meanings) was being applied to
  *stored history*: the old record contained the empty meanings that fix was meant to prevent, so
  reading it back failed.
- **Fix:** Generation records are history and are no longer re-validated against the current
  LLM-facing schemas. The log and warning views read them through narrow, lenient models that keep
  only what they display (steps, issues, model calls, warnings) and ignore everything else. Added a
  regression test that corrupts a stored record with now-invalid and unknown fields and checks that
  the language still loads.
- **A second problem hid the fix:** After the code change the live server still returned 500. The
  traceback named a function that no longer existed. The log showed uvicorn's auto-reloader had
  detected the change ("Reloading…") but never started a new process, so the old code kept serving.
  A full restart fixed it.
- **Takeaway:** Validation rules for *new* output and for *reading stored data* are different
  concerns; tightening the first must not break the second. And when a fix "doesn't work", first
  check that the running process is actually running the new code.

### 27. "sh" was rejected as a consonant cluster

- **Issue:** A user typed *shira* as a custom word and was told it had an unsupported consonant
  cluster. But *sh* is one sound (/ʃ/) written with two letters.
- **How it was diagnosed:** The orthography was deliberately "one letter = one sound", so the parser
  read *s* + *h*, two consonants in one syllable onset, which the syllable rules forbid.
- **Fix:** Added digraphs (sh, ch, zh, th, dh, ng) as single phonemes, read longest-match first. That
  created a new risk: in a language with *s*, *h* and *sh*, the sequence *s* + *h* would be written
  "sh" and read back as /ʃ/. So the validator now requires every form's spelling to read back to
  the same sounds (`AMBIGUOUS_SPELLING`), and the random-phonology property test now includes
  digraphs to prove the generator never produces such a form. Two places that converted spelled
  strings back to sounds character by character were found and changed to use phoneme lists.
  Because *ch* took /tʃ/, the letter *c* became /ts/ (stored languages keep their recorded IPA).
- **Takeaway:** User feedback can reveal a wrong modeling assumption. Fixing it means finding every
  place the assumption was baked in, and the new edge cases the fix creates.

### 28. An invisible byte broke every questionnaire request

- **Issue:** Right after a small text edit to the question bank, the questionnaire endpoint returned 500.
- **How it was diagnosed:** The server log showed `JSONDecodeError: Unexpected UTF-8 BOM`. The first
  three bytes of `questions.json` were `EF BB BF`: the file had been edited with Windows PowerShell 5's
  `Set-Content -Encoding utf8`, which silently adds a byte-order mark.
- **Fix:** Stripped the BOM, made every JSON loader read with `utf-8-sig` (BOM-tolerant), and added a
  test that fails if any data file starts with a BOM.
- **Takeaway:** Tooling defaults differ by platform. A guard test turns a one-time surprise into a
  permanent check.

### 29. Auto-reload kept hanging, so fixes didn't take effect

- **Issue:** Several times the dev server logged "Reloading…" or "Waiting for application shutdown"
  and never started again, so it kept serving old code (see #26).
- **How it was diagnosed:** Compared reload attempts in the server log. All stopped during shutdown.
  The first theory was the background job thread; an explicit shutdown hook for it helped but didn't
  cover every case. The remaining hangs pointed to uvicorn waiting indefinitely for HTTP connections
  held open by the frontend's dev proxy.
- **Fix:** Shut the worker thread down in the app's lifespan hook, restrict reload watching to
  `app/` (editing tests no longer restarts the server), and set `--timeout-graceful-shutdown 3` so a
  lingering connection can't block a restart.
- **Takeaway:** When a problem has more than one cause, fixing the first one isn't the end. Check
  whether the symptom is really gone.

### 30. A new state key would have silently disabled an existing reducer (caught before shipping)

- **Issue (found in review):** To pass the user's language notes to the dictionary agent, a `notes`
  field was added to the planner's LangGraph state. That state already had a `notes` key declared
  with an `operator.add` reducer, which accumulates validation notes across loop iterations.
  Redeclaring it would have replaced the reducer declaration, so notes would overwrite instead of
  accumulate, with no error.
- **How it was diagnosed:** Re-reading the TypedDict after the edit.
- **Fix:** Renamed the new key to `user_notes`, with a comment explaining why the name matters.
- **Takeaway:** In LangGraph, the state schema is behavior: a key's reducer defines how updates merge.

---

## Patterns worth talking about

1. **Read the records.** Nearly every agent fix came from opening a saved run and seeing exactly what
   the model produced. Recording every prompt, reply and validation issue was the most valuable
   debugging decision in the project.
2. **Code decides what's unambiguous; the LLM decides what needs judgment.** Many fixes moved a
   decision from the LLM into code (#7, #10, #14), or from code into the LLM's hands with better
   framing (#4, #11).
3. **Feedback loops are only as good as their feedback.** Rubber-stamping (#4), noise (#6), anchoring
   (#2) and forced menus (#11) all broke the generate–validate–repair loop in different ways.
4. **Degrade gracefully.** The dictionary is always complete (failed derivations become roots), taste
   never fails a run (#8), and an unavailable critic is noted rather than fatal.
5. **Valid ≠ good.** Deterministic validation was necessary but not sufficient (#3, #9); quality
   needed both targeted checks and a well-designed critic.
