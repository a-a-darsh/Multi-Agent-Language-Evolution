"""The evolution agent's tools: deterministic code the LLM calls by name.

Every tool reads the language and returns compact text (tool results go back
into the model's context, so they must be short and to the point). Nothing
here changes the language; only a successful `submit_evolution` produces a
new state, and the caller decides whether to save it.
"""

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from app.evolution.sound_change_engine import (
    SoundChangeError, SoundChangePreview, SoundChangeResult, resolve_sound,
)
from app.evolution.specs import ChangeSpec, ExceptionSpec, SpecError, run_step
from app.generation.phoneme_generator import UnsupportedSpellingError, parse_spelling
from app.llm.tool_calling import ToolSpec
from app.models.language import LanguageState

EXAMPLE_WORDS = 12
MAX_LISTED = 8


# --- argument schemas (sent to the model) -----------------------------------------------

class NoArgs(BaseModel):
    pass


class FindWordsArgs(BaseModel):
    letters: str | None = Field(default=None, description="Find words containing this sound or sequence, written "
                                "in the language's letters, e.g. 'k' or 'nr'.")
    position: Literal["anywhere", "start", "end"] = Field(default="anywhere", description="Where in the word.")
    concept: str | None = Field(default=None, description="Or find words whose meaning contains this text.")
    limit: int = Field(default=15, ge=1, le=40)


class StepArgs(BaseModel):
    changes: list[ChangeSpec] = Field(min_length=1, max_length=4, description=(
        "The sound changes of this step, applied in order. Each is a template (with optional params) or a rule."))
    exceptions: list[ExceptionSpec] = Field(default_factory=list, description=(
        "Rare words that resist a change. Use sparingly and give a reason; usually empty."))


class SubmitArgs(StepArgs):
    rationale: str = Field(min_length=20, description=(
        "Why these changes fit this language and its history, in 2-5 sentences."))


TOOLS = [
    ToolSpec(name="get_language_overview", args=NoArgs, description=(
        "The language's sounds (with how many words use each), syllable rules, past sound changes, "
        "and sample words. Call this first.")),
    ToolSpec(name="find_words", args=FindWordsArgs, description=(
        "List words containing a sound or sequence (optionally at the start or end), or whose meaning matches.")),
    ToolSpec(name="preview_sound_changes", args=StepArgs, description=(
        "Try a step of sound changes on every word without saving: which words change, new homophones, "
        "words that would become illegal, new or lost sounds. Use it to test and refine ideas.")),
    ToolSpec(name="submit_evolution", args=SubmitArgs, description=(
        "Submit your final step. It is checked again; if any word would become illegal it is refused "
        "and you can revise. Call it exactly once when you are satisfied.")),
]


# --- tools ---------------------------------------------------------------------------------

def _word(state: LanguageState, lexeme_id: str) -> str:
    form = state.form_of(lexeme_id)
    return f"{form.orthography} /{''.join(form.phonemes)}/"


def language_overview(state: LanguageState) -> str:
    ph = state.version.phonology
    forms = [state.form_of(lx.id) for lx in state.lexemes]
    usage = Counter(p for f in forms for p in set(f.phonemes))
    n = len(forms)

    def sounds(cls: str) -> str:
        return ", ".join(f"{p.grapheme} /{p.id}/ {round(100 * usage[p.id] / n)}%"
                         for p in sorted(ph.phonemes, key=lambda p: -usage[p.id])
                         if p.phoneme_class.value == cls)

    lines = [
        f"Language: {state.language.name}, year {state.version.valid_from_year}, "
        f"version {len(state.previous_versions) + 1}, {n} words.",
        "Consonants (letter /ipa/ share of words using it): " + sounds("C"),
        "Vowels: " + sounds("V"),
        f"Syllable shapes: {', '.join(ph.syllable_patterns)}. Adjacent vowels allowed: "
        f"{'yes' if ph.allow_hiatus else 'no'}. s+stop clusters (st-, -ks): {'yes' if ph.allow_s_clusters else 'no'}.",
    ]
    if ph.restrictions:
        lines.append("Restrictions: " + "; ".join(
            f"{ph.spell([r.phoneme_id])} not in {', '.join(p.value for p in r.forbidden_positions)}"
            for r in ph.restrictions))
    past = [(v.valid_from_year, c) for v in [*state.previous_versions, state.version] for c in v.sound_changes]
    lines.append("Past sound changes: " + ("; ".join(f"{year}: {c.name} ({c.description})" for year, c in past)
                                           if past else "none yet (this is the founding language)."))
    if (repair := state.version.boundary_repair) and (repair.linking_vowel or repair.linking_consonant):
        lines.append(f"Linking sounds between affix and root: vowel {repair.linking_vowel}, "
                     f"consonant {repair.linking_consonant}.")
    step = max(1, n // 20)
    lines.append("Sample words: " + "; ".join(f"{lx.concept_id} {_word(state, lx.id)}"
                                              for lx in state.lexemes[::step][:20]))
    return "\n".join(lines)


def find_words(state: LanguageState, args: FindWordsArgs) -> str:
    ph = state.version.phonology
    target = None
    if args.letters:
        try:
            target = parse_spelling(args.letters, ph)
        except UnsupportedSpellingError:
            return f"'{args.letters}' is not written with this language's letters ({' '.join(p.grapheme for p in ph.phonemes)})."
    hits = []
    for lx in state.lexemes:
        form = state.form_of(lx.id).phonemes
        if args.concept and args.concept.lower() not in lx.concept_id.lower() and \
                not any(args.concept.lower() in s.gloss.lower() for s in lx.senses):
            continue
        if target:
            k = len(target)
            spots = [i for i in range(len(form) - k + 1) if form[i:i + k] == target]
            if args.position == "start":
                spots = [i for i in spots if i == 0]
            elif args.position == "end":
                spots = [i for i in spots if i == len(form) - k]
            if not spots:
                continue
        hits.append(f"{lx.concept_id} {_word(state, lx.id)}")
    if not hits:
        return "No words match."
    shown = hits[:args.limit]
    return f"{len(hits)} words match" + (f" (showing {len(shown)})" if len(shown) < len(hits) else "") + ": " \
        + "; ".join(shown)


def _reversals(state: LanguageState, preview: SoundChangePreview) -> list[str]:
    """New mappings that undo an earlier change (a -> b, later b -> a): rarely natural."""
    versions = [*state.previous_versions, state.version]
    every_sound = {p.id for v in versions for p in v.phonology.phonemes}
    earlier = {(a, out): c.name for v in versions for c in v.sound_changes for a in every_sound
               if c.targets(a) and (out := resolve_sound(c, a)) not in (None, a)}
    return [f"{step.change.name} turns /{a}/ back into /{b}/, undoing the earlier {earlier[(b, a)]!r}"
            for step in preview.steps for a, b in step.sound_mapping.items()
            if b is not None and (b, a) in earlier]


def preview_report(state: LanguageState, result: SoundChangeResult, verdict: bool = True) -> str:
    p = result.preview
    ph = (result.state or state).version.phonology
    old = state.version.phonology
    share = round(100 * len(p.changed) / max(p.total_words, 1))
    lines = [f"{len(p.changed)} of {p.total_words} words change ({share}%)."]
    for i, s in enumerate(p.steps):
        sounds = ", ".join([f"{old.spell([a])}→{ph.spell([b]) if b else '∅'}" for a, b in s.sound_mapping.items()]
                           + [f"inserts {ph.spell([x])}" for x in s.inserted])
        extra = f"; unchanged (no counterpart): {', '.join(s.unchanged_sounds)}" if s.unchanged_sounds else ""
        skipped = f"; exceptions: {', '.join(s.exceptions)}" if s.exceptions else ""
        lines.append(f"  {i}. {s.change.name}: {s.words_changed} words; {sounds or 'no sound changed'}{extra}{skipped}")
    if p.changed:
        lines.append("Examples: " + "; ".join(f"{c.concept_id} {c.before}→{c.after}" for c in p.changed[:EXAMPLE_WORDS]))
    if p.new_sounds or p.lost_sounds:
        lines.append(f"New sounds: {', '.join(p.new_sounds) or '-'}. Lost sounds (merged or deleted): "
                     f"{', '.join(p.lost_sounds) or '-'}.")
    if p.phonotactic_changes:
        lines.append("Sound-system changes: " + "; ".join(p.phonotactic_changes) + ".")
    if p.new_homophones:
        lines.append(f"New homophones ({len(p.new_homophones)}): " + "; ".join(
            f"{h.form} = {', '.join(h.concept_ids)}" for h in p.new_homophones[:MAX_LISTED]))
    warnings = _reversals(state, p)
    if share > 50:
        warnings.append(f"{share}% of words change in one step: very fast for a few centuries")
    if not p.changed:
        warnings.append("nothing changes: check the rule's sounds and environment")
    elif idle := idle_rules(result):
        warnings.append(f"rule(s) {', '.join(idle)} change no word: drop them (submit refuses rules that do nothing)")
    if warnings:
        lines.append("Warnings: " + "; ".join(warnings) + ".")
    if p.illegal:
        lines.append(f"ILLEGAL ({len(p.illegal)} words): " + "; ".join(
            f"{w.concept_id} {w.form!r}: {w.problems[0]}" for w in p.illegal[:MAX_LISTED]))
        lines.append("Fix by narrowing the rule's environment, adding a repair rule to the same step "
                     "(e.g. cluster_epenthesis), or (rarely) an exception." if verdict else "")
    elif verdict:
        lines.append("All words stay legal: this step could be submitted.")
    return "\n".join(line for line in lines if line)


def idle_rules(result: SoundChangeResult) -> list[str]:
    return [f"{i} ({s.change.name})" for i, s in enumerate(result.preview.steps)
            if s.words_changed == 0 and not s.exceptions]


def try_step(state: LanguageState, args: StepArgs, years: int) -> tuple[SoundChangeResult | None, str]:
    try:
        result = run_step(state, args.changes, args.exceptions, years)
    except (SpecError, SoundChangeError) as e:
        return None, f"Error: {e}"
    return result, preview_report(state, result)
