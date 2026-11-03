"""Whole-lexicon checks (validation levels 3 and 4 in the plan).

Run on a finished LanguageState. The engine already prevents these problems
while building; this validator guards against anything constructed or
edited outside it (persistence bugs, hand edits, future sound changes).
"""

from collections import Counter

from app.evolution.history import realize
from app.generation.phoneme_generator import render_phonemes
from app.models.language import LanguageState
from app.models.validation import Severity, ValidationResult
from app.validation.phonology_validator import validate_phoneme_sequence


def validate_lexicon(state: LanguageState, max_depth: int | None = None) -> ValidationResult:
    result = ValidationResult()
    phonology = state.version.phonology
    # Compositionality: every stored form must be exactly what its history produces - founding
    # assembly, each version's sound changes (minus recorded exceptions), and words coined
    # later built from their parts as they sounded then. See app.evolution.history.
    expected_forms, history_problems = realize(state)
    result.extend(history_problems)
    morphemes = {m.id: m for m in state.morphemes}
    rules = {r.id: r for r in state.derivational_rules}
    lexemes = {lx.id: lx for lx in state.lexemes}
    forms_by_lexeme: dict[str, list] = {}
    for f in state.forms:
        if f.language_version_id == state.version.id:
            forms_by_lexeme.setdefault(f.lexeme_id, []).append(f)

    for rule in rules.values():
        if rule.morpheme_id not in morphemes:
            result.add("UNKNOWN_AFFIX", f"Rule {rule.id!r} references a missing affix", object_id=rule.id)

    for lx in lexemes.values():
        missing = [m for m in lx.morpheme_ids if m not in morphemes]
        if missing:
            result.add("MISSING_MORPHEME", f"{lx.concept_id!r} references missing morphemes",
                       object_id=lx.id, details={"morpheme_ids": missing})
            continue

        forms = forms_by_lexeme.get(lx.id, [])
        if len(forms) != 1:
            result.add("FORM_COUNT", f"{lx.concept_id!r} has {len(forms)} forms in this version (expected 1)",
                       object_id=lx.id)
            continue
        form = forms[0]

        expected = expected_forms.get(lx.id)
        if expected is not None and expected.phonemes != form.phonemes:
            result.add("FORM_MISMATCH",
                       f"{lx.concept_id!r} is stored as {phonology.spell(form.phonemes)!r} but its "
                       f"history gives {phonology.spell(expected.phonemes)!r}", object_id=lx.id)

        if form.orthography != render_phonemes(form.phonemes, phonology):
            result.add("ORTHOGRAPHY_MISMATCH", f"Spelling of {lx.concept_id!r} does not match its phonemes",
                       object_id=lx.id)
        result.extend(validate_phoneme_sequence(form.phonemes, phonology, object_id=lx.id))

        if lx.derivation:
            _check_derivation(lx, lexemes, rules, result)
        if lx.compound:
            _check_compound(lx, lexemes, result)

    duplicates = Counter(tuple(f[0].phonemes) for f in forms_by_lexeme.values() if len(f) == 1)
    for form, count in duplicates.items():
        if count > 1:
            # A founding language never has homophones (the generator avoids them), but sound
            # changes merge words for real, as English "knight"/"night" did.
            result.add("DUPLICATE_FORM", f"{phonology.spell(form)!r} is used by {count} different lexemes",
                       details={"form": phonology.spell(form)},
                       severity=Severity.WARNING if state.previous_versions else Severity.ERROR)

    _check_graph(lexemes, result, max_depth)
    return result


def _check_derivation(lx, lexemes, rules, result: ValidationResult) -> None:
    d = lx.derivation
    base = lexemes.get(d.base_lexeme_id)
    rule = rules.get(d.rule_id)
    if base is None:
        result.add("UNKNOWN_BASE", f"{lx.concept_id!r} derives from a missing lexeme", object_id=lx.id)
        return
    if rule is None:
        result.add("UNKNOWN_RULE", f"{lx.concept_id!r} uses missing rule {d.rule_id!r}", object_id=lx.id)
        return
    if d.applied_morpheme_ids != [rule.morpheme_id]:
        result.add("COMPOSITION_MISMATCH", f"{lx.concept_id!r} records the wrong affix for {rule.id!r}",
                   object_id=lx.id)
    expected = {(rule.morpheme_id, *base.morpheme_ids), (*base.morpheme_ids, rule.morpheme_id)}
    if tuple(lx.morpheme_ids) not in expected:
        result.add("COMPOSITION_MISMATCH",
                   f"{lx.concept_id!r} should be its base's morphemes plus the {rule.id!r} affix",
                   object_id=lx.id)
    # Parts of speech are not checked: an affix's "takes X, makes Y" is typical usage, not a rule.


def _check_compound(lx, lexemes, result: ValidationResult) -> None:
    parts = [lexemes.get(p) for p in lx.compound.part_lexeme_ids]
    if any(p is None for p in parts):
        result.add("UNKNOWN_BASE", f"{lx.concept_id!r} is a compound of a missing lexeme", object_id=lx.id)
        return
    if lx.morpheme_ids != [*parts[0].morpheme_ids, *parts[1].morpheme_ids]:
        result.add("COMPOSITION_MISMATCH", f"{lx.concept_id!r} should be its two parts' morphemes in order",
                   object_id=lx.id)


def _check_graph(lexemes, result: ValidationResult, max_depth: int | None) -> None:
    """The building graph (derivation bases and compound parts) must be acyclic, optionally depth-limited."""
    depths: dict[str, int] = {}

    def depth(lexeme_id: str, path: tuple[str, ...]) -> int | None:
        if lexeme_id in path:
            return None  # cycle
        if lexeme_id not in depths:
            bases = [b for b in lexemes[lexeme_id].base_lexeme_ids if b in lexemes]
            below = [depth(b, (*path, lexeme_id)) for b in bases]
            if any(d is None for d in below):
                return None
            depths[lexeme_id] = 1 + max(below) if below else 0
        return depths[lexeme_id]

    for start in lexemes.values():
        d = depth(start.id, ())
        if d is None:
            result.add("DERIVATION_CYCLE", f"{start.concept_id!r} is part of a derivation cycle", object_id=start.id)
        elif max_depth is not None and d > max_depth:
            result.add("DERIVATION_TOO_DEEP", f"{start.concept_id!r} is {d} steps from its root",
                       object_id=start.id)
