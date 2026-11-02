"""Whole-lexicon checks (validation levels 3 and 4 in the plan).

Run on a finished LanguageState. The engine already prevents these problems
while building; this validator guards against anything constructed or
edited outside it (persistence bugs, hand edits, future sound changes).
"""

from collections import Counter

from app.generation.morphological_engine import MorphologyError, assemble
from app.generation.phoneme_generator import render_phonemes
from app.models.language import LanguageState
from app.models.validation import ValidationResult
from app.validation.phonology_validator import validate_phoneme_sequence


def validate_lexicon(state: LanguageState, max_depth: int | None = None) -> ValidationResult:
    result = ValidationResult()
    phonology = state.version.phonology
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

        # Compositionality: the stored form must be exactly what the morphemes produce.
        try:
            rebuilt = assemble([morphemes[m] for m in lx.morpheme_ids], phonology, state.version.boundary_repair)
        except MorphologyError as e:
            result.extend(e.result)
        else:
            if rebuilt.phonemes != form.phonemes:
                result.add("FORM_MISMATCH",
                           f"{lx.concept_id!r} is stored as {''.join(form.phonemes)!r} but its morphemes "
                           f"give {''.join(rebuilt.phonemes)!r}", object_id=lx.id)

        if form.orthography != render_phonemes(form.phonemes, phonology):
            result.add("ORTHOGRAPHY_MISMATCH", f"Spelling of {lx.concept_id!r} does not match its phonemes",
                       object_id=lx.id)
        result.extend(validate_phoneme_sequence(form.phonemes, phonology, object_id=lx.id))

        if lx.derivation:
            _check_derivation(lx, lexemes, rules, result)

    duplicates = Counter(tuple(f[0].phonemes) for f in forms_by_lexeme.values() if len(f) == 1)
    for form, count in duplicates.items():
        if count > 1:
            result.add("DUPLICATE_FORM", f"{''.join(form)!r} is used by {count} different lexemes",
                       details={"form": "".join(form)})

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
    if base.part_of_speech not in rule.input_parts_of_speech:
        result.add("POS_MISMATCH", f"{rule.id!r} cannot apply to {base.part_of_speech.value} {base.concept_id!r}",
                   object_id=lx.id)
    if lx.part_of_speech != rule.output_part_of_speech:
        result.add("POS_MISMATCH", f"{lx.concept_id!r} should be a {rule.output_part_of_speech.value}",
                   object_id=lx.id)


def _check_graph(lexemes, result: ValidationResult, max_depth: int | None) -> None:
    """The derivation graph must be acyclic (and optionally depth-limited)."""
    for start in lexemes.values():
        seen, current, depth = {start.id}, start, 0
        while current.derivation and current.derivation.base_lexeme_id in lexemes:
            current = lexemes[current.derivation.base_lexeme_id]
            depth += 1
            if current.id in seen:
                result.add("DERIVATION_CYCLE", f"{start.concept_id!r} is part of a derivation cycle",
                           object_id=start.id)
                break
            seen.add(current.id)
        else:
            if max_depth is not None and depth > max_depth:
                result.add("DERIVATION_TOO_DEEP", f"{start.concept_id!r} is {depth} steps from its root",
                           object_id=start.id)
