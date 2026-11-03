"""Coining words in an evolved language.

New words are made from the language as it sounds *now*:
- a new root is written in the current sound system (given, or generated);
- a new derivation joins its base word's current form with the affix's current
  shape (the founding affix with every sound change since applied to it), using
  the current boundary rules. Old-style derivations would rebuild the word from
  founding morphemes, as if it had always existed, which is what sound change
  makes impossible: Modern English forms new words from "night", not from Old
  English "niht".

Each coined word records the version it was coined in, so `history.realize`
can rebuild it and every later sound change applies to it like any other word.
Nothing is saved here; functions return a new LanguageState.
"""

from app.evolution.history import Realized, build_coined, realize
from app.generation.morphological_engine import MorphologyError
from app.generation.phoneme_generator import GenerationError, parse_spelling, render_phonemes
from app.generation.root_generator import RootRegistry
from app.models.common import new_id
from app.models.language import LanguageState
from app.models.lexicon import (
    Compound, Derivation, FormSegment, Lexeme, LexemeForm, LexemeOrigin, LexicalSense, PartOfSpeech,
)
from app.models.morphology import Morpheme, MorphemeOrigin, MorphemeType
from app.models.validation import ValidationResult
from app.validation.phonology_validator import validate_phoneme_sequence


class CoinageError(ValueError):
    def __init__(self, result: ValidationResult):
        self.result = result
        super().__init__("; ".join(i.message for i in result.errors))


def _fail(code: str, message: str, **details) -> CoinageError:
    result = ValidationResult()
    result.add(code, message, details=details)
    return CoinageError(result)


def _current_forms(state: LanguageState) -> dict[tuple[str, ...], str]:
    return {tuple(state.form_of(lx.id).phonemes): lx.concept_id for lx in state.lexemes}


def _check_new_form(state: LanguageState, concept_id: str, phonemes: list[str]) -> None:
    if any(lx.concept_id == concept_id for lx in state.lexemes):
        raise _fail("CONCEPT_EXISTS", f"The language already has a word for {concept_id!r}", concept=concept_id)
    phonology = state.version.phonology
    check = validate_phoneme_sequence(phonemes, phonology, object_id=concept_id)
    if not check.ok:
        raise CoinageError(check)
    # New coinages avoid homophones; mergers may create them, but nobody coins one on purpose.
    if (owner := _current_forms(state).get(tuple(phonemes))) is not None:
        raise _fail("DUPLICATE_FORM", f"{phonology.spell(phonemes)!r} is already the word for {owner!r}",
                    form=phonology.spell(phonemes), existing_concept=owner)


def _add(state: LanguageState, lexeme: Lexeme, form: Realized, morphemes: list[Morpheme]) -> LanguageState:
    phonology = state.version.phonology
    stored = LexemeForm(lexeme_id=lexeme.id, language_version_id=state.version.id, phonemes=form.phonemes,
                        orthography=render_phonemes(form.phonemes, phonology), segments=form.segments)
    return state.model_copy(update={"lexemes": [*state.lexemes, lexeme], "forms": [*state.forms, stored],
                                    "morphemes": [*state.morphemes, *morphemes]})


def coin_root(state: LanguageState, concept_id: str, gloss: str, part_of_speech: PartOfSpeech,
              spelling: str | None = None, seed: int = 0) -> LanguageState:
    """A brand-new root word: written as `spelling` in the current orthography, or generated."""
    phonology = state.version.phonology
    if spelling is not None:
        try:
            phonemes = parse_spelling(spelling, phonology)
        except ValueError as e:
            raise _fail("UNSUPPORTED_SPELLING", f"{spelling!r} uses letters this language no longer has: {e}",
                        spelling=spelling) from e
        origin = MorphemeOrigin.USER_DEFINED
    else:
        registry = RootRegistry(phonology, seed=seed, stream=f"coin:{state.version.id}:{concept_id}")
        for form in _current_forms(state):
            registry.reserve(list(form), owner_id="existing", owner_kind="root")
        try:
            phonemes = registry.generate_candidates(1)[0].phonemes
        except GenerationError as e:
            raise _fail("GENERATION_FAILED", f"Could not generate a new root: {e}") from e
        origin = MorphemeOrigin.GENERATED
    _check_new_form(state, concept_id, phonemes)

    morpheme = Morpheme(language_id=state.language.id, phonemes=phonemes, type=MorphemeType.ROOT, gloss=gloss,
                        origin=origin, introduced_in_version_id=state.version.id)
    lexeme = Lexeme(language_id=state.language.id, concept_id=concept_id, part_of_speech=part_of_speech,
                    origin=LexemeOrigin.USER_DEFINED if spelling is not None else LexemeOrigin.GENERATED_ROOT,
                    morpheme_ids=[morpheme.id], senses=[LexicalSense(concept_id=concept_id, gloss=gloss)],
                    coined_in_version_id=state.version.id)
    return _add(state, lexeme, Realized(phonemes, [FormSegment(phonemes=phonemes, morpheme_id=morpheme.id)]),
                [morpheme])


def derive_word(state: LanguageState, base_lexeme_id: str, rule_id: str, concept_id: str, gloss: str,
                part_of_speech: PartOfSpeech | None = None) -> LanguageState:
    """A new derived word from an existing word, using an affix as it sounds today."""
    rules = {r.id: r for r in state.derivational_rules}
    if rule_id not in rules:
        raise _fail("UNKNOWN_RULE", f"No word-building rule {rule_id!r}; known: {', '.join(rules)}")
    lexemes = {lx.id: lx for lx in state.lexemes}
    if base_lexeme_id not in lexemes:
        raise _fail("UNKNOWN_BASE", f"No word with id {base_lexeme_id!r}")
    rule, base = rules[rule_id], lexemes[base_lexeme_id]
    if rule.morpheme_id in base.morpheme_ids:
        raise _fail("AFFIX_REPEATED", f"{base.concept_id!r} already has the {rule.id!r} affix; adding it again "
                    "doesn't make a new meaning. Use a compound or a different rule")

    affix = next(m for m in state.morphemes if m.id == rule.morpheme_id)
    lexeme_id = new_id()
    lexeme = Lexeme(
        id=lexeme_id, language_id=state.language.id, concept_id=concept_id,
        part_of_speech=part_of_speech or rule.output_part_of_speech, origin=LexemeOrigin.DERIVED,
        morpheme_ids=([affix.id, *base.morpheme_ids] if affix.type == MorphemeType.PREFIX
                      else [*base.morpheme_ids, affix.id]),
        senses=[LexicalSense(concept_id=concept_id, gloss=gloss)], coined_in_version_id=state.version.id,
        derivation=Derivation(output_lexeme_id=lexeme_id, base_lexeme_id=base.id, rule_id=rule.id,
                              applied_morpheme_ids=[affix.id]))

    forms, _ = realize(state)
    try:
        built = build_coined(state, lexeme, len(state.previous_versions), forms)
    except MorphologyError as e:
        raise CoinageError(e.result) from e
    _check_new_form(state, concept_id, built.phonemes)
    return _add(state, lexeme, built, [])


def compound_word(state: LanguageState, part_lexeme_ids: list[str], concept_id: str, gloss: str,
                  part_of_speech: PartOfSpeech) -> LanguageState:
    """A new compound of two existing words, as they sound today (seahorse = sea + horse)."""
    lexemes = {lx.id: lx for lx in state.lexemes}
    if len(part_lexeme_ids) != 2 or any(p not in lexemes for p in part_lexeme_ids):
        raise _fail("UNKNOWN_BASE", "A compound needs two existing words")
    if part_lexeme_ids[0] == part_lexeme_ids[1]:
        raise _fail("SELF_COMPOUND", "A compound needs two different words")
    first, second = (lexemes[p] for p in part_lexeme_ids)
    lexeme_id = new_id()
    lexeme = Lexeme(
        id=lexeme_id, language_id=state.language.id, concept_id=concept_id, part_of_speech=part_of_speech,
        origin=LexemeOrigin.COMPOUND, morpheme_ids=[*first.morpheme_ids, *second.morpheme_ids],
        senses=[LexicalSense(concept_id=concept_id, gloss=gloss)], coined_in_version_id=state.version.id,
        compound=Compound(output_lexeme_id=lexeme_id, part_lexeme_ids=[first.id, second.id]))
    forms, _ = realize(state)
    try:
        built = build_coined(state, lexeme, len(state.previous_versions), forms)
    except MorphologyError as e:
        raise CoinageError(e.result) from e
    _check_new_form(state, concept_id, built.phonemes)
    return _add(state, lexeme, built, [])
