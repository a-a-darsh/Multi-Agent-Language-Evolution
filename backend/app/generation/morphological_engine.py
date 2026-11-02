"""Deterministic word building: roots + affixes -> validated word forms.

This is the *tool* the morphological planner (an LLM, Phase 5) will use.
The LLM decides "sailor = sea + associated-person suffix"; this engine builds
the form, checks it, and returns precise errors the LLM can act on.

Forms are always computed from morphemes by `assemble`, never stored as
free-standing strings, so later sound changes can re-derive every word.
"""

from collections.abc import Iterable, Sequence
from itertools import combinations

from app.generation.phoneme_generator import render_phonemes
from app.generation.root_generator import DuplicateFormError, RootRegistry
from app.models.common import DomainModel, new_id
from app.models.lexicon import (
    Derivation,
    FormSegment,
    LexemeForm,
    LexemeOrigin,
    Lexeme,
    LexicalSense,
    PartOfSpeech,
)
from app.models.morphology import BoundaryRepair, DerivationalRule, Morpheme, MorphemeOrigin, MorphemeType
from app.models.phonology import Phonology, PhonemeClass
from app.models.validation import ValidationResult
from app.validation.phonology_validator import validate_phoneme_sequence

DEFAULT_MAX_DERIVATION_DEPTH = 2


class MorphologyError(Exception):
    def __init__(self, result: ValidationResult):
        self.result = result
        super().__init__("; ".join(i.message for i in result.errors))


def _article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def _fail(code: str, message: str, **kwargs) -> MorphologyError:
    result = ValidationResult()
    result.add(code, message, **kwargs)
    return MorphologyError(result)


class AssemblyResult(DomainModel):
    phonemes: list[str]
    segments: list[FormSegment]


def _build(pieces: Sequence[Morpheme], phonology: Phonology, repair: BoundaryRepair | None,
           insert_at: set[int]) -> AssemblyResult:
    """Concatenate pieces, inserting a linking segment before piece i for each i in insert_at."""
    segments: list[FormSegment] = []
    for i, piece in enumerate(pieces):
        if i in insert_at:
            left_is_vowel = phonology.phoneme(pieces[i - 1].phonemes[-1]).phoneme_class == PhonemeClass.VOWEL
            link = repair.linking_consonant if left_is_vowel else repair.linking_vowel
            segments.append(FormSegment(phonemes=[link]))
        segments.append(FormSegment(phonemes=list(piece.phonemes), morpheme_id=piece.id))
    return AssemblyResult(phonemes=[p for s in segments for p in s.phonemes], segments=segments)


def _repairable_boundaries(pieces: Sequence[Morpheme], phonology: Phonology,
                           repair: BoundaryRepair | None) -> list[int]:
    if repair is None:
        return []
    boundaries = []
    for i in range(1, len(pieces)):
        left = phonology.phoneme(pieces[i - 1].phonemes[-1]).phoneme_class
        right = phonology.phoneme(pieces[i].phonemes[0]).phoneme_class
        if left == right == PhonemeClass.CONSONANT and repair.linking_vowel:
            boundaries.append(i)
        elif left == right == PhonemeClass.VOWEL and repair.linking_consonant:
            boundaries.append(i)
    return boundaries


def assemble(pieces: Sequence[Morpheme], phonology: Phonology,
             repair: BoundaryRepair | None = None, object_id: str | None = None) -> AssemblyResult:
    """Build a legal form from morphemes, using the fewest boundary repairs possible.

    Tries no repairs first, then every combination of 1, 2, ... repairable
    boundaries in a fixed order, so the result is deterministic.
    """
    naive = _build(pieces, phonology, repair, set())
    first_check = validate_phoneme_sequence(naive.phonemes, phonology, object_id=object_id)
    if first_check.ok:
        return naive

    repairable = _repairable_boundaries(pieces, phonology, repair)
    for count in range(1, len(repairable) + 1):
        for chosen in combinations(repairable, count):
            candidate = _build(pieces, phonology, repair, set(chosen))
            if validate_phoneme_sequence(candidate.phonemes, phonology).ok:
                return candidate

    form = "".join(naive.phonemes)
    result = ValidationResult()
    result.add(
        "ILLEGAL_DERIVED_FORM",
        f"{'+'.join(''.join(p.phonemes) for p in pieces)} gives {form!r}, which is not a legal word"
        + (" even with boundary repairs" if repairable else ""),
        object_id=object_id,
        details={"pieces": ["".join(p.phonemes) for p in pieces], "form": form,
                 "causes": [i.message for i in first_check.errors], "repairs_tried": bool(repairable)},
    )
    raise MorphologyError(result)


class MorphologyEngine:
    """Builds a lexicon one entry at a time, rejecting anything invalid.

    Every word form is reserved in the RootRegistry, so a derived word can
    never accidentally duplicate an existing root (or vice versa).
    """

    def __init__(
        self,
        *,
        language_id: str,
        version_id: str,
        phonology: Phonology,
        registry: RootRegistry,
        affixes: Iterable[Morpheme] = (),
        rules: Iterable[DerivationalRule] = (),
        boundary_repair: BoundaryRepair | None = None,
        max_depth: int = DEFAULT_MAX_DERIVATION_DEPTH,
    ):
        self.language_id = language_id
        self.version_id = version_id
        self.phonology = phonology
        self.registry = registry
        self.boundary_repair = boundary_repair
        self.max_depth = max_depth
        self.morphemes: dict[str, Morpheme] = {m.id: m for m in affixes}
        self.rules: dict[str, DerivationalRule] = {r.id: r for r in rules}
        self.lexemes: dict[str, Lexeme] = {}
        self.forms: dict[str, LexemeForm] = {}
        self._check_configuration()

    def _check_configuration(self) -> None:
        result = ValidationResult()
        for m in self.morphemes.values():
            if m.type == MorphemeType.ROOT:
                result.add("INVALID_AFFIX", f"Morpheme {m.gloss!r} is a root, not an affix", object_id=m.id)
            unknown = sorted({p for p in m.phonemes if not self.phonology.has_phoneme(p)})
            if unknown:
                result.add("UNKNOWN_PHONEME", f"Affix -{''.join(m.phonemes)} uses unknown phonemes {unknown}",
                           object_id=m.id, details={"phonemes": unknown})
        for r in self.rules.values():
            if r.morpheme_id not in self.morphemes:
                result.add("UNKNOWN_AFFIX", f"Rule {r.id!r} references a missing affix", object_id=r.id)
        if self.boundary_repair:
            for field, cls in (("linking_vowel", PhonemeClass.VOWEL), ("linking_consonant", PhonemeClass.CONSONANT)):
                p = getattr(self.boundary_repair, field)
                if p and (not self.phonology.has_phoneme(p) or self.phonology.phoneme(p).phoneme_class != cls):
                    result.add("INVALID_REPAIR", f"{field} {p!r} must be a {cls.name.lower()} in the inventory")
        if not result.ok:
            raise MorphologyError(result)

    # --- roots ------------------------------------------------------------

    def add_root(self, concept_id: str, gloss: str, part_of_speech: PartOfSpeech, phonemes: Sequence[str],
                 user_defined: bool = False) -> Lexeme:
        lexeme_id = new_id()
        check = validate_phoneme_sequence(phonemes, self.phonology, object_id=concept_id)
        if not check.ok:
            raise MorphologyError(check)
        self._reserve(phonemes, lexeme_id, "user_word" if user_defined else "root", concept_id)
        return self._store_root(lexeme_id, concept_id, gloss, part_of_speech, phonemes, user_defined)

    def add_generated_root(self, concept_id: str, gloss: str, part_of_speech: PartOfSpeech,
                           candidate_id: str) -> Lexeme:
        """Assign a root candidate from the registry to a concept."""
        lexeme_id = new_id()
        candidate = self.registry.accept(candidate_id, owner_id=lexeme_id)
        return self._store_root(lexeme_id, concept_id, gloss, part_of_speech, candidate.phonemes, False)

    def _store_root(self, lexeme_id: str, concept_id: str, gloss: str, part_of_speech: PartOfSpeech,
                    phonemes: Sequence[str], user_defined: bool) -> Lexeme:
        morpheme = Morpheme(language_id=self.language_id, phonemes=list(phonemes), type=MorphemeType.ROOT,
                            gloss=gloss, origin=MorphemeOrigin.USER_DEFINED if user_defined else MorphemeOrigin.GENERATED)
        lexeme = Lexeme(
            id=lexeme_id, language_id=self.language_id, concept_id=concept_id, part_of_speech=part_of_speech,
            origin=LexemeOrigin.USER_DEFINED if user_defined else LexemeOrigin.GENERATED_ROOT,
            morpheme_ids=[morpheme.id], senses=[LexicalSense(concept_id=concept_id, gloss=gloss)],
        )
        self.morphemes[morpheme.id] = morpheme
        self._store(lexeme, AssemblyResult(phonemes=list(phonemes),
                                           segments=[FormSegment(phonemes=list(phonemes), morpheme_id=morpheme.id)]))
        return lexeme

    # --- derivation ---------------------------------------------------------

    def derive(self, base_lexeme_id: str, rule_id: str, concept_id: str, gloss: str,
               part_of_speech: PartOfSpeech | None = None) -> Lexeme:
        """Apply a derivational rule to an existing lexeme, or raise with the reason it can't be done.

        Any affix may attach to any word: the rule's parts of speech are typical usage, not
        a constraint (enforcing them blocked natural words like wave = water + movement).
        The new word's part of speech is the concept's, falling back to the rule's usual output.
        """
        base = self.lexemes.get(base_lexeme_id)
        if base is None:
            raise _fail("UNKNOWN_BASE", f"No lexeme {base_lexeme_id!r} to derive {concept_id!r} from",
                        object_id=concept_id)
        rule = self.rules.get(rule_id)
        if rule is None:
            raise _fail("UNKNOWN_RULE", f"No derivational rule {rule_id!r}", object_id=concept_id,
                        details={"available_rules": sorted(self.rules)})
        if (depth := self.depth(base.id) + 1) > self.max_depth:
            raise _fail("DERIVATION_TOO_DEEP",
                        f"{concept_id!r} would be {depth} derivations from a root (max {self.max_depth})",
                        object_id=concept_id)

        affix = self.morphemes[rule.morpheme_id]
        morpheme_ids = ([affix.id, *base.morpheme_ids] if affix.type == MorphemeType.PREFIX
                        else [*base.morpheme_ids, affix.id])
        assembly = assemble([self.morphemes[m] for m in morpheme_ids], self.phonology,
                            self.boundary_repair, object_id=concept_id)

        lexeme_id = new_id()
        self._reserve(assembly.phonemes, lexeme_id, "derived", concept_id)
        lexeme = Lexeme(
            id=lexeme_id, language_id=self.language_id, concept_id=concept_id,
            part_of_speech=part_of_speech or rule.output_part_of_speech, origin=LexemeOrigin.DERIVED,
            morpheme_ids=morpheme_ids,
            senses=[LexicalSense(concept_id=concept_id, gloss=gloss)],
            derivation=Derivation(output_lexeme_id=lexeme_id, base_lexeme_id=base.id, rule_id=rule.id,
                                  applied_morpheme_ids=[affix.id]),
        )
        self._store(lexeme, assembly)
        return lexeme

    def depth(self, lexeme_id: str) -> int:
        """Number of derivation steps between a lexeme and its root."""
        lexeme, steps = self.lexemes[lexeme_id], 0
        while lexeme.derivation:
            lexeme, steps = self.lexemes[lexeme.derivation.base_lexeme_id], steps + 1
        return steps

    def word_family(self, root_lexeme_id: str) -> list[Lexeme]:
        """Every lexeme derived (directly or indirectly) from a root, root first."""
        family = [self.lexemes[root_lexeme_id]]
        for lexeme in family:
            family.extend(lx for lx in self.lexemes.values()
                          if lx.derivation and lx.derivation.base_lexeme_id == lexeme.id)
        return family

    def form_of(self, lexeme_id: str) -> LexemeForm:
        return self.forms[lexeme_id]

    # --- internals ----------------------------------------------------------

    def _reserve(self, phonemes: Sequence[str], owner_id: str, kind: str, concept_id: str) -> None:
        try:
            self.registry.reserve(phonemes, owner_id, kind)
        except DuplicateFormError as e:
            owner = self.lexemes.get(e.existing.owner_id)
            raise _fail("DUPLICATE_FORM",
                        f"{''.join(phonemes)!r} for {concept_id!r} is already the word for "
                        f"{owner.concept_id if owner else e.existing.owner_kind!r}",
                        object_id=concept_id,
                        details={"form": "".join(phonemes), "phonemes": list(phonemes),
                                 "existing_concept": owner.concept_id if owner else None}) from e

    def _store(self, lexeme: Lexeme, assembly: AssemblyResult) -> None:
        self.lexemes[lexeme.id] = lexeme
        self.forms[lexeme.id] = LexemeForm(
            lexeme_id=lexeme.id, language_version_id=self.version_id, phonemes=assembly.phonemes,
            orthography=render_phonemes(assembly.phonemes, self.phonology), segments=assembly.segments,
        )
