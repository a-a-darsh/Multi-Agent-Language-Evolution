"""Deterministic half of the design agent: convert a draft and check it.

The LLM proposes; this code decides whether the proposal is usable. Every
problem becomes a ValidationIssue precise enough to send back to the LLM.
"""

import re

from pydantic import ValidationError

from app.agents.design_schema import Critique, LanguageDesignDraft
from app.generation.morphological_engine import MorphologyEngine, MorphologyError, assemble
from app.generation.phoneme_generator import GenerationError, UnsupportedSpellingError, parse_spelling
from app.generation.root_generator import RootRegistry
from app.generation.user_vocabulary import PhonologyRequirements, check_phonology_satisfies
from app.models.design import LanguageDesign
from app.models.generation import GenerationRequest
from app.models.grammar import Grammar, GrammaticalFeature
from app.models.morphology import BoundaryRepair, DerivationalRule, Morpheme, MorphemeType
from app.models.phonology import GRAPHEME_CATALOG, Phoneme, PhonemeClass, Phonology, PositionRestriction
from app.models.validation import Severity, ValidationIssue, ValidationResult

# Enough distinct roots for a ~150-entry dictionary, with room to spare.
MIN_ROOT_CAPACITY = 250
AFFIX_PROBE_ROOTS = 20
# Above this, an inventory is likely "every letter" rather than a design (catalog has 21).
MAX_DESIGNED_CONSONANTS = 14


class DesignBuildError(Exception):
    def __init__(self, result: ValidationResult):
        self.result = result
        super().__init__("; ".join(i.message for i in result.errors))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "feature"


def apply_user_requirements(draft: LanguageDesignDraft,
                            requirements: PhonologyRequirements) -> tuple[LanguageDesignDraft, list[ValidationIssue]]:
    """Deterministically add whatever the user's words need: letters, syllable shapes, hiatus.

    There is only one correct fix when a user word needs /r/ (include /r/), so code
    makes it rather than asking the LLM - in testing, the model kept "fixing" this
    the wrong way, by removing r from its affixes instead.
    """
    if not requirements.words:
        return draft, []
    is_vowel = lambda p: GRAPHEME_CATALOG[p].phoneme_class == PhonemeClass.VOWEL  # noqa: E731
    needed = requirements.required_phonemes
    update = {
        "consonants": draft.consonants + [p for p in needed if not is_vowel(p) and p not in draft.consonants],
        "vowels": draft.vowels + [p for p in needed if is_vowel(p) and p not in draft.vowels],
        "syllable_patterns": draft.syllable_patterns + [
            p for p in requirements.required_patterns if p not in draft.syllable_patterns],
        "allow_hiatus": draft.allow_hiatus or requirements.requires_hiatus,
    }
    changes = [f"{field}: added {sorted(set(update[field]) - set(getattr(draft, field)))}"
               for field in ("consonants", "vowels", "syllable_patterns") if update[field] != getattr(draft, field)]
    if update["allow_hiatus"] != draft.allow_hiatus:
        changes.append("allow_hiatus: set to true")
    if not changes:
        return draft, []
    words = ", ".join(w.spelling for w in requirements.words)
    note = ValidationIssue(code="ADJUSTED_FOR_USER_WORDS", severity=Severity.WARNING,
                           message=f"Adjusted the design so the user's words ({words}) stay legal: {'; '.join(changes)}")
    return draft.model_copy(update=update), [note]


def build_design(draft: LanguageDesignDraft, language_id: str) -> LanguageDesign:
    """Convert the LLM's draft into validated domain models, or raise with every problem found."""
    result = ValidationResult()

    # Duplicates in letter lists are harmless and unambiguous to fix, so dedupe silently.
    consonants = list(dict.fromkeys(draft.consonants))
    vowels = list(dict.fromkeys(draft.vowels))
    common = set(draft.common_sounds)
    if stray := sorted(common - set(consonants) - set(vowels)):
        result.add("COMMON_SOUND_NOT_IN_INVENTORY", f"common_sounds {stray} are not in the inventory; ignored",
                   details={"letters": stray}, severity=Severity.WARNING)

    try:
        phonology = Phonology(
            phonemes=[Phoneme.from_grapheme(g, frequency=2.0 if g in common else 1.0) for g in consonants + vowels],
            syllable_patterns=list(dict.fromkeys(draft.syllable_patterns)),
            restrictions=[PositionRestriction(phoneme_id=r.letter, forbidden_positions=r.forbidden_positions)
                          for r in draft.restrictions],
            allow_hiatus=draft.allow_hiatus,
        )
    except ValidationError as e:
        result.add("PHONOLOGY_INVALID", f"The sound system is inconsistent: {_first_error(e)}")
        raise DesignBuildError(result) from e

    affixes: list[Morpheme] = []
    rules: list[DerivationalRule] = []
    seen_forms: dict[str, str] = {}
    for a in draft.derivational_affixes:
        try:
            # Models often write affixes as "-ka" or "mo-"; the hyphen is notation, not a sound.
            phonemes = parse_spelling(a.form.strip("- "), phonology)
        except UnsupportedSpellingError as e:
            result.add("AFFIX_UNKNOWN_SOUNDS",
                       f"Affix {a.id!r} ({a.form!r}) uses letters not in the inventory: {e.unsupported}",
                       object_id=a.id, details={"form": a.form, "letters": e.unsupported})
            continue
        key = f"{a.position}:{''.join(phonemes)}"
        if a.id in {r.id for r in rules}:
            result.add("DUPLICATE_AFFIX_ID", f"Two affixes share the id {a.id!r}", object_id=a.id)
            continue
        if key in seen_forms:
            result.add("DUPLICATE_AFFIX_FORM",
                       f"Affixes {seen_forms[key]!r} and {a.id!r} are both the {a.position} {a.form!r}",
                       object_id=a.id)
            continue
        seen_forms[key] = a.id
        morpheme = Morpheme(id=f"{language_id}:affix:{a.id}", language_id=language_id, phonemes=phonemes,
                            type=MorphemeType.PREFIX if a.position == "prefix" else MorphemeType.SUFFIX,
                            gloss=a.meaning)
        affixes.append(morpheme)
        rules.append(DerivationalRule(id=a.id, name=a.name, meaning=a.meaning, morpheme_id=morpheme.id,
                                      input_parts_of_speech=a.input_parts_of_speech,
                                      output_part_of_speech=a.output_part_of_speech))

    repair = None
    if draft.linking_vowel or draft.linking_consonant:
        repair = BoundaryRepair(linking_vowel=draft.linking_vowel, linking_consonant=draft.linking_consonant)
        try:
            MorphologyEngine(language_id=language_id, version_id="check", phonology=phonology,
                             registry=RootRegistry(phonology, seed=0), boundary_repair=repair)
        except MorphologyError as e:
            result.extend(e.result)

    if not result.ok:
        raise DesignBuildError(result)

    grammar = Grammar(
        word_order=draft.word_order, morphological_type=draft.morphological_type,
        adposition_type=draft.adposition_type,
        features=[GrammaticalFeature(id=_slug(f.name), name=f.name, description=f.description,
                                     realization=f.realization) for f in draft.features],
    )
    return LanguageDesign(language_id=language_id, name=draft.name, phonology=phonology, grammar=grammar,
                          affixes=affixes, derivational_rules=rules, boundary_repair=repair, summary=draft.summary)


def check_design(design: LanguageDesign, draft: LanguageDesignDraft, request: GenerationRequest,
                 requirements: PhonologyRequirements) -> ValidationResult:
    """Checks that need a built design: user constraints and whether it can actually generate words."""
    result = ValidationResult()
    prefs = request.language_preferences

    # 1. User words are authoritative and must be legal exactly as written.
    result.extend(check_phonology_satisfies(design.phonology, requirements))

    # 2. Explicit answers must be honored.
    for field, wanted, got in [
        ("word_order", prefs.word_order, draft.word_order),
        ("morphological_type", prefs.morphological_type, draft.morphological_type),
        ("grammatical_gender", prefs.grammatical_gender, draft.has_grammatical_gender),
        ("case_marking", prefs.case_marking, draft.has_case_marking),
    ]:
        if wanted is not None and wanted != got:
            result.add("CONSTRAINT_VIOLATED", f"The user chose {field} = {wanted}, but the design has {got}",
                       object_id=field, details={"required": str(wanted), "got": str(got)})
    if prefs.special_features and not draft.features:
        result.add("MISSING_FEATURE",
                   f"The `features` list is empty, but the user asked for: {'; '.join(prefs.special_features)}. "
                   "Add an entry to `features` for each request, saying how the language expresses it "
                   "(e.g. a verb suffix).",
                   object_id="features", details={"requested": prefs.special_features})

    # 3. The sound system must be big enough to make a dictionary's worth of roots.
    try:
        probes = RootRegistry(design.phonology, seed=request.seed, stream="design-check").generate_candidates(
            MIN_ROOT_CAPACITY)
    except GenerationError:
        result.add("INVENTORY_TOO_SMALL",
                   f"These sounds and syllable shapes cannot make {MIN_ROOT_CAPACITY} distinct roots; "
                   "add consonants, vowels, or syllable patterns")
        return result

    # 4. Every affix must attach to real roots. Trying it on sample roots catches
    #    e.g. a vowel-initial suffix in a language with no hiatus and no linking consonant.
    roots = [Morpheme(language_id=design.language_id, phonemes=c.phonemes, type=MorphemeType.ROOT, gloss="probe")
             for c in probes[:AFFIX_PROBE_ROOTS]]
    by_id = {m.id: m for m in design.affixes}
    for rule in design.derivational_rules:
        affix = by_id[rule.morpheme_id]
        ok = 0
        for root in roots:
            pieces = [affix, root] if affix.type == MorphemeType.PREFIX else [root, affix]
            try:
                assemble(pieces, design.phonology, design.boundary_repair)
                ok += 1
            except MorphologyError:
                pass
        if ok == 0:
            result.add("AFFIX_UNUSABLE",
                       f"The {affix.type.value} {''.join(affix.phonemes)!r} ({rule.id}) cannot attach to any "
                       "sample root without breaking the syllable rules. Change its form, allow more syllable "
                       "shapes, or add a linking vowel/consonant",
                       object_id=rule.id, details={"form": "".join(affix.phonemes), "position": affix.type.value})
        elif ok < len(roots) / 2:
            result.add("AFFIX_RARELY_USABLE",
                       f"{''.join(affix.phonemes)!r} ({rule.id}) only attaches to {ok}/{len(roots)} sample roots",
                       object_id=rule.id, severity=Severity.WARNING)
    return result


def check_quality(design: LanguageDesign, requirements: PhonologyRequirements) -> ValidationResult:
    """Deterministic quality checks. Soft: they trigger repair but never fail a design."""
    result = ValidationResult()
    if len(design.phonology.consonants) > MAX_DESIGNED_CONSONANTS:
        result.add("INVENTORY_LARGE",
                   f"{len(design.phonology.consonants)} consonants is close to every available letter. Choose "
                   f"at most {MAX_DESIGNED_CONSONANTS} that give the language a distinctive sound",
                   object_id="consonants", severity=Severity.WARNING)
    # (A "must have a noun -> noun affix" check lived here while parts of speech were enforced;
    # with affixes free to attach to any word it no longer protects anything.)
    words = {w.concept_id: "".join(w.phonemes) for w in requirements.words}
    by_id = {m.id: m for m in design.affixes}
    for rule in design.derivational_rules:
        form = "".join(by_id[rule.morpheme_id].phonemes)
        # Short affixes sharing a syllable with a user word (-va / sava) are unavoidable
        # in CV-heavy languages; only longer overlaps (-mar / mari) look like copying.
        if len(form) < 3:
            continue
        for concept, word in words.items():
            if word.startswith(form) or word.endswith(form):
                result.add("AFFIX_COPIES_USER_WORD",
                           f"The affix {form!r} ({rule.id}) looks like a piece of the user's word {word!r} "
                           f"({concept}); give it a form of its own",
                           object_id=f"affix:{rule.id}", details={"form": form, "word": word},
                           severity=Severity.WARNING)
                break
    return result


def critique_issues(critique: Critique,
                    draft: LanguageDesignDraft) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    """Derive findings from the critic's reviews. The verdict is computed here, not by the model.

    Returns (actionable, advisory). Only narrow, high-precision judgments (per-affix
    classification, inventory) are actionable and trigger repair. Open-ended findings
    are advisory: in testing they were mostly matters of taste, and chasing them made
    designs worse (features ballooned from 4 to 12 across repairs).
    """
    affixes = {a.id: a for a in draft.derivational_affixes}
    issues, advisory = [], []
    for review in critique.affix_reviews:
        affix = affixes.get(review.affix_id)
        if affix is None:
            continue  # the critic named an affix that doesn't exist; ignore rather than mislead the designer
        if review.kind == "inflectional":
            shown = f"{affix.form}-" if affix.position == "prefix" else f"-{affix.form}"
            issues.append(ValidationIssue(
                code="CRITIC_INFLECTIONAL_AFFIX", object_id=f"affix:{affix.id}", severity=Severity.WARNING,
                message=(f"affix:{affix.id} ({shown}, '{affix.meaning}') is inflectional ({review.example}). "
                         "Move it to `features` if the language needs it, and replace it in "
                         "`derivational_affixes` with an affix that creates new words (agent, place, "
                         "instrument, diminutive, quality, collective...).")))
    if not critique.inventory_looks_designed:
        issues.append(ValidationIssue(
            code="CRITIC_UNDESIGNED_INVENTORY", object_id="consonants", severity=Severity.WARNING,
            message="The consonants look like a list of every available letter. Choose a smaller, "
                    "distinctive set that suits this language."))
    for f in critique.other_findings:
        advisory.append(ValidationIssue(
            code=f"CRITIC_{f.problem_type.upper()}", object_id=f.target, severity=Severity.WARNING,
            message=f"{f.target}: {f.explanation} Suggestion: {f.suggestion}"))
    return issues, advisory


def _first_error(e: ValidationError) -> str:
    err = e.errors()[0]
    return str(err.get("ctx", {}).get("error") or err["msg"])
