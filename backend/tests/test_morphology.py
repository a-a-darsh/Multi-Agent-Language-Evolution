import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.generation.morphological_engine import MorphologyEngine, MorphologyError
from app.generation.root_generator import RootRegistry
from app.models.civilization import Civilization
from app.models.grammar import Grammar, MorphologicalType, WordOrder
from app.models.language import Language, LanguageState, LanguageVersion
from app.models.lexicon import PartOfSpeech as POS
from app.models.morphology import BoundaryRepair, DerivationalRule, Morpheme, MorphemeType
from app.validation.morphology_validator import validate_lexicon
from tests.factories import sample_language, sample_phonology

LANG = "lang"


def affix(form: str, gloss: str, kind=MorphemeType.SUFFIX) -> Morpheme:
    return Morpheme(id=f"aff_{gloss}", language_id=LANG, phonemes=list(form), type=kind, gloss=gloss)


AGENT = affix("ka", "agent")
PLACE = affix("ri", "place")
NEG = affix("mo", "negative", MorphemeType.PREFIX)
PLURAL_S = affix("s", "plural")
SUFFIX_A = affix("a", "adjectivizer")

RULES = [
    DerivationalRule(id="agent", name="Agent", meaning="person associated with", morpheme_id=AGENT.id,
                     input_parts_of_speech=[POS.NOUN, POS.VERB], output_part_of_speech=POS.NOUN),
    DerivationalRule(id="place", name="Place", meaning="place of", morpheme_id=PLACE.id,
                     input_parts_of_speech=[POS.NOUN], output_part_of_speech=POS.NOUN),
    DerivationalRule(id="negative", name="Negative", meaning="not", morpheme_id=NEG.id,
                     input_parts_of_speech=[POS.ADJECTIVE], output_part_of_speech=POS.ADJECTIVE),
    DerivationalRule(id="plural", name="Collective", meaning="group of", morpheme_id=PLURAL_S.id,
                     input_parts_of_speech=[POS.NOUN], output_part_of_speech=POS.NOUN),
    DerivationalRule(id="adj", name="Adjective", meaning="having the quality of", morpheme_id=SUFFIX_A.id,
                     input_parts_of_speech=[POS.NOUN], output_part_of_speech=POS.ADJECTIVE),
]


def engine(repair: BoundaryRepair | None = None, phonology=None, **kwargs) -> MorphologyEngine:
    phonology = phonology or sample_phonology()
    return MorphologyEngine(language_id=LANG, version_id="v1", phonology=phonology,
                            registry=RootRegistry(phonology, seed=1),
                            affixes=[AGENT, PLACE, NEG, PLURAL_S, SUFFIX_A], rules=RULES,
                            boundary_repair=repair, **kwargs)


def to_state(e: MorphologyEngine) -> LanguageState:
    civ = Civilization(name="C", founding_year=0, geography="coastal", economy="mixed")
    return LanguageState(
        civilization=civ,
        language=Language(id=LANG, civilization_id=civ.id, name="L", founding_year=0),
        version=LanguageVersion(id="v1", language_id=LANG, valid_from_year=0, phonology=e.phonology,
                                grammar=Grammar(word_order=WordOrder.SOV,
                                                morphological_type=MorphologicalType.AGGLUTINATIVE),
                                boundary_repair=e.boundary_repair),
        morphemes=list(e.morphemes.values()), derivational_rules=list(e.rules.values()),
        lexemes=list(e.lexemes.values()), forms=list(e.forms.values()),
    )


def code_of(exc: pytest.ExceptionInfo) -> str:
    return exc.value.result.errors[0].code


def test_sea_to_sailor():
    e = engine()
    sea = e.add_root("sea", "sea", POS.NOUN, list("nal"))
    sailor = e.derive(sea.id, "agent", "sailor", "sailor")

    form = e.form_of(sailor.id)
    assert form.orthography == "nalka"
    assert [("".join(s.phonemes), s.morpheme_id) for s in form.segments] == [
        ("nal", sea.morpheme_ids[0]), ("ka", AGENT.id)]
    assert sailor.derivation.base_lexeme_id == sea.id
    assert sailor.derivation.applied_morpheme_ids == [AGENT.id]
    assert validate_lexicon(to_state(e)).ok


def test_prefix_goes_first():
    e = engine()
    good = e.add_root("good", "good", POS.ADJECTIVE, list("sel"))
    assert e.form_of(e.derive(good.id, "negative", "bad", "bad").id).orthography == "mosel"


def test_any_affix_attaches_and_word_takes_given_part_of_speech():
    e = engine()
    water = e.add_root("water", "water", POS.NOUN, list("mari"))
    wave = e.derive(water.id, "negative", "wave", "wave", part_of_speech=POS.NOUN)  # negative usually wants adjectives
    assert wave.part_of_speech == POS.NOUN and e.form_of(wave.id).orthography == "momari"
    assert validate_lexicon(to_state(e)).ok


def test_unknown_rule_lists_alternatives():
    e = engine()
    sea = e.add_root("sea", "sea", POS.NOUN, list("nal"))
    with pytest.raises(MorphologyError) as exc:
        e.derive(sea.id, "diminutive", "pond", "pond")
    issue = exc.value.result.errors[0]
    assert issue.code == "UNKNOWN_RULE" and "agent" in issue.details["available_rules"]


def test_derivation_depth_limit():
    e = engine(max_depth=2)
    sea = e.add_root("sea", "sea", POS.NOUN, list("nal"))
    sailor = e.derive(sea.id, "agent", "sailor", "sailor")
    crew = e.derive(sailor.id, "place", "sailors_quarter", "sailors' quarter")
    assert e.depth(crew.id) == 2
    with pytest.raises(MorphologyError) as exc:
        e.derive(crew.id, "agent", "quartermaster", "quartermaster")
    assert code_of(exc) == "DERIVATION_TOO_DEEP"


def test_derived_form_cannot_duplicate_existing_word():
    e = engine()
    e.add_root("boat", "boat", POS.NOUN, list("nalka"))
    sea = e.add_root("sea", "sea", POS.NOUN, list("nal"))
    with pytest.raises(MorphologyError) as exc:
        e.derive(sea.id, "agent", "sailor", "sailor")
    issue = exc.value.result.errors[0]
    assert issue.code == "DUPLICATE_FORM" and issue.details["existing_concept"] == "boat"


def test_illegal_junction_rejected_without_repair():
    e = engine()
    sea = e.add_root("sea", "sea", POS.NOUN, list("nal"))
    with pytest.raises(MorphologyError) as exc:
        e.derive(sea.id, "plural", "seas", "seas")  # nal + s = "nals": CVCC is not allowed
    issue = exc.value.result.errors[0]
    assert issue.code == "ILLEGAL_DERIVED_FORM"
    assert issue.details["form"] == "nals" and issue.details["pieces"] == ["nal", "s"]
    assert "sea" not in {lx.concept_id for lx in e.lexemes.values() if lx.derivation}


def test_linking_vowel_repairs_consonant_clash():
    e = engine(BoundaryRepair(linking_vowel="i"))
    sea = e.add_root("sea", "sea", POS.NOUN, list("nal"))
    form = e.form_of(e.derive(sea.id, "plural", "seas", "seas").id)
    assert form.orthography == "nalis"
    assert [s.morpheme_id is None for s in form.segments] == [False, True, False]
    assert validate_lexicon(to_state(e)).ok


def test_repair_only_where_needed():
    e = engine(BoundaryRepair(linking_vowel="i"))
    sea = e.add_root("sea", "sea", POS.NOUN, list("nal"))
    assert e.form_of(e.derive(sea.id, "agent", "sailor", "sailor").id).orthography == "nalka"


def test_linking_consonant_repairs_hiatus():
    e = engine(BoundaryRepair(linking_consonant="n"))
    hello = e.add_root("water", "water", POS.NOUN, list("sava"))
    assert e.form_of(e.derive(hello.id, "adj", "wet", "wet").id).orthography == "savana"


def test_invalid_engine_configuration():
    with pytest.raises(MorphologyError) as exc:
        engine(BoundaryRepair(linking_vowel="k"))
    assert code_of(exc) == "INVALID_REPAIR"
    root = Morpheme(language_id=LANG, phonemes=list("nal"), type=MorphemeType.ROOT, gloss="sea")
    with pytest.raises(MorphologyError) as exc:
        MorphologyEngine(language_id=LANG, version_id="v1", phonology=sample_phonology(),
                         registry=RootRegistry(sample_phonology(), seed=1), affixes=[root])
    assert code_of(exc) == "INVALID_AFFIX"


def test_generated_root_leaves_candidate_pool():
    e = engine()
    [candidate] = e.registry.generate_candidates(1)
    lexeme = e.add_generated_root("stone", "stone", POS.NOUN, candidate.id)
    assert e.form_of(lexeme.id).phonemes == candidate.phonemes
    assert e.registry.unused_candidates == []


def test_word_family():
    e = engine()
    sea = e.add_root("sea", "sea", POS.NOUN, list("nal"))
    sailor = e.derive(sea.id, "agent", "sailor", "sailor")
    e.derive(sea.id, "place", "harbor", "harbor")
    e.derive(sailor.id, "place", "sailors_quarter", "sailors' quarter")
    e.add_root("stone", "stone", POS.NOUN, list("tok"))
    assert [lx.concept_id for lx in e.word_family(sea.id)] == ["sea", "sailor", "harbor", "sailors_quarter"]


# --- whole-lexicon validator ---------------------------------------------------

def test_sample_language_is_valid():
    assert validate_lexicon(sample_language()).ok


def test_validator_catches_form_not_built_from_morphemes():
    state = sample_language()
    form = state.form_of("lex_sailor")
    form.phonemes, form.orthography = list("nalko"), "nalko"
    assert "FORM_MISMATCH" in {i.code for i in validate_lexicon(state).errors}


def test_validator_catches_cycles():
    state = sample_language()
    sea = next(lx for lx in state.lexemes if lx.concept_id == "sea")
    sailor = state.lexeme("lex_sailor")
    # Make sea "derived from" sailor, closing a loop.
    sea.origin = sailor.origin
    sea.derivation = sailor.derivation.model_copy(update={"output_lexeme_id": sea.id, "base_lexeme_id": sailor.id})
    codes = {i.code for i in validate_lexicon(state).errors}
    assert "DERIVATION_CYCLE" in codes


def test_validator_catches_duplicates_and_bad_spelling():
    state = sample_language()
    hello_form = next(f for f in state.forms if f.orthography == "sava")
    hello_form.phonemes = list("nal")
    codes = {i.code for i in validate_lexicon(state).errors}
    assert {"DUPLICATE_FORM", "ORTHOGRAPHY_MISMATCH", "FORM_MISMATCH"} <= codes


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10_000), plan=st.lists(st.tuples(st.integers(0, 39), st.sampled_from(RULES)), max_size=40),
       repair=st.booleans())
def test_random_lexicons_always_validate(seed, plan, repair):
    """Whatever the engine accepts, the independent validator must also accept."""
    e = engine(BoundaryRepair(linking_vowel="i", linking_consonant="n") if repair else None)
    candidates = e.registry.generate_candidates(40)
    for i, c in enumerate(candidates):
        e.add_generated_root(f"c{i}", f"concept {i}", [POS.NOUN, POS.VERB, POS.ADJECTIVE][i % 3], c.id)
    roots = list(e.lexemes.values())
    for n, (index, rule) in enumerate(plan):
        try:
            e.derive(roots[index].id, rule.id, f"d{n}", f"derived {n}")
        except MorphologyError:
            pass  # rejections are fine; accepted words must be valid
    assert validate_lexicon(to_state(e), max_depth=e.max_depth).errors == []
