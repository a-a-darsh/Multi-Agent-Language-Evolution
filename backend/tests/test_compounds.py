"""Compounds (two words joined), no stacked affixes, and no generic affixes for specific meanings."""

from hypothesis import given, settings
from hypothesis import strategies as st

from app.agents.design_validation import generic_affixes
from app.agents.dictionary_schema import CompoundEntry
from app.agents.language_designer import run_language_design
from app.evolution import templates
from app.evolution.lexicon import CoinageError, compound_word, derive_word
from app.evolution.sound_change_engine import apply_sound_change
from app.llm.structured import ScriptedLLM
from app.models.lexicon import LexemeOrigin
from app.services.dictionary_view import DictionaryView
from app.validation.morphology_validator import validate_lexicon
from tests.test_dictionary import CONCEPTS, NOUN_IDS, build, entry, fixes, plan, reviews, run, word
import pytest

from tests.test_language_designer import AffixDraft, POS, good_draft, request


def compound(concept, first, second) -> CompoundEntry:
    return CompoundEntry(concept_id=concept, first_concept_id=first, second_concept_id=second)


def assert_valid(state):
    result = validate_lexicon(state, max_depth=2)
    assert result.ok, [i.message for i in result.errors]


# --- building compounds ------------------------------------------------------------------

def test_a_compound_joins_two_words():
    result = build(compound("harbor", "ship", "sea"))
    harbor, form = word(result, "harbor")
    _, ship = word(result, "ship")
    _, sea = word(result, "sea")
    assert harbor.origin == LexemeOrigin.COMPOUND and not result.failures
    assert form.startswith(ship) and form.endswith(sea[-2:])   # a linking segment may join them
    assert [result.state.lexeme(p).concept_id for p in harbor.compound.part_lexeme_ids] == ["ship", "sea"]
    assert_valid(result.state)


def test_compounds_and_derivations_build_on_each_other():
    result = build(entry("sailor", "sea"),                     # sea + agent
                   compound("merchant", "sailor", "trade"),    # (sea + agent) + trade
                   compound("harbor", "ship", "sea"))
    assert not result.failures
    assert word(result, "merchant")[0].origin == LexemeOrigin.COMPOUND
    assert_valid(result.state)


def test_compound_plan_checks():
    cyc = build(compound("sailor", "fisher", "sea"), compound("fisher", "sailor", "fish"))
    assert {f.code for f in cyc.failures.values()} == {"DERIVATION_CYCLE"}
    same = build(compound("harbor", "ship", "sea"), compound("market", "ship", "sea"))
    assert same.failures["market"].code == "SAME_COMPOSITION"
    assert build(compound("harbor", "sea", "sea")).failures["harbor"].code == "SELF_COMPOUND"
    deep = build(entry("sailor", "sea"), compound("merchant", "sailor", "trade"), compound("market", "merchant", "fish"))
    assert deep.failures["market"].code == "DERIVATION_TOO_DEEP"


def test_grammatical_words_are_not_compound_parts():
    pronouns = [c.id for c in CONCEPTS if c.part_of_speech.value in ("pronoun", "adposition", "particle")]
    assert pronouns, "the fixture needs a grammatical word"
    result = build(compound("market", "trade", pronouns[0]))
    assert result.failures["market"].code == "GRAMMATICAL_COMPOUND_PART"


def test_the_critic_saying_nothing_is_missing_is_not_a_rejection():
    from app.agents.morphological_planner import _names_a_gap
    for text in ["", "none", "None.", "nothing", "nothing missing", "Nothing important missing", "n/a", "-"]:
        assert not _names_a_gap(text), text
    for text in ["a tool; rowing", "hard, internal", "steers the ship"]:
        assert _names_a_gap(text), text
    outcome = run(ScriptedLLM([plan(("sailor", "sea", "agent"))]),
                  ScriptedLLM([reviews(("sailor", True, "nothing missing"))]))
    assert outcome.warnings == [] and word(outcome, "sailor")[0].origin == LexemeOrigin.DERIVED


def test_the_same_affix_is_never_stacked():
    # fisher = fish + agent; then fisher + agent again ("ma-ma-bimaf").
    result = build(entry("fisher", "fish"), entry("sailor", "fisher"))
    assert result.failures["sailor"].code == "AFFIX_REPEATED"
    assert word(result, "sailor")[0].origin == LexemeOrigin.GENERATED_ROOT   # fell back to its own root
    # A different affix on a derived word is still fine.
    assert not build(entry("fisher", "fish"), entry("market", "fisher", "place")).failures


@settings(max_examples=50, deadline=None)
@given(derived=st.lists(st.tuples(st.sampled_from(NOUN_IDS), st.sampled_from(NOUN_IDS),
                                  st.sampled_from(["agent", "place"])), max_size=25),
       compounds=st.lists(st.tuples(st.sampled_from(NOUN_IDS), st.sampled_from(NOUN_IDS),
                                    st.sampled_from(NOUN_IDS)), max_size=25),
       seed=st.integers(0, 500))
def test_any_mix_of_compounds_and_derivations_is_valid(derived, compounds, seed):
    result = build(*[entry(c, b, r) for c, b, r in derived], *[compound(c, a, b) for c, a, b in compounds], seed=seed)
    assert_valid(result.state)
    assert {lx.concept_id for lx in result.state.lexemes} >= {c.id for c in CONCEPTS} - {"water", "hello"}


# --- dictionary view ------------------------------------------------------------------------

def test_a_compound_belongs_to_both_families():
    result = build(compound("harbor", "ship", "sea"))
    view = DictionaryView(result.state)
    harbor = view.detail(word(result, "harbor")[0].id)
    assert [p.concept_id for p in harbor.compound_of] == ["ship", "sea"]
    assert [s.kind for s in harbor.segments if s.kind != "linking"] == ["root", "root"]
    for part in ("ship", "sea"):
        detail = view.detail(word(result, part)[0].id)
        assert "harbor" in [w.concept_id for w in detail.derived_words]
        assert "harbor" in [w.concept_id for w in detail.family]


# --- the planner -----------------------------------------------------------------------------

def test_planner_builds_compounds():
    outcome = run(ScriptedLLM([plan(("sailor", "sea", "agent"), compounds=[("harbor", "ship", "sea")])]), target=2)
    assert outcome.status == "succeeded" and outcome.warnings == []
    assert word(outcome, "harbor")[0].origin == LexemeOrigin.COMPOUND
    assert {type(e).__name__ for e in outcome.plan} == {"PlanEntry", "CompoundEntry"}
    prompt = outcome.llm_calls[0].messages[0]["content"]
    assert "COMPOUND" in prompt and "ma-ma-bimaf" in prompt


def test_critic_rejects_derivations_that_leave_out_what_matters():
    # oar = swim + agent: the critic names what's missing, so it's sent back and becomes a compound.
    llm = ScriptedLLM([plan(("navigator", "sea", "agent")), fixes(compounds=[("navigator", "ship", "sea")])])
    critic = ScriptedLLM([reviews(("navigator", True, "steers the ship, not just any sea person")),
                          reviews(("navigator", True))])
    outcome = run(llm, critic)
    assert outcome.status == "succeeded" and outcome.warnings == []
    assert word(outcome, "navigator")[0].origin == LexemeOrigin.COMPOUND
    repair_prompt = llm.received[1][-1].content
    assert "missing: steers the ship" in repair_prompt and "compound" in repair_prompt
    assert "(compound)" in critic.received[1][-1].content


def test_repeated_affix_failures_go_back_to_the_planner():
    llm = ScriptedLLM([plan(("fisher", "fish", "agent"), ("sailor", "fisher", "agent")),
                       fixes(compounds=[("sailor", "ship", "sea")])])
    outcome = run(llm, target=2)
    assert outcome.status == "succeeded" and outcome.warnings == []
    assert "AFFIX_REPEATED" in llm.received[1][-1].content


# --- design: generic affixes ------------------------------------------------------------------

def test_generic_affix_meanings_are_sent_back():
    generic = good_draft(derivational_affixes=[
        AffixDraft(id="noun_formation", name="Noun formation", meaning="noun formation", position="prefix",
                   form="mo", input_parts_of_speech=[POS.VERB], output_part_of_speech=POS.NOUN),
        *good_draft().derivational_affixes[:2]])
    outcome = run_language_design(request(), ScriptedLLM([generic, good_draft()]))
    assert outcome.status == "succeeded"
    # Soft quality findings come from the critique step (attempt 1), then a repair fixes them.
    assert outcome.attempts[1].issues[0].code == "AFFIX_TOO_GENERIC"
    assert "specific meaning" in outcome.attempts[1].issues[0].message


def test_specific_meanings_are_not_flagged():
    from app.agents.design_validation import build_design
    design = build_design(good_draft(), "lang")
    assert generic_affixes(design) == []


# --- evolved languages -----------------------------------------------------------------------

def test_compounds_evolve_and_can_be_coined_later():
    from tests.test_evolution import ROOTS, build as build_evolving, lexeme, words

    state = build_evolving(ROOTS)
    state = compound_word(state, [lexeme(state, "house").id, lexeme(state, "person").id], "family", "family",
                          "noun")
    assert words(state)["family"].startswith("kasa")
    state = apply_sound_change(state, templates.palatalization(), -700).state
    # kasa + kina -> kasachina: the second part's k palatalized like any other word.
    assert words(state)["family"] == "kasachina"
    state = compound_word(state, [lexeme(state, "now").id, lexeme(state, "person").id], "modern_person", "x", "noun")
    assert words(state)["modern_person"].endswith("china")
    assert validate_lexicon(state).ok
    with pytest.raises(CoinageError, match="already has"):
        state = derive_word(state, lexeme(state, "house").id, "diminutive", "hut", "hut")
        derive_word(state, lexeme(state, "hut").id, "diminutive", "tiny_hut", "tiny hut")


# --- vowel elision at seams ------------------------------------------------------------------

def _no_hiatus_engine(repair=None):
    from app.generation.morphological_engine import assemble  # noqa: F401
    from tests.test_evolution import PHONOLOGY
    return PHONOLOGY.model_copy(update={"allow_hiatus": False})


def _morpheme(text, kind="root"):
    from app.models.morphology import Morpheme, MorphemeType
    return Morpheme(language_id="l", phonemes=list(text), type=MorphemeType(kind), gloss=text)


def test_affixes_drop_a_vowel_only_as_a_last_resort():
    from app.generation.morphological_engine import assemble
    from app.models.morphology import BoundaryRepair
    ph = _no_hiatus_engine()
    root, suffix = _morpheme("kasa"), _morpheme("ik", "suffix")
    built = assemble([root, suffix], ph)
    assert "".join(built.phonemes) == "kasik"
    assert [("".join(s.phonemes), s.morpheme_id) for s in built.segments] == [("kas", root.id), ("ik", suffix.id)]
    # The design's linking consonant comes first: affixes stay recognizable.
    assert "".join(assemble([root, suffix], ph, BoundaryRepair(linking_consonant="r")).phonemes) == "kasarik"
    # A one-vowel piece can't vanish: the other vowel goes instead (a + ika -> aka).
    assert "".join(assemble([_morpheme("a", "prefix"), _morpheme("ika", "suffix")], ph).phonemes) == "aka"


def test_compound_seams_merge_vowels_by_default():
    from app.generation.morphological_engine import assemble
    from app.models.morphology import BoundaryRepair
    ph = _no_hiatus_engine()
    tribe, house = _morpheme("tikti"), _morpheme("opip")
    # Even with a linking consonant available, a compound seam elides.
    for repair in (None, BoundaryRepair(linking_consonant="r")):
        assert "".join(assemble([tribe, house], ph, repair, compound_seams=[1]).phonemes) == "tiktopip"
    # And even where adjacent vowels would be legal.
    loose = ph.model_copy(update={"allow_hiatus": True})
    assert "".join(assemble([tribe, house], loose, compound_seams=[1]).phonemes) == "tiktopip"
    assert "".join(assemble([tribe, house], loose).phonemes) == "tiktiopip"   # not a compound seam


def test_compounds_with_vowel_seams_build_and_replay():
    from app.generation.morphological_engine import MorphologyEngine
    from app.generation.root_generator import RootRegistry
    from app.models.lexicon import PartOfSpeech as P
    ph = _no_hiatus_engine().model_copy(update={"allow_hiatus": True})
    engine = MorphologyEngine(language_id="l", version_id="v", phonology=ph, registry=RootRegistry(ph, seed=1))
    tribe = engine.add_root("tribe", "tribe", P.NOUN, list("tikti"))
    house = engine.add_root("house", "house", P.NOUN, list("opip"))
    council = engine.compound([tribe.id, house.id], "council", "council", P.NOUN)
    assert engine.form_of(council.id).orthography == "tiktopip"


def test_restrictions_on_absent_sounds_are_dropped_not_fatal():
    from app.agents.design_schema import RestrictionDraft
    from app.agents.design_validation import build_design
    draft = good_draft(restrictions=[RestrictionDraft(letter="q", forbidden_positions=["coda"])])
    design = build_design(draft, "lang")   # q isn't in the inventory: no error
    assert design.phonology.restrictions == []


def test_unambiguous_design_fixes_are_made_by_code():
    from app.agents.design_schema import RestrictionDraft
    from app.agents.design_validation import apply_user_requirements
    from app.generation.user_vocabulary import analyze_user_words
    reqs, _ = analyze_user_words({"water": "mari"})
    draft = good_draft(consonants=[c for c in good_draft().consonants if c != "k"],   # but the agent suffix is -ka
                       restrictions=[RestrictionDraft(letter="m", forbidden_positions=["word_initial"])])
    fixed, notes = apply_user_requirements(draft, reqs)
    assert "k" in fixed.consonants and fixed.restrictions == []
    assert {n.code for n in notes} >= {"ADDED_AFFIX_SOUNDS", "DROPPED_RESTRICTION_FOR_USER_WORDS"}


# --- rules from the Amina review (black = night + one, buy = trade + person) --------------------

def test_basic_vocabulary_is_never_a_compound_but_may_take_a_fitting_affix():
    result = build(compound("black", "night", "fire"), entry("enemy", "fight"))
    assert word(result, "black")[0].origin == LexemeOrigin.GENERATED_ROOT
    assert "BASIC_WORD_KEPT_AS_ROOT" in {n.code for n in result.notes} and "black" not in result.failures
    assert word(result, "enemy")[0].origin == LexemeOrigin.DERIVED   # fight + agent is fine


def test_numbers_and_quantities_are_not_compound_parts():
    result = build(compound("harbor", "ship", "one"), compound("market", "house", "many"))
    assert result.failures["harbor"].code == result.failures["market"].code == "GRAMMATICAL_COMPOUND_PART"


def test_an_affix_must_make_what_the_concept_is():
    # The agent affix makes a person (a noun); "buy" is a verb. (Amina had buy = trade + person.)
    result = build(entry("buy", "trade"))
    assert result.failures["buy"].code == "AFFIX_WRONG_RESULT"
    assert "makes a noun" in result.failures["buy"].message


def test_a_word_used_as_filler_in_many_compounds_is_stopped():
    from app.generation.dictionary_builder import MAX_COMPOUNDS_PER_PART, is_basic_vocabulary
    cultural = [c.id for c in CONCEPTS if c.part_of_speech.value == "noun" and not is_basic_vocabulary(c)]
    others = [c.id for c in CONCEPTS if c.part_of_speech.value == "noun" and is_basic_vocabulary(c)
              and c.id != "sea"]
    count = MAX_COMPOUNDS_PER_PART + 2
    result = build(*[compound(t, "sea", o) for t, o in zip(cultural[:count], others)])
    over = [c for c, f in result.failures.items() if f.code == "OVERUSED_PART"]
    assert len(over) == 2


def test_planner_offers_basic_vocabulary_for_derivation_only():
    from app.agents.morphological_planner import _compoundable, _derivable
    state = {"concepts": CONCEPTS, "user_words": []}
    assert "black" in _derivable(state) and "black" not in _compoundable(state)
    assert "forge" not in {c.id for c in CONCEPTS} or "forge" in _compoundable(state)
    assert "goat" not in {c.id for c in CONCEPTS} or "goat" in _compoundable(state)


def test_repair_lists_are_bounded():
    from app.agents.dictionary_schema import repair_schema
    from pydantic import ValidationError
    schema = repair_schema(["sailor"], [c.id for c in CONCEPTS], ["agent"])
    fix = {"concept_id": "sailor", "action": "independent_root", "base_concept_id": None,
           "second_base_concept_id": None, "rule_id": None}
    with pytest.raises(ValidationError):
        schema.model_validate({"fixes": [fix] * 3})
