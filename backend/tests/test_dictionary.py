import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.agents.design_validation import build_design
from app.agents.dictionary_schema import PlanEntry, SenseExtension, plan_schema, repair_schema
from app.agents.morphological_planner import run_morphological_planning
from app.agents.pipeline import AgentLLMs, run_generation
from app.data import load_semantic_inventory
from app.generation.concept_selection import select_concepts
from app.generation.dictionary_builder import build_dictionary, concepts_with_words
from app.generation.user_vocabulary import analyze_user_words
from app.llm.structured import LLMCallError, ScriptedLLM
from app.models.civilization import Civilization
from app.models.generation import GenerationRequest
from app.models.lexicon import LexemeOrigin
from app.validation.morphology_validator import validate_lexicon
from tests.test_language_designer import good_draft

INVENTORY = load_semantic_inventory()
CIV = Civilization(name="Savari", founding_year=-1000, geography="coastal", economy="mercantile",
                   social_organization="city_states", cultural_traits=["seafaring"])
DESIGN = build_design(good_draft(), "lang")
USER_WORDS = analyze_user_words({"hello": "sava", "water": "mari"})[0].words
CONCEPTS = select_concepts(INVENTORY, CIV, [w.concept_id for w in USER_WORDS])


def entry(concept, base, rule="agent") -> PlanEntry:
    return PlanEntry(concept_id=concept, base_concept_id=base, rule_id=rule)


def build(*entries, user_words=USER_WORDS, seed=7, extensions=()):
    concepts = select_concepts(INVENTORY, CIV, [w.concept_id for w in user_words])
    return build_dictionary(DESIGN, CIV, concepts, user_words, entries, seed, extensions=extensions)


def word(result, concept_id: str):
    lexeme = next(lx for lx in result.state.lexemes if lx.concept_id == concept_id)
    return lexeme, result.state.form_of(lexeme.id).orthography


# --- concept selection -----------------------------------------------------------

def test_culture_adds_vocabulary_without_removing_core():
    ids = {c.id for c in CONCEPTS}
    assert {"harbor", "ship", "merchant", "market", "sailor"} <= ids
    assert "camel" not in ids and "herd" not in ids
    core = {c.id for c in INVENTORY if c.priority.value == "core"}
    assert core <= ids


def test_era_adds_its_vocabulary_and_excludes_anachronisms():
    bronze = select_concepts(INVENTORY, CIV.model_copy(update={"era": "bronze_age"}))
    ids = {c.id for c in bronze}
    assert {"copper", "tin", "bronze", "chariot"} <= ids
    assert not {"coin", "iron", "gunpowder", "castle", "printing_press"} & ids  # not invented yet

    medieval = {c.id for c in select_concepts(INVENTORY, CIV.model_copy(update={"era": "medieval"}))}
    assert {"castle", "guild", "coin"} <= medieval and "printing_press" not in medieval
    assert "chariot" not in medieval  # exists, but no longer characteristic of the era


def test_user_word_kept_even_if_anachronistic():
    ids = {c.id for c in select_concepts(INVENTORY, CIV.model_copy(update={"era": "stone_age"}), ["gunpowder"])}
    assert "gunpowder" in ids


def test_required_concepts_always_included():
    ids = {c.id for c in select_concepts(INVENTORY, CIV, ["camel"])}
    assert "camel" in ids


def test_cultural_cap():
    assert sum(c.priority.value == "cultural" for c in select_concepts(INVENTORY, CIV, max_cultural=5)) == 5


# --- builder -----------------------------------------------------------------------

def test_empty_plan_gives_every_concept_a_root():
    result = build()
    assert len(result.state.lexemes) == len(CONCEPTS) >= 150
    assert validate_lexicon(result.state).ok
    assert not result.failures


def test_derivation_builds_on_base_root():
    result = build(entry("sailor", "sea"), entry("merchant", "trade"), entry("market", "merchant", "place"))
    _, sea = word(result, "sea")
    sailor_lexeme, sailor = word(result, "sailor")
    assert sailor == sea + "ka" or sailor == sea + "ika"  # linking vowel only if needed
    assert sailor_lexeme.origin == LexemeOrigin.DERIVED
    _, merchant = word(result, "merchant")
    assert word(result, "market")[1].startswith(merchant)  # depth-2 chain
    assert validate_lexicon(result.state, max_depth=2).ok


def test_user_words_kept_exactly_and_never_derived():
    result = build(entry("water", "sea", "place"))
    assert word(result, "water")[1] == "mari" and word(result, "hello")[1] == "sava"
    assert [n.code for n in result.notes] == ["USER_WORD_KEPT"]


def test_user_word_can_be_a_base():
    _, sailor = word(build(entry("sailor", "water")), "sailor")
    assert sailor.startswith("mari")


@pytest.mark.parametrize("concept, base, rule", [
    ("wave", "water", "agent"),     # the user's example shape: a noun from a noun via any affix
    ("merchant", "buy", "agent"),   # agent usually takes nouns; a verb works too
    ("sailor", "good", "place"),    # place usually takes nouns; any word works
])
def test_any_affix_attaches_to_any_word(concept, base, rule):
    """Parts of speech guide the planner but never block a derivation (it blocked wave = water + movement)."""
    result = build(entry(concept, base, rule))
    assert not result.failures
    lexeme, form = word(result, concept)
    assert lexeme.origin == LexemeOrigin.DERIVED
    assert lexeme.part_of_speech == next(c for c in CONCEPTS if c.id == concept).part_of_speech
    assert form.startswith(word(result, base)[1])
    assert validate_lexicon(result.state).ok


@pytest.mark.parametrize("bad, code", [
    (entry("sailor", "sailor"), "SELF_DERIVATION"),
])
def test_invalid_entries_fall_back_to_roots(bad, code):
    result = build(bad)
    assert result.failures[bad.concept_id].code == code
    lexeme, _ = word(result, bad.concept_id)
    assert lexeme.origin == LexemeOrigin.GENERATED_ROOT
    assert validate_lexicon(result.state).ok


def test_same_composition_detected_before_building():
    result = build(entry("merchant", "trade"), entry("market", "trade"))
    assert result.failures["market"].code == "SAME_COMPOSITION"
    assert word(result, "merchant")[0].origin == LexemeOrigin.DERIVED


def test_critic_still_reviews_when_budget_is_spent():
    critic = ScriptedLLM([reviews(("sailor", False))])
    outcome = run(ScriptedLLM([plan(("sailor", "sea", "agent"), *CYCLE), fixes(*CYCLE), fixes(*CYCLE)]), critic)
    assert len(critic.received) == 1
    # The only derivation was demoted, so the 1-word target is reported as missed too.
    assert {w.code for w in outcome.warnings} == {"DERIVATION_CYCLE", "IMPLAUSIBLE_DERIVATION", "DERIVATION_TARGET_MISSED"}
    # With no repairs left, the rejected derivation is demoted to an independent root.
    assert word(outcome, "sailor")[0].origin == LexemeOrigin.GENERATED_ROOT
    assert validate_lexicon(outcome.state).ok


def shared_word(concept, base):
    return SenseExtension(concept_id=concept, base_concept_id=base, explanation="linked")


def test_shared_word_is_one_entry_with_two_meanings():
    result = build(entry("merchant", "trade"), extensions=[shared_word("coin", "trade"), shared_word("price", "merchant")])
    trade, trade_word = word(result, "trade")
    assert [s.concept_id for s in trade.senses] == ["trade", "coin"]
    merchant, _ = word(result, "merchant")  # a derived word can carry a second meaning too
    assert [s.concept_id for s in merchant.senses] == ["merchant", "price"]
    assert not any(lx.concept_id in ("coin", "price") for lx in result.state.lexemes)  # no separate entries
    assert concepts_with_words(result.state) == {c.id for c in CONCEPTS}  # every meaning still expressed
    assert validate_lexicon(result.state).ok


def test_self_shared_word_gets_its_own_root():
    result = build(extensions=[shared_word("sea", "sea")])
    assert result.failures["sea"].code == "SELF_DERIVATION"
    assert word(result, "sea")[0].origin == LexemeOrigin.GENERATED_ROOT


def test_shared_word_across_parts_of_speech_is_allowed():
    result = build(extensions=[shared_word("burn", "fire")])  # one word for "fire" and "to burn"
    fire, _ = word(result, "fire")
    assert [s.concept_id for s in fire.senses] == ["fire", "burn"]


def test_shared_word_cannot_borrow_a_borrowed_word():
    result = build(extensions=[shared_word("coin", "trade"), shared_word("price", "coin")])
    assert result.failures["price"].code == "EXTENSION_CHAIN"


def test_same_user_word_for_two_meanings_is_one_word():
    words = analyze_user_words({"water": "kimur", "sea": "kimur"})[0].words
    result = build(user_words=words)
    lexeme, form = word(result, "water")
    assert form == "kimur" and [s.concept_id for s in lexeme.senses] == ["water", "sea"]
    assert [n.code for n in result.notes] == ["USER_POLYSEMY"]


def test_derivation_cannot_build_on_a_shared_word():
    # Regression from a live run: sky shared god's word, then a word was built on sky.
    result = build(entry("sailor", "coin"), extensions=[shared_word("coin", "trade")])
    assert result.failures["sailor"].code == "BASE_IS_SHARED_WORD"
    assert validate_lexicon(result.state).ok


NOUN_IDS = [c.id for c in CONCEPTS if c.part_of_speech.value == "noun"]


@settings(max_examples=60, deadline=None)
@given(pairs=st.lists(st.tuples(st.sampled_from(NOUN_IDS), st.sampled_from(NOUN_IDS),
                                st.sampled_from(["agent", "place"])), max_size=40),
       shares=st.lists(st.tuples(st.sampled_from(NOUN_IDS), st.sampled_from(NOUN_IDS)), max_size=15),
       seed=st.integers(0, 1000))
def test_any_mix_of_derivations_and_shared_words_is_valid(pairs, shares, seed):
    """Derivations and shared words were only tested separately; their interaction crashed live."""
    result = build(*[entry(c, b, r) for c, b, r in pairs], seed=seed,
                   extensions=[shared_word(c, b) for c, b in shares])
    assert validate_lexicon(result.state, max_depth=2).errors == []
    assert concepts_with_words(result.state) == {c.id for c in CONCEPTS}


def test_planner_shared_words_and_critic():
    llm = ScriptedLLM([plan(("sailor", "sea", "agent"), shared=[("coin", "trade"), ("price", "fish")]),
                       fixes(roots=["price"])])
    critic = ScriptedLLM([reviews(("sailor", True), ("coin", True), ("price", False)),
                          reviews(("sailor", True), ("coin", True))])
    outcome = run(llm, critic)
    assert outcome.status == "succeeded" and outcome.warnings == []
    trade, _ = word(outcome, "trade")
    assert [s.concept_id for s in trade.senses] == ["trade", "coin"]
    assert word(outcome, "price")[0].origin == LexemeOrigin.GENERATED_ROOT
    assert "same word as trade" in critic.received[0][-1].content


def test_cycles_and_depth_detected():
    cyc = build(entry("sailor", "fisher"), entry("fisher", "sailor"))
    assert {f.code for f in cyc.failures.values()} == {"DERIVATION_CYCLE"}
    deep = build(entry("merchant", "trade"), entry("market", "merchant", "place"),
                 entry("sailor", "market"))  # 3 steps
    assert deep.failures["sailor"].code == "DERIVATION_TOO_DEEP"


def test_building_is_deterministic():
    plan = [entry("sailor", "sea"), entry("fisher", "fish")]
    a, b = build(*plan), build(*plan)
    forms = lambda r: sorted((lx.concept_id, r.state.form_of(lx.id).orthography) for lx in r.state.lexemes)  # noqa: E731
    assert forms(a) == forms(b)


NOUNS = [c.id for c in CONCEPTS if c.part_of_speech.value == "noun"]


@settings(max_examples=40, deadline=None)
@given(pairs=st.lists(st.tuples(st.sampled_from(NOUNS), st.sampled_from(NOUNS),
                                st.sampled_from(["agent", "place", "negative"])), max_size=60),
       seed=st.integers(0, 1000))
def test_any_plan_yields_a_complete_valid_dictionary(pairs, seed):
    result = build(*[entry(c, b, r) for c, b, r in pairs], seed=seed)
    assert validate_lexicon(result.state, max_depth=2).errors == []
    assert {lx.concept_id for lx in result.state.lexemes} == {c.id for c in CONCEPTS}
    for lx in result.state.lexemes:  # every derived word really follows its recorded derivation
        if lx.derivation:
            base = next(b for b in result.state.lexemes if b.id == lx.derivation.base_lexeme_id)
            assert result.state.form_of(lx.id).orthography.startswith(result.state.form_of(base.id).orthography) \
                or lx.derivation.rule_id == "negative"


# --- dynamic schemas ---------------------------------------------------------------

CONCEPT_IDS = [c.id for c in CONCEPTS]
RULE_IDS = [r.id for r in DESIGN.derivational_rules]
PLAN_SCHEMA = plan_schema([c for c in CONCEPT_IDS if c not in {"water", "hello"}], CONCEPT_IDS, RULE_IDS)


def derivation(concept, base, rule="agent") -> dict:
    return {"concept_id": concept, "base_concept_id": base, "rule_id": rule, "explanation": "x"}


@pytest.mark.parametrize("bad", [
    derivation("spaceship", "sea"),        # not in this dictionary
    derivation("sailor", "sea", "magic"),  # not one of this language's rules
    derivation("water", "sea", "place"),   # user word: cannot be derived
])
def test_plan_schema_only_accepts_valid_ids(bad):
    PLAN_SCHEMA.model_validate({"reasoning": "r", "compounds": [], "derivations": [derivation("sailor", "sea")],
                                "extended_senses": []})
    with pytest.raises(ValidationError):
        PLAN_SCHEMA.model_validate({"reasoning": "r", "compounds": [], "derivations": [bad], "extended_senses": []})


def test_repair_schema_only_accepts_failing_ids():
    schema = repair_schema(["sailor"], CONCEPT_IDS, RULE_IDS)
    with pytest.raises(ValidationError):
        schema.model_validate({"fixes": [{"concept_id": "fish", "action": "independent_root",
                                          "base_concept_id": None, "second_base_concept_id": None,
                                          "rule_id": None}]})


# --- planner graph -------------------------------------------------------------------

def plan(*derivations, shared=(), compounds=()) -> dict:
    return {"reasoning": "Maritime families.",
            "compounds": [{"concept_id": c, "first_concept_id": a, "second_concept_id": b, "explanation": "x"}
                          for c, a, b in compounds],
            "derivations": [derivation(c, b, r) for c, b, r in derivations],
            "extended_senses": [{"concept_id": c, "base_concept_id": b, "explanation": "linked"} for c, b in shared]}


def fixes(*derivations, roots=(), compounds=()) -> dict:
    return {"fixes": [{"concept_id": c, "action": "derive", "base_concept_id": b, "second_base_concept_id": None,
                       "rule_id": r} for c, b, r in derivations] +
                     [{"concept_id": c, "action": "compound", "base_concept_id": a, "second_base_concept_id": b,
                       "rule_id": None} for c, a, b in compounds] +
                     [{"concept_id": c, "action": "independent_root", "base_concept_id": None,
                       "second_base_concept_id": None, "rule_id": None} for c in roots]}


def reviews(*items) -> dict:
    """(concept, plausible) or (concept, plausible, missing)."""
    return {"reviews": [{"concept_id": i[0], "literal_meaning": "m", "plausible": i[1],
                         "missing": i[2] if len(i) > 2 else ""} for i in items]}


def run(llm, critic=None, target=1):
    # An explicit small target keeps these tests about repair/critique, not expansion.
    return run_morphological_planning(DESIGN, CIV, USER_WORDS, 7, llm, critic, derived_ratio=target / len(CONCEPTS))


def test_planner_happy_path():
    outcome = run(ScriptedLLM([plan(("sailor", "sea", "agent"), ("harbor", "ship", "place"))]))
    assert outcome.status == "succeeded" and outcome.warnings == []
    assert word(outcome, "harbor")[0].origin == LexemeOrigin.DERIVED
    prompt = outcome.llm_calls[0].messages[1]["content"]
    assert "sailor (noun): sailor" in prompt and "water (noun): water (user word)" in prompt and "-ka" in prompt


# Cycles can't be prevented by the schema, so they exercise the repair loop.
# navigator = fisher + place (not + agent again: fish+agent+agent would stack the same affix).
CYCLE = (("navigator", "fisher", "place"), ("fisher", "navigator", "agent"))


def test_partial_repair_only_sends_failures():
    llm = ScriptedLLM([plan(("sailor", "sea", "agent"), *CYCLE), fixes(("fisher", "fish", "agent"))])
    outcome = run(llm)
    assert outcome.status == "succeeded" and outcome.warnings == []
    repair_prompt = llm.received[1][-1].content
    assert "DERIVATION_CYCLE" in repair_prompt and "sailor =" not in repair_prompt
    _, fish = word(outcome, "fish")
    assert word(outcome, "navigator")[1].startswith(word(outcome, "fisher")[1]) and \
        word(outcome, "fisher")[1].startswith(fish)


def test_repair_can_choose_independent_root():
    llm = ScriptedLLM([plan(("sailor", "sea", "agent"), *CYCLE), fixes(roots=["fisher", "navigator"])])
    outcome = run(llm)
    assert outcome.status == "succeeded" and outcome.warnings == []
    assert {e.concept_id for e in outcome.plan} == {"sailor"}


def test_planner_prompt_presents_parts_of_speech_as_guidance():
    outcome = run(ScriptedLLM([plan(("sailor", "sea", "agent"))]))
    prompt = outcome.llm_calls[0].messages[1]["content"]
    assert "Usually on a noun/verb, usually makes a noun" in prompt


def test_unrepaired_failures_become_warnings():
    outcome = run(ScriptedLLM([plan(("sailor", "sea", "agent"), *CYCLE), fixes(*CYCLE), fixes(*CYCLE)]))
    assert outcome.status == "succeeded"
    assert {w.code for w in outcome.warnings} == {"DERIVATION_CYCLE"}
    assert all("own root" in w.message for w in outcome.warnings)


def test_no_plan_still_gives_dictionary():
    outcome = run(ScriptedLLM([LLMCallError("down")]))
    assert outcome.status == "succeeded" and len(outcome.state.lexemes) == len(CONCEPTS)
    assert outcome.warnings[0].code == "LLM_UNAVAILABLE"


def test_critic_rejection_triggers_repair():
    llm = ScriptedLLM([plan(("sailor", "sea", "agent"), ("merchant", "fish", "agent")),
                       fixes(roots=["merchant"])])
    critic = ScriptedLLM([reviews(("sailor", True), ("merchant", False)), reviews(("sailor", True))])
    outcome = run(llm, critic)
    assert outcome.status == "succeeded" and outcome.warnings == []
    assert word(outcome, "merchant")[0].origin == LexemeOrigin.GENERATED_ROOT
    assert "IMPLAUSIBLE_DERIVATION" in llm.received[1][-1].content


def test_critic_failure_does_not_block():
    outcome = run(ScriptedLLM([plan(("sailor", "sea", "agent"))]), ScriptedLLM([LLMCallError("down")]))
    assert outcome.status == "succeeded"
    assert [w.code for w in outcome.warnings] == ["CRITIC_UNAVAILABLE"]


def test_full_pipeline():
    request = GenerationRequest.model_validate({
        "civilization": {"geography": "coastal", "economy": "mercantile"},
        "language_preferences": {"custom_words": {"hello": "sava"}, "derived_ratio": 0.006}, "seed": 3})
    llm = ScriptedLLM([good_draft(), plan(("sailor", "sea", "agent"))])
    steps: list[str] = []
    outcome = run_generation(request, AgentLLMs(designer=llm, planner=llm, critic=None),
                             civilization=CIV, on_step=steps.append)
    assert outcome.status == "succeeded"
    assert word(outcome.dictionary, "hello")[1] == "sava"
    # The given civilization is used as-is (same object identity as the world's).
    assert outcome.design.civilization.id == CIV.id and outcome.dictionary.state.civilization.id == CIV.id
    # Progress is reported node by node, through both sub-graphs.
    assert steps[0] == "design:analyze_inputs" and "dictionary:plan" in steps and steps[-1] == "dictionary:finalize"


def test_new_concepts_appear_in_dictionary_and_can_be_bases():
    from app.models.lexicon import ConceptPriority, PartOfSpeech, SemanticConcept
    festival = SemanticConcept(id="user_moonlight_festival", gloss="moonlight festival",
                               part_of_speech=PartOfSpeech.NOUN, priority=ConceptPriority.USER)
    words = analyze_user_words({"user_moonlight_festival": "lunara"})[0].words
    # The planner may build from the user's new concept like any other.
    llm = ScriptedLLM([plan(("friend", "user_moonlight_festival", "agent"))])
    outcome = run_morphological_planning(DESIGN, CIV, words, 7, llm, extra_concepts=[festival],
                                         user_notes="Lots of festival words", derived_ratio=0.006)
    assert word(outcome, "user_moonlight_festival")[1] == "lunara"
    assert word(outcome, "friend")[1].startswith("lunara")
    assert outcome.llm_calls[0].messages[1]["content"].count("Lots of festival words") == 1


def test_pipeline_stops_if_design_fails():
    request = GenerationRequest.model_validate({"language_preferences": {"custom_words": {"oasis": "rtak"}}})
    outcome = run_generation(request, AgentLLMs(designer=ScriptedLLM([]), planner=ScriptedLLM([]), critic=None))
    assert outcome.status == "failed" and outcome.dictionary is None
