import pytest
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.agents.design_schema import AffixDraft, AffixReview, Critique, CritiqueFinding, LanguageDesignDraft
from app.agents.design_validation import build_design, check_design, critique_issues
from app.agents.language_designer import build_design_graph, run_language_design
from app.generation.user_vocabulary import analyze_user_words
from app.llm.chat_model import OPENROUTER_BASE_URL, build_chat_model
from app.llm.settings import LLMSettings
from app.llm.structured import LLMCallError, LLMOutputError, ReplayLLM, ScriptedLLM
from app.models.generation import GenerationRequest
from app.models.lexicon import PartOfSpeech as POS


def good_draft(**overrides) -> LanguageDesignDraft:
    fields = dict(
        reasoning="A coastal trading people: open syllables, liquid sounds.",
        name="Old Savari",
        consonants=list("ptkmnslrv"), vowels=list("aeiou"), common_sounds=["l", "a"],
        syllable_patterns=["CV", "CVC"], allow_hiatus=False, restrictions=[], linking_consonant=None,
        word_order="SOV", morphological_type="agglutinative", adposition_type="postpositions",
        has_grammatical_gender=False, has_case_marking=True, features=[],
        derivational_affixes=[
            AffixDraft(id="agent", name="Agent", meaning="person associated with", position="suffix", form="ka",
                       input_parts_of_speech=[POS.NOUN, POS.VERB], output_part_of_speech=POS.NOUN),
            AffixDraft(id="place", name="Place", meaning="place of", position="suffix", form="ri",
                       input_parts_of_speech=[POS.NOUN], output_part_of_speech=POS.NOUN),
            AffixDraft(id="negative", name="Negative", meaning="not", position="prefix", form="mo",
                       input_parts_of_speech=[POS.ADJECTIVE], output_part_of_speech=POS.ADJECTIVE),
        ],
        linking_vowel="i", summary="A flowing language of traders.",
    )
    fields.update(overrides)
    return LanguageDesignDraft(**fields)


def request(**prefs) -> GenerationRequest:
    return GenerationRequest.model_validate({"civilization": {"geography": "coastal"},
                                             "language_preferences": prefs, "seed": 42})


def test_happy_path_builds_design():
    llm = ScriptedLLM([good_draft()])
    outcome = run_language_design(request(word_order="SOV", custom_words={"hello": "sava"}), llm)

    assert outcome.status == "succeeded"
    design = outcome.design
    assert design.name == "Old Savari"
    assert [r.id for r in design.derivational_rules] == ["agent", "place", "negative"]
    assert design.phonology.phoneme("l").frequency == 2.0
    assert design.boundary_repair.linking_vowel == "i"
    assert [a.step for a in outcome.attempts] == ["design", "critique"]
    assert len(outcome.llm_calls) == 1  # no critic LLM configured; deterministic quality checks only
    # The prompt carried the user's constraints.
    prompt = llm.received[0][1].content
    assert "SOV" in prompt and "hello: sava" in prompt and outcome.civilization.geography in prompt


def test_repair_loop_sends_precise_errors_back():
    llm = ScriptedLLM([good_draft(word_order="SVO"), good_draft()])
    outcome = run_language_design(request(word_order="SOV"), llm)

    assert outcome.status == "succeeded"
    assert [a.step for a in outcome.attempts] == ["design", "repair", "critique"]
    assert outcome.attempts[0].issues[0].code == "CONSTRAINT_VIOLATED"
    repair_prompt = llm.received[1][-1].content
    assert "CONSTRAINT_VIOLATED" in repair_prompt and "word_order" in repair_prompt
    assert '"consonants"' in repair_prompt  # previous draft included so it can edit, not restart


def test_user_word_needs_are_added_by_code_not_llm():
    missing = good_draft(consonants=list("ptkmnsl"), syllable_patterns=["CV", "CVC"])  # no r, v; no V syllables
    llm = ScriptedLLM([missing])
    outcome = run_language_design(request(custom_words={"water": "mari", "hello": "sava", "mother": "ama"}), llm)

    assert outcome.status == "succeeded" and len(llm.received) == 1  # no repair round needed
    ph = outcome.design.phonology
    assert ph.has_phoneme("r") and ph.has_phoneme("v") and "V" in ph.syllable_patterns
    # r is already added for the affix -ri; v only for the user's words.
    notes = {w.code: w.message for w in outcome.warnings}
    assert "'v'" in notes["ADJUSTED_FOR_USER_WORDS"] and "'r'" in notes["ADDED_AFFIX_SOUNDS"]


def test_gives_up_after_two_repairs():
    wrong_order = good_draft(word_order="SVO")
    llm = ScriptedLLM([wrong_order, wrong_order, wrong_order])
    outcome = run_language_design(request(word_order="SOV"), llm)

    assert outcome.status == "failed"
    assert len(llm.received) == 3  # 1 design + 2 repairs, never more
    assert outcome.issues[0].code == "CONSTRAINT_VIOLATED"
    assert outcome.design is None


def test_malformed_output_is_retried():
    llm = ScriptedLLM([LLMOutputError("not JSON"), good_draft()])
    outcome = run_language_design(request(), llm)
    assert outcome.status == "succeeded"
    assert outcome.attempts[0].issues[0].code == "LLM_OUTPUT_INVALID"
    assert "could not be read" in llm.received[1][-1].content


def test_schema_forces_explicit_decisions():
    # Fields without defaults are "required" in the JSON schema sent to the model.
    required = set(LanguageDesignDraft.model_json_schema()["required"])
    assert {"features", "restrictions", "common_sounds", "linking_vowel", "linking_consonant"} <= required


def test_unreachable_model_fails_immediately():
    llm = ScriptedLLM([LLMCallError("connection refused")])
    outcome = run_language_design(request(), llm)
    assert outcome.status == "failed" and len(llm.received) == 1
    assert outcome.issues[0].code == "LLM_UNAVAILABLE"
    assert outcome.llm_calls[0].error


def test_bad_user_words_fail_before_calling_llm():
    llm = ScriptedLLM([])
    outcome = run_language_design(request(custom_words={"oasis": "rtak"}), llm)
    assert outcome.status == "failed" and llm.received == []
    assert outcome.issues[0].code == "UNSUPPORTED_SYLLABLE_STRUCTURE"


@pytest.mark.parametrize("draft, code", [
    # A bare consonant suffix in a CV-only language with no linking vowel: can never attach.
    # (A vowel suffix like -a now can: it replaces the root's final vowel, as in Italian.)
    (good_draft(syllable_patterns=["CV"], linking_vowel=None, derivational_affixes=[
        AffixDraft(id="adj", name="Adj", meaning="having", position="suffix", form="k",
                   input_parts_of_speech=[POS.NOUN], output_part_of_speech=POS.ADJECTIVE)] * 1 + good_draft(
        ).derivational_affixes[:2]), "AFFIX_UNUSABLE"),
    (good_draft(derivational_affixes=good_draft().derivational_affixes[:2] + [
        # A missing letter is now added by code; characters that aren't letters at all still fail.
        AffixDraft(id="buzzing", name="Buzzing", meaning="like a buzz", position="suffix", form="z9",
                   input_parts_of_speech=[POS.NOUN], output_part_of_speech=POS.NOUN)]), "AFFIX_UNKNOWN_SOUNDS"),
    (good_draft(linking_vowel=None, linking_consonant="z"), "INVALID_REPAIR"),
])
def test_design_checks(draft, code):
    outcome = run_language_design(request(), ScriptedLLM([draft] * 3))
    assert outcome.status == "failed"
    assert code in {i.code for i in outcome.issues}


def test_affix_hyphens_are_notation():
    hyphenated = good_draft(derivational_affixes=[
        a.model_copy(update={"form": f"{a.form}-" if a.position == "prefix" else f"-{a.form}"})
        for a in good_draft().derivational_affixes])
    outcome = run_language_design(request(), ScriptedLLM([hyphenated]))
    assert outcome.status == "succeeded"
    assert ["".join(m.phonemes) for m in outcome.design.affixes] == ["ka", "ri", "mo"]


def test_tiny_inventory_rejected():
    # The schema's minimum sizes make this hard to reach via the LLM, so build
    # the draft with model_construct (skips validation) and check it directly.
    draft = LanguageDesignDraft.model_construct(**{
        **good_draft().__dict__, "consonants": ["p", "t"], "vowels": ["a"], "common_sounds": [],
        "syllable_patterns": ["CV"], "linking_vowel": None,
        "derivational_affixes": [
            AffixDraft(id=affix_id, name="Thing", meaning="a kind of thing", position=pos, form=form,
                       input_parts_of_speech=[POS.NOUN], output_part_of_speech=POS.NOUN)
            for affix_id, pos, form in [("one", "suffix", "pa"), ("two", "suffix", "ta"), ("three", "prefix", "ta")]],
    })
    design = build_design(draft, "lang")
    reqs, _ = analyze_user_words({})
    assert [i.code for i in check_design(design, draft, request(), reqs).errors] == ["INVENTORY_TOO_SMALL"]


def test_requested_feature_must_appear():
    outcome = run_language_design(request(special_features=["evidentiality"]), ScriptedLLM([good_draft()] * 3))
    assert outcome.issues[0].code == "MISSING_FEATURE"


def test_replay_reproduces_a_run():
    original = run_language_design(request(), ScriptedLLM([LLMOutputError("bad"), good_draft()]))
    replayed = run_language_design(request(), ReplayLLM(original.llm_calls))
    assert replayed.status == "succeeded"
    assert [a.draft for a in replayed.attempts] == [a.draft for a in original.attempts]
    assert [i.code for a in replayed.attempts for i in a.issues] == ["LLM_OUTPUT_INVALID"]


def test_graph_shape():
    graph = build_design_graph(ScriptedLLM([])).get_graph()
    assert {"analyze_inputs", "design", "validate", "critique", "repair", "finalize", "fail"} <= set(graph.nodes)


# --- critic ----------------------------------------------------------------------

def inflectional_draft() -> LanguageDesignDraft:
    affixes = good_draft().derivational_affixes[:2] + [
        AffixDraft(id="accusative", name="Accusative", meaning="marks the object", position="suffix", form="ma",
                   input_parts_of_speech=[POS.NOUN], output_part_of_speech=POS.NOUN)]
    return good_draft(derivational_affixes=affixes)


def review(affix_id: str, kind: str = "derivational") -> AffixReview:
    return AffixReview(affix_id=affix_id, example="x -> y", kind=kind)


def flag(affix_id="accusative") -> Critique:
    return Critique(affix_reviews=[review("agent"), review("place"), review(affix_id, "inflectional")],
                    inventory_looks_designed=True, other_findings=[])


ACCEPT = Critique(affix_reviews=[review("agent"), review("place"), review("negative")],
                  inventory_looks_designed=True, other_findings=[])


def test_critic_finding_triggers_repair():
    designer = ScriptedLLM([inflectional_draft(), good_draft()])
    critic = ScriptedLLM([flag(), ACCEPT])
    outcome = run_language_design(request(), designer, critic_llm=critic)

    assert outcome.status == "succeeded" and outcome.warnings == []
    assert [a.step for a in outcome.attempts] == ["design", "critique", "repair", "critique"]
    assert "accusative" not in [r.id for r in outcome.design.derivational_rules]
    repair_prompt = designer.received[1][-1].content
    assert "CRITIC_INFLECTIONAL_AFFIX" in repair_prompt and "-ma" in repair_prompt and "instrument" in repair_prompt
    assert [c.step for c in outcome.llm_calls] == ["design", "critique", "repair", "critique"]


def test_unresolved_critic_findings_become_warnings_not_failure():
    designer = ScriptedLLM([inflectional_draft()] * 3)
    critic = ScriptedLLM([flag()] * 3)
    outcome = run_language_design(request(), designer, critic_llm=critic)

    assert outcome.status == "succeeded"
    assert len(designer.received) == 3  # repair budget is shared and still enforced
    assert [w.code for w in outcome.warnings] == ["CRITIC_INFLECTIONAL_AFFIX"]


def test_broken_repair_falls_back_to_last_valid_design():
    broken = good_draft(word_order="SVO")  # violates the user's SOV choice
    # quality repair -> broken, then 2 structural repairs -> still broken -> fall back
    designer = ScriptedLLM([inflectional_draft(), broken, broken, broken])
    critic = ScriptedLLM([flag()])
    outcome = run_language_design(request(word_order="SOV"), designer, critic_llm=critic)

    assert outcome.status == "succeeded"
    assert "accusative" in [r.id for r in outcome.design.derivational_rules]  # the earlier valid draft
    assert {w.code for w in outcome.warnings} == {"CRITIC_INFLECTIONAL_AFFIX", "REPAIR_REVERTED"}


def test_structural_repairs_do_not_use_up_quality_budget():
    bad_order = inflectional_draft().model_copy(update={"word_order": "SVO"})
    designer = ScriptedLLM([bad_order, bad_order, inflectional_draft(), good_draft()])
    critic = ScriptedLLM([flag(), ACCEPT])
    outcome = run_language_design(request(word_order="SOV"), designer, critic_llm=critic)
    # 2 structural repairs, then the critic's finding still gets its own repair.
    assert [a.step for a in outcome.attempts] == ["design", "repair", "repair", "critique", "repair", "critique"]
    assert outcome.status == "succeeded" and outcome.warnings == []
    assert "accusative" not in [r.id for r in outcome.design.derivational_rules]


def test_critic_failure_does_not_block():
    outcome = run_language_design(request(), ScriptedLLM([good_draft()]),
                                  critic_llm=ScriptedLLM([LLMCallError("critic down")]))
    assert outcome.status == "succeeded"
    assert [w.code for w in outcome.warnings] == ["CRITIC_UNAVAILABLE"]


def test_critic_hallucinated_target_ignored():
    outcome = run_language_design(request(), ScriptedLLM([good_draft()]),
                                  critic_llm=ScriptedLLM([flag(affix_id="does_not_exist")]))
    assert outcome.status == "succeeded" and outcome.warnings == []


def test_verdict_computed_from_reviews():
    lazy_inventory = Critique(affix_reviews=[review("agent"), review("place"), review("negative")],
                              inventory_looks_designed=False, other_findings=[CritiqueFinding(
                                  target="grammar", problem_type="incoherent_grammar",
                                  explanation="x", suggestion="y")])
    actionable, advisory = critique_issues(lazy_inventory, good_draft())
    assert [i.code for i in actionable] == ["CRITIC_UNDESIGNED_INVENTORY"]
    assert [i.code for i in advisory] == ["CRITIC_INCOHERENT_GRAMMAR"]


def test_advisory_findings_reported_but_never_trigger_repair():
    opinionated = ACCEPT.model_copy(update={"other_findings": [CritiqueFinding(
        target="grammar", problem_type="incoherent_grammar", explanation="Mixed prefixes and suffixes.",
        suggestion="Use only suffixes.")]})
    designer = ScriptedLLM([good_draft()])
    outcome = run_language_design(request(), designer, critic_llm=ScriptedLLM([opinionated]))
    assert outcome.status == "succeeded" and len(designer.received) == 1
    assert [w.code for w in outcome.warnings] == ["CRITIC_INCOHERENT_GRAMMAR"]


def test_class_changing_affixes_are_acceptable():
    # Affixes can attach to any word now, so a design of class-changing affixes no longer needs repair.
    class_changers = good_draft(derivational_affixes=[
        AffixDraft(id="verbalizer", name="Verbalizer", meaning="to do X", position="suffix", form="ta",
                   input_parts_of_speech=[POS.NOUN], output_part_of_speech=POS.VERB),
        AffixDraft(id="nominalizer", name="Nominalizer", meaning="act of X", position="suffix", form="ri",
                   input_parts_of_speech=[POS.VERB], output_part_of_speech=POS.NOUN),
        good_draft().derivational_affixes[2]])
    outcome = run_language_design(request(), ScriptedLLM([class_changers]))
    assert outcome.status == "succeeded" and outcome.warnings == []


def test_oversized_inventory_triggers_repair():
    everything = good_draft(consonants=list("pbtdkgmnfvszhlr"))  # 15 consonants
    outcome = run_language_design(request(), ScriptedLLM([everything, good_draft()]))
    assert outcome.attempts[1].issues[0].code == "INVENTORY_LARGE"
    assert outcome.status == "succeeded" and outcome.warnings == []


def test_critic_not_called_until_structure_is_valid():
    critic = ScriptedLLM([ACCEPT])
    run_language_design(request(word_order="SOV"), ScriptedLLM([good_draft(word_order="SVO"), good_draft()]),
                        critic_llm=critic)
    assert len(critic.received) == 1  # only the valid second draft was reviewed


def test_affix_copying_user_word_is_flagged_deterministically():
    copycat = good_draft(derivational_affixes=good_draft().derivational_affixes[1:] + [
        AffixDraft(id="agent", name="Agent", meaning="person associated with", position="suffix", form="mar",
                   input_parts_of_speech=[POS.NOUN], output_part_of_speech=POS.NOUN)])
    # "mara" shares no piece with good_draft's affixes (-ka, -ri, mo-), so the repair clears the warning.
    outcome = run_language_design(request(custom_words={"water": "mara"}), ScriptedLLM([copycat, good_draft()]))
    assert outcome.attempts[1].issues[0].code == "AFFIX_COPIES_USER_WORD"
    assert outcome.status == "succeeded" and outcome.warnings == []


@pytest.mark.parametrize("settings, cls", [
    (dict(provider="local", model="m"), ChatOllama),
    (dict(provider="openai", model="m", api_key="sk-x"), ChatOpenAI),
    (dict(provider="anthropic", model="m", api_key="k"), ChatAnthropic),
    (dict(provider="openrouter", model="vendor/m", api_key="k"), ChatOpenAI),
])
def test_chat_model_per_provider(settings, cls):
    model = build_chat_model(LLMSettings(**settings))
    assert isinstance(model, cls)
    if settings["provider"] == "openrouter":
        assert model.openai_api_base == OPENROUTER_BASE_URL
