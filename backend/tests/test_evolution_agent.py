from langchain_core.messages import ToolMessage

from app.agents.evolution_agent import run_evolution_agent
from app.agents.evolution_tools import FindWordsArgs, find_words, language_overview
from app.evolution import templates
from app.evolution.sound_change_engine import apply_sound_change
from app.llm.structured import LLMCallError
from app.llm.tool_calling import ScriptedToolLLM
from app.validation.morphology_validator import validate_lexicon
from tests.test_evolution import ROOTS, build, words
from tests.test_languages_api import env  # noqa: F401  (fixture)

LOSS = {"template": "final_vowel_loss"}
REPAIR = {"template": "cluster_epenthesis"}
RATIONALE = "Unstressed final vowels weakened and were lost; clusters were broken up by e."


def tool_results(llm: ScriptedToolLLM) -> list[str]:
    """The tool results the model saw before its last reply."""
    return [m.content for m in llm.received[-1] if isinstance(m, ToolMessage)]


# --- tools on their own -------------------------------------------------------------

def test_overview_describes_sounds_rules_and_history():
    state = build(ROOTS)
    text = language_overview(state)
    assert "Testish" in text and "Syllable shapes: V, CV, CVC" in text
    assert "k /k/" in text and "none yet" in text
    evolved = apply_sound_change(state, templates.palatalization(), -700).state
    assert "Velar palatalization" in language_overview(evolved)


def test_find_words():
    state = build(ROOTS)
    assert "earth kanra" in find_words(state, FindWordsArgs(letters="nr"))
    assert "No words" in find_words(state, FindWordsArgs(letters="nr", position="start"))
    assert find_words(state, FindWordsArgs(letters="a", position="end")).startswith("4 words")
    assert "not written with" in find_words(state, FindWordsArgs(letters="ñ"))
    assert "person" in find_words(state, FindWordsArgs(concept="pers"))


# --- the agent loop ------------------------------------------------------------------

def test_agent_inspects_previews_revises_and_submits():
    llm = ScriptedToolLLM([
        {"tool": "get_language_overview"},
        {"tool": "preview_sound_changes", "args": {"changes": [LOSS]}},
        {"tool": "preview_sound_changes", "args": {"changes": [LOSS, REPAIR]}},
        {"tool": "submit_evolution", "args": {"changes": [LOSS, REPAIR], "rationale": RATIONALE}},
    ])
    outcome = run_evolution_agent(build(ROOTS), llm, years=250)
    assert outcome.status == "succeeded", outcome.issues
    assert outcome.previews == 2 and len(outcome.llm_calls) == 4
    assert [c.name for c in outcome.changes] == ["Final vowel loss", "Final cluster epenthesis"]
    assert words(outcome.state)["earth"] == "kaner"
    assert outcome.state.version.valid_from_year == -1000 + 250
    assert outcome.rationale == RATIONALE
    assert validate_lexicon(outcome.state).ok
    # What the model was shown: the first preview flagged illegal words, the second said all clear.
    log = {u.tool: u.result for u in outcome.tool_log}
    assert "ILLEGAL" in outcome.tool_log[1].result and "kanr" in outcome.tool_log[1].result
    assert "could be submitted" in outcome.tool_log[2].result
    assert log["submit_evolution"].startswith("Accepted")


def test_refused_submission_is_explained_and_can_be_fixed():
    llm = ScriptedToolLLM([
        {"tool": "submit_evolution", "args": {"changes": [LOSS], "rationale": RATIONALE}},
        {"tool": "submit_evolution", "args": {"changes": [LOSS, REPAIR], "rationale": RATIONALE}},
    ])
    outcome = run_evolution_agent(build(ROOTS), llm)
    assert outcome.status == "succeeded"
    assert outcome.tool_log[0].result.startswith("Refused") and "kanr" in outcome.tool_log[0].result


def test_exceptions_by_concept():
    llm = ScriptedToolLLM([{"tool": "submit_evolution", "args": {
        "changes": [LOSS, REPAIR], "rationale": RATIONALE,
        "exceptions": [{"change_index": 0, "concept_id": "now", "reason": "very frequent word"}]}}])
    outcome = run_evolution_agent(build(ROOTS), llm)
    assert outcome.status == "succeeded" and words(outcome.state)["now"] == "ado"
    assert outcome.exceptions[0].reason == "very frequent word"


def test_bad_arguments_and_unknown_tools_are_reported_back():
    llm = ScriptedToolLLM([
        {"calls": [("preview_sound_changes", {"changes": [{"template": "teleportation"}]}),
                   ("preview_sound_changes", {"changes": "not a list"}),
                   ("summon_dragon", {})]},
        {"tool": "submit_evolution", "args": {"changes": [LOSS, REPAIR], "rationale": RATIONALE}},
    ])
    outcome = run_evolution_agent(build(ROOTS), llm)
    assert outcome.status == "succeeded"
    first, second, third = [u.result for u in outcome.tool_log[:3]]
    assert "no template 'teleportation'" in first
    assert second.startswith("Invalid arguments")
    assert "Unknown tool" in third


def test_custom_rules_and_json_string_arguments():
    rule = {"name": "h-dropping", "description": "k is lost before a", "output_mapping": {"k": None},
            "right": [{"kind": "segment", "matcher": {"ipa": ["a"]}}]}
    # Some models send nested objects as JSON text.
    import json
    llm = ScriptedToolLLM([{"tool": "submit_evolution", "args": {
        "changes": json.dumps([{"rule": rule}]), "rationale": RATIONALE}}])
    outcome = run_evolution_agent(build(ROOTS), llm)
    assert outcome.status == "succeeded", outcome.tool_log[0].result
    assert words(outcome.state)["house"] == "asa"


def test_thinking_aloud_between_tools_is_fine():
    llm = ScriptedToolLLM([{"text": "Let me look first."}, {"tool": "get_language_overview"},
                           {"text": "Final vowels look weak."}, {"text": "I will submit."},
                           {"tool": "submit_evolution", "args": {"changes": [LOSS, REPAIR], "rationale": RATIONALE}}])
    assert run_evolution_agent(build(ROOTS), llm).status == "succeeded"


def test_a_step_written_as_text_is_run_as_a_preview_only():
    text = """I'll nasalize. {"sound_changes": [{"rule": {"name": "nasalization", "target": {"features":
        {"class": "V", "nasal": False}}, "output_features": {"nasal": True},
        "right": [{"kind": "segment", "matcher": {"ipa": "n"}}, {"kind": "boundary"}]}}]}"""
    llm = ScriptedToolLLM([{"text": text},
                           {"tool": "submit_evolution", "args": {"changes": [LOSS, REPAIR], "rationale": RATIONALE}}])
    outcome = run_evolution_agent(build(ROOTS), llm)
    first = outcome.tool_log[0]
    assert first.tool == "preview_sound_changes" and "run as a preview" in first.result
    assert "words change" in first.result   # the rule really ran (ipa "n" accepted as ["n"])
    assert outcome.previews == 1 and outcome.status == "succeeded"
    # A text step is never treated as a submission.
    llm = ScriptedToolLLM([{"text": text}, {"text": "done"}, {"text": "done"}, {"text": "done"}])
    assert run_evolution_agent(build(ROOTS), llm).status == "failed"


def test_described_actions_are_recovered_through_structured_output():
    llm = ScriptedToolLLM([
        {"text": "I'll preview final vowel loss with epenthesis."},
        {"structured": {"tool": "preview_sound_changes", "changes": [LOSS, REPAIR]}},
        {"text": "Looks good, I will submit it."},
        {"structured": {"tool": "submit_evolution", "changes": [LOSS, REPAIR], "rationale": RATIONALE}},
    ])
    outcome = run_evolution_agent(build(ROOTS), llm)
    assert outcome.status == "succeeded", outcome.issues
    assert [u.tool for u in outcome.tool_log] == ["preview_sound_changes", "submit_evolution"]
    assert [c.step for c in outcome.llm_calls] == ["act", "next_action", "act", "next_action"]


def test_text_replies_are_nudged_then_fail():
    llm = ScriptedToolLLM([{"text": "I think palatalization."}, {"text": "Yes."}, {"text": "Done!"}])
    outcome = run_evolution_agent(build(ROOTS), llm)
    assert outcome.status == "failed" and "stopped calling tools" in outcome.issues[0]
    # Each text reply also tried the structured fallback, which produced nothing here.
    assert [c.step for c in outcome.llm_calls].count("act") == 3
    assert all(c.error for c in outcome.llm_calls if c.step == "next_action")


def test_budget_is_enforced_with_a_reminder():
    llm = ScriptedToolLLM([{"tool": "get_language_overview"}] * 5)
    outcome = run_evolution_agent(build(ROOTS), llm, max_llm_calls=5)
    assert outcome.status == "failed" and "within 5 model calls" in outcome.issues[-1]
    assert "actions left" in outcome.tool_log[-1].result
    assert outcome.state is None


def test_unreachable_model_fails_immediately():
    outcome = run_evolution_agent(build(ROOTS), ScriptedToolLLM([LLMCallError("connection refused")]))
    assert outcome.status == "failed" and "connection refused" in outcome.issues[0]


def test_reversal_of_an_earlier_change_is_flagged():
    state = apply_sound_change(build(ROOTS), templates.intervocalic_voicing(), -700).state   # kasa: no; ado
    back = {"rule": {"name": "devoicing", "output_mapping": {"d": "t"}}}
    llm = ScriptedToolLLM([{"tool": "preview_sound_changes", "args": {"changes": [back]}},
                           {"tool": "submit_evolution", "args": {"changes": [LOSS, REPAIR], "rationale": RATIONALE}}])
    outcome = run_evolution_agent(state, llm)
    assert "undoing the earlier 'Intervocalic voicing'" in outcome.tool_log[0].result


def test_the_task_mentions_years_and_context():
    llm = ScriptedToolLLM([{"tool": "submit_evolution", "args": {"changes": [LOSS, REPAIR], "rationale": RATIONALE}}])
    run_evolution_agent(build(ROOTS), llm, years=300, context="Conquered by horse riders from the steppe.")
    task = llm.received[0][1].content
    assert "300 years" in task and "horse riders" in task


def test_rules_that_change_nothing_are_refused():
    idle = {"template": "prothesis"}   # no word starts with s + consonant
    llm = ScriptedToolLLM([
        {"tool": "preview_sound_changes", "args": {"changes": [LOSS, REPAIR, idle]}},
        {"tool": "submit_evolution", "args": {"changes": [LOSS, REPAIR, idle], "rationale": RATIONALE}},
        {"tool": "submit_evolution", "args": {"changes": [LOSS, REPAIR], "rationale": RATIONALE}},
    ])
    outcome = run_evolution_agent(build(ROOTS), llm)
    assert "change no word" in outcome.tool_log[0].result
    assert outcome.tool_log[1].result.startswith("Refused: rule(s) 2 (Prothesis)")
    assert outcome.status == "succeeded" and len(outcome.changes) == 2


def test_agent_api(env):  # noqa: F811
    from app.api.deps import get_tool_llm
    from app.main import app
    from tests.test_language_designer import good_draft
    from tests.test_languages_api import PLAN, generate

    client, scripts, _ = env
    lang_id = generate(client, scripts, [good_draft(), PLAN])["language_id"]
    vowel = client.get(f"/api/languages/{lang_id}").json()["vowels"][0]["ipa"]
    shift = {"rule": {"name": "vowel shift", "description": "a vowel opens", "output_mapping": {vowel: "ɔ"}}}
    replies = [{"tool": "get_language_overview"},
               {"tool": "submit_evolution", "args": {"changes": [shift], "rationale": RATIONALE}}]
    app.dependency_overrides[get_tool_llm] = lambda: ScriptedToolLLM(list(replies))
    try:
        dry = client.post(f"/api/languages/{lang_id}/evolve-agent", json={"years": 150}).json()
        assert dry["status"] == "succeeded" and not dry["saved"] and dry["changes"][0]["name"] == "vowel shift"
        assert client.get(f"/api/languages/{lang_id}").json()["version_count"] == 1
        saved = client.post(f"/api/languages/{lang_id}/evolve-agent", json={"years": 150, "save": True}).json()
        assert saved["saved"] and client.get(f"/api/languages/{lang_id}").json()["version_count"] == 2
        assert [t["tool"] for t in saved["tool_log"]] == ["get_language_overview", "submit_evolution"]
    finally:
        app.dependency_overrides.pop(get_tool_llm, None)


def test_loose_rules_in_text_are_collected():
    from app.agents.evolution_agent import _changes_from_text

    text = ('Here is the rule:\n```json\n{"rule": {"name": "N", "target": {"features": {"class": "V"}}, '
            '"output_features": {"nasal": true}, "right": [{"kind": "segment", "matcher": {"ipa": ["n"]}}, '
            '{"kind": "boundary"}]}}\n```\nand then {"template": "h_loss"} and a bare one '
            '{"name": "n loss", "output_mapping": {"n": null}}')
    found = _changes_from_text(text)
    assert [list(c)[0] for c in found["changes"]] == ["rule", "template", "rule"]
    assert found["changes"][2]["rule"]["name"] == "n loss"
    assert _changes_from_text("No JSON here, just {braces}.") is None
