from concurrent.futures import Executor, Future

import pytest
from fastapi.testclient import TestClient

from app.agents.pipeline import AgentLLMs
from app.api.deps import get_generation_service, get_repository
from app.api.errors import ApiError
from app.llm.structured import ScriptedLLM
from app.main import app
from app.persistence.database import make_session_factory
from app.persistence.repository import Repository
from app.services.generation_jobs import GenerationService
from tests.test_language_designer import good_draft


class InlineExecutor(Executor):
    """Runs jobs immediately so tests see the finished result."""

    def submit(self, fn, *args, **kwargs):
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


PLAN = {"reasoning": "r", "compounds": [], "derivations": [
    {"concept_id": "enemy", "base_concept_id": "fight", "rule_id": "agent", "explanation": "fighter"},
    {"concept_id": "bad", "base_concept_id": "good", "rule_id": "negative", "explanation": "not good"},
], "extended_senses": []}


@pytest.fixture
def env(tmp_path):
    repo = Repository(make_session_factory(tmp_path / "test.sqlite3"))
    scripts: list = []

    def factory():
        llm = ScriptedLLM(scripts.pop(0) if scripts else [])
        return AgentLLMs(designer=llm, planner=llm, critic=None), {"provider": "test", "model": "scripted"}

    service = GenerationService(repo, factory, InlineExecutor())
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_generation_service] = lambda: service
    yield TestClient(app), scripts, repo
    app.dependency_overrides.clear()


def generate(client, scripts, responses, answers=None):
    scripts.append(responses)
    started = client.post("/api/generations", json={"seed": 11, "answers": answers or {}, "derived_ratio": 0.0125})
    assert started.status_code == 202, started.text
    return client.get(f"/api/generations/{started.json()['generation_id']}").json()


def test_full_generation_is_saved_and_browsable(env):
    client, scripts, _ = env
    job = generate(client, scripts, [good_draft(), PLAN])
    assert job["status"] == "succeeded", job
    assert job["steps"][0] == "world" and "design:critique" in job["steps"] and job["steps"][-1] == "dictionary:finalize"
    lang_id = job["language_id"]

    [summary] = client.get("/api/languages").json()
    assert summary["id"] == lang_id and summary["name"] == "Old Savari" and summary["derived_count"] == 2

    overview = client.get(f"/api/languages/{lang_id}").json()
    assert [s["letter"] for s in overview["vowels"]] == list("aeiou")
    agent = next(a for a in overview["affixes"] if a["rule_id"] == "agent")
    assert agent["form"] == "-ka" and [e["concept_id"] for e in agent["examples"]] == ["enemy"]
    assert overview["counts"]["derived"] == 2

    # The world and the language share one civilization.
    world = client.get(f"/api/languages/{lang_id}/world").json()
    assert world["focal_civilization_id"] == overview["civilization"]["id"]
    assert len(world["civilizations"]) >= 4


def test_dictionary_search_and_detail(env):
    client, scripts, _ = env
    lang_id = generate(client, scripts, [good_draft(), PLAN])["language_id"]
    base = f"/api/languages/{lang_id}/dictionary"

    [enemy] = client.get(base, params={"q": "enemy"}).json()["entries"]
    assert enemy["origin"] == "derived" and enemy["word"].endswith("ka")
    assert client.get(base, params={"q": enemy["word"]}).json()["entries"][0]["id"] == enemy["id"]

    detail = client.get(f"{base}/{enemy['id']}").json()
    assert [s["kind"] for s in detail["segments"]][-1] == "suffix"
    assert detail["derived_from"]["concept_id"] == "fight" and detail["rule"]["id"] == "agent"
    assert detail["family_root"]["concept_id"] == "fight"
    assert {e["concept_id"] for e in detail["family"]} == {"fight", "enemy"}
    assert "".join(detail["syllables"]) == detail["word"]

    verbs = client.get(base, params={"pos": "verb", "page_size": 200}).json()
    assert verbs["total"] > 20 and all(e["part_of_speech"] == "verb" for e in verbs["entries"])
    derived = client.get(base, params={"origin": "derived"}).json()
    assert derived["total"] == 2


def test_generation_log(env):
    client, scripts, _ = env
    lang_id = generate(client, scripts, [good_draft(), PLAN])["language_id"]
    log = client.get(f"/api/languages/{lang_id}/generation").json()
    assert log["seed"] == 11
    assert [c["step"] for c in log["llm_calls"]] == ["design", "plan"]
    assert {s["agent"] for s in log["steps"]} == {"design", "dictionary"}


def test_old_records_load_after_schemas_get_stricter(env):
    """Regression: a saved language returned 500 once a schema gained a min_length."""
    import json

    from app.persistence.database import GenerationRow

    client, scripts, repo = env
    job = generate(client, scripts, [good_draft(), PLAN])
    with repo.sessions.begin() as s:
        row = s.get(GenerationRow, job["id"])
        record = json.loads(row.outcome_json)
        record["design"]["attempts"][0]["draft"]["derivational_affixes"][0]["meaning"] = ""  # now invalid
        record["design"]["field_from_an_old_version"] = True
        row.outcome_json = json.dumps(record)

    lang = job["language_id"]
    assert client.get(f"/api/languages/{lang}").status_code == 200
    assert client.get(f"/api/languages/{lang}/generation").status_code == 200


def test_failed_generation_saves_no_language(env):
    client, scripts, _ = env
    wrong = good_draft(word_order="SVO")
    seed = next(s for s in range(500) if "word_order" in {q["id"] for q in client.get(
        "/api/questionnaire", params={"seed": s}).json()["questions"]})
    scripts.append([wrong, wrong, wrong])
    gid = client.post("/api/generations", json={"seed": seed, "answers": {"word_order": "SOV"}}).json()["generation_id"]
    job = client.get(f"/api/generations/{gid}").json()
    assert job["status"] == "failed" and job["errors"][0]["code"] == "CONSTRAINT_VIOLATED"
    assert job["language_id"] is None
    assert client.get("/api/languages").json() == []


def test_crash_is_recorded_not_left_running(env):
    client, scripts, _ = env
    job = generate(client, scripts, [good_draft()])  # the planner has no scripted reply -> crash
    assert job["status"] == "failed" and job["errors"][0]["code"] == "INTERNAL_ERROR"


def test_unconfigured_llm_rejected_before_queueing(env, tmp_path):
    client, _, repo = env

    def no_llm():
        raise ApiError(409, "LLM_NOT_CONFIGURED", "Choose an AI model in the setup screen first")

    app.dependency_overrides[get_generation_service] = lambda: GenerationService(repo, no_llm, InlineExecutor())
    response = client.post("/api/generations", json={"seed": 1, "answers": {}})
    assert response.status_code == 409 and response.json()["error"]["code"] == "LLM_NOT_CONFIGURED"


def test_invalid_answers_rejected_before_queueing(env):
    client, _, _ = env
    response = client.post("/api/generations", json={"seed": 1, "answers": {"nonsense": 1}})
    assert response.status_code == 422


def test_interrupted_jobs_marked_failed_on_restart(tmp_path):
    from app.models.generation import GenerationRequest
    repo = Repository(make_session_factory(tmp_path / "db.sqlite3"))
    gid = repo.create_generation(GenerationRequest(seed=1), {})
    repo.record_step(gid, "design:design")
    Repository(repo.sessions).mark_interrupted()
    assert repo.get_generation(gid).errors[0].code == "INTERRUPTED"


@pytest.mark.parametrize("path", ["/api/languages/nope", "/api/languages/nope/dictionary",
                                  "/api/generations/nope"])
def test_not_found(env, path):
    client, _, _ = env
    assert client.get(path).status_code == 404
