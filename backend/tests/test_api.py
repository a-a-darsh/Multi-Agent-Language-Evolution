from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_questionnaire_is_reproducible_by_seed():
    first = client.get("/api/questionnaire", params={"seed": 5}).json()
    second = client.get("/api/questionnaire", params={"seed": 5}).json()
    assert first == second
    ids = [q["id"] for q in first["questions"]]
    assert first["seed"] == 5 and len(ids) == 8  # 5 random + name + era + notes
    assert ids[0] == "civilization_name" and ids[-1] == "language_notes"


def test_questionnaire_without_seed_assigns_one():
    body = client.get("/api/questionnaire").json()
    assert isinstance(body["seed"], int)


def test_concepts_listed():
    concepts = client.get("/api/concepts").json()
    assert any(c["id"] == "water" for c in concepts)


def _seed_asking(question_id: str) -> int:
    return next(s for s in range(1000)
                if question_id in {q["id"] for q in client.get("/api/questionnaire", params={"seed": s}).json()["questions"]})


def test_generate_world():
    seed = _seed_asking("geography")
    response = client.post("/api/worlds", json={"seed": seed, "answers": {"geography": "steppe"}})
    assert response.status_code == 200
    body = response.json()
    focal = next(c for c in body["world"]["civilizations"] if c["id"] == body["world"]["focal_civilization_id"])
    assert focal["geography"] == "steppe"
    assert "geography" not in focal["generated_fields"]
    assert 3 <= len(body["world"]["civilizations"]) - 1 <= 5


def test_invalid_answers_return_issues():
    seed = _seed_asking("geography")
    response = client.post("/api/worlds", json={"seed": seed, "answers": {"geography": "moon"}})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "INVALID_ANSWERS"
    assert error["issues"][0]["object_id"] == "geography"


def test_answer_to_unasked_question_rejected():
    seed = next(s for s in range(1000)
                if "geography" not in {q["id"] for q in client.get("/api/questionnaire", params={"seed": s}).json()["questions"]})
    response = client.post("/api/worlds", json={"seed": seed, "answers": {"geography": "steppe"}})
    assert response.json()["error"]["issues"][0]["code"] == "UNKNOWN_QUESTION"


def test_unusable_custom_words_rejected():
    seed = _seed_asking("custom_words")
    response = client.post("/api/worlds", json={"seed": seed, "answers": {"custom_words": {"oasis": "rtak"}}})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "INVALID_CUSTOM_WORDS"
    assert error["issues"][0]["object_id"] == "oasis"
