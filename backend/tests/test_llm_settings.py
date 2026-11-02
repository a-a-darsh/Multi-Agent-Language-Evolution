import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.settings import get_http_client, get_store
from app.llm.settings import SettingsStore
from app.main import app

OPENAI_KEY = "sk-test-0123456789abcd"


def fake_provider(request: httpx.Request) -> httpx.Response:
    """Stands in for every provider's API."""
    url, headers = str(request.url), request.headers
    if url == "http://localhost:11434/api/tags":
        return httpx.Response(200, json={"models": [{"name": "mistral-small"}, {"name": "nomic-embed-text"}]})
    if url == "https://api.openai.com/v1/models":
        if headers.get("authorization") != f"Bearer {OPENAI_KEY}":
            return httpx.Response(401)
        return httpx.Response(200, json={"data": [{"id": "gpt-x"}, {"id": "text-embedding-3"}, {"id": "tts-1"}]})
    if url.startswith("https://api.anthropic.com/v1/models"):
        if headers.get("x-api-key") != "ant-key-0123456789" or "anthropic-version" not in headers:
            return httpx.Response(401)
        return httpx.Response(200, json={"data": [{"id": "claude-a"}, {"id": "claude-b"}]})
    if url == "https://openrouter.ai/api/v1/key":
        ok = headers.get("authorization") == "Bearer or-key-0123456789"
        return httpx.Response(200 if ok else 401, json={"data": {}})
    if url == "https://openrouter.ai/api/v1/models":
        return httpx.Response(200, json={"data": [{"id": "vendor/model"}]})
    return httpx.Response(404)


@pytest.fixture
def client(tmp_path):
    store = SettingsStore(tmp_path / "llm_settings.json")
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_http_client] = lambda: httpx.Client(transport=httpx.MockTransport(fake_provider))
    yield TestClient(app), store
    app.dependency_overrides.clear()


def test_unconfigured_on_first_run(client):
    http, _ = client
    assert http.get("/api/settings/llm").json() == {
        "configured": False, "provider": None, "model": None, "base_url": None, "api_key_hint": None,
    }


def test_local_lists_installed_models_and_saves(client):
    http, store = client
    models = http.post("/api/settings/llm/models", json={"provider": "local"}).json()["models"]
    assert models == ["mistral-small"]  # embedding models hidden

    view = http.put("/api/settings/llm", json={"provider": "local", "model": "mistral-small"}).json()
    assert view["configured"] and view["base_url"] == "http://localhost:11434"
    assert store.load().model == "mistral-small"


def test_openai_hides_non_chat_models(client):
    http, _ = client
    response = http.post("/api/settings/llm/models", json={"provider": "openai", "api_key": OPENAI_KEY})
    assert response.json()["models"] == ["gpt-x"]


@pytest.mark.parametrize("provider, key", [("anthropic", "ant-key-0123456789"), ("openrouter", "or-key-0123456789")])
def test_remote_providers_verify_key(client, provider, key):
    http, _ = client
    assert http.post("/api/settings/llm/models", json={"provider": provider, "api_key": key}).status_code == 200
    bad = http.post("/api/settings/llm/models", json={"provider": provider, "api_key": "wrong"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "INVALID_API_KEY"


def test_api_key_never_returned_but_stored(client):
    http, store = client
    view = http.put("/api/settings/llm", json={"provider": "openai", "api_key": OPENAI_KEY, "model": "gpt-x"}).json()
    assert OPENAI_KEY not in json.dumps(view)
    assert view["api_key_hint"] == "sk-…abcd"
    assert OPENAI_KEY not in http.get("/api/settings/llm").text
    assert json.loads(store.path.read_text())["api_key"] == OPENAI_KEY


def test_saved_key_reused_when_omitted(client):
    http, _ = client
    http.put("/api/settings/llm", json={"provider": "openai", "api_key": OPENAI_KEY, "model": "gpt-x"})
    assert http.post("/api/settings/llm/models", json={"provider": "openai"}).status_code == 200
    # ...but not for a different provider.
    other = http.post("/api/settings/llm/models", json={"provider": "anthropic"})
    assert other.json()["error"]["code"] == "MISSING_API_KEY"


def test_unknown_model_rejected(client):
    http, store = client
    response = http.put("/api/settings/llm", json={"provider": "local", "model": "gpt-99"})
    assert response.json()["error"]["code"] == "UNKNOWN_MODEL"
    assert store.load() is None


def test_unreachable_local_server(client):
    http, _ = client
    app.dependency_overrides[get_http_client] = lambda: httpx.Client(
        transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused"))))
    response = http.post("/api/settings/llm/models", json={"provider": "local"})
    assert response.json()["error"]["code"] == "PROVIDER_UNREACHABLE"
