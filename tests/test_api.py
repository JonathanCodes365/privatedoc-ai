import pytest
from fastapi.testclient import TestClient

from app.main import app, documents


@pytest.fixture(autouse=True)
def clear_documents():
    documents.clear()
    yield
    documents.clear()


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_document_lifecycle_and_search():
    created = client.post(
        "/documents",
        json={"title": "Privacy policy", "content": "Private documents stay on the local server."},
    )

    assert created.status_code == 201
    document_id = created.json()["id"]
    results = client.post("/search", json={"query": "local server"})
    assert results.status_code == 200
    assert results.json()[0]["document_id"] == document_id

    deleted = client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 204
    assert client.get("/documents").json() == []


def test_ask_has_local_fallback_without_ollama(monkeypatch):
    client.post(
        "/documents",
        json={"title": "Notes", "content": "The project uses local language models."},
    )
    monkeypatch.setattr("app.main.ollama_answer", lambda question, sources: None)

    response = client.post("/ask", json={"question": "local language models"})

    assert response.status_code == 200
    assert response.json()["generated"] is False
    assert "local LLM" in response.json()["answer"]