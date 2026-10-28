from fastapi.testclient import TestClient

import main
from ingest import ingest_documents
from tests.test_ingest import FakeEmbeddings


client = TestClient(main.app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "index_ready" in body


def test_ask_requires_question():
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_rejects_blank_question():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_ask_without_index_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path / "missing-index"))
    main._vectorstore = None
    main._loaded_mtime = None
    main.get_embeddings.cache_clear()
    response = client.post("/ask", json={"question": "Where is headquarters?"})
    assert response.status_code == 503


def test_ask_returns_answer_and_sources(monkeypatch, tmp_path):
    docs = tmp_path / "docs"
    index = tmp_path / "index"
    docs.mkdir()
    (docs / "handbook.txt").write_text(
        "Northwind Analytics headquarters is in Seattle, Washington.",
        encoding="utf-8",
    )
    embeddings = FakeEmbeddings()
    ingest_documents(documents_path=docs, index_path=index, embeddings=embeddings)

    monkeypatch.setenv("FAISS_INDEX_PATH", str(index))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    main._vectorstore = None
    main._loaded_mtime = None
    main.get_embeddings.cache_clear()
    monkeypatch.setattr(main, "get_embeddings", lambda: embeddings)
    monkeypatch.setattr(main, "generate_answer", lambda question, context: "Seattle, Washington")

    response = client.post("/ask", json={"question": "Where is Northwind headquarters?"})
    assert response.status_code == 200
    body = response.json()
    assert "Seattle" in body["answer"]
    assert body["sources"]
    assert body["sources"][0]["source"] == "handbook.txt"
