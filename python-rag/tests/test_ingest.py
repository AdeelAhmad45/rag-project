from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from ingest import ingest_documents, load_file, split_documents


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float((len(text) + i) % 11), 0.1, 0.2, 0.3] for i, text in enumerate(texts)]

    def embed_query(self, text):
        return [float(len(text) % 11), 0.1, 0.2, 0.3]


def test_load_txt_preserves_source_metadata(tmp_path: Path):
    doc = tmp_path / "handbook.txt"
    doc.write_text("Northwind Analytics headquarters is in Seattle.", encoding="utf-8")
    loaded = load_file(doc)
    assert loaded
    assert loaded[0].metadata["source"] == "handbook.txt"
    assert "Seattle" in loaded[0].page_content


def test_split_documents_keeps_source(tmp_path: Path):
    doc = tmp_path / "notes.txt"
    doc.write_text("Alpha " * 400, encoding="utf-8")
    loaded = load_file(doc)
    chunks = split_documents(loaded, chunk_size=80, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "notes.txt" for chunk in chunks)


def test_ingest_writes_faiss_index(tmp_path: Path):
    docs = tmp_path / "docs"
    index = tmp_path / "index"
    docs.mkdir()
    (docs / "policy.txt").write_text(
        "Atlas Insights retains raw telemetry for 90 days.",
        encoding="utf-8",
    )
    result = ingest_documents(
        documents_path=docs,
        index_path=index,
        embeddings=FakeEmbeddings(),
        chunk_size=50,
        chunk_overlap=10,
    )
    assert result["chunks"] >= 1
    assert (index / "index.faiss").is_file()
    assert (index / "index.pkl").is_file()


def test_ingest_rejects_empty_directory(tmp_path: Path):
    docs = tmp_path / "empty"
    docs.mkdir()
    with pytest.raises(ValueError, match="No PDF or TXT"):
        ingest_documents(documents_path=docs, index_path=tmp_path / "index", embeddings=FakeEmbeddings())
