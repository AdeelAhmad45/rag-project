from pathlib import Path

import pytest

from ingest import load_trusted_index


def test_refuses_parent_directory_index_path(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FAISS_INDEX_PATH", "../untrusted")
    with pytest.raises(ValueError, match="parent-directory"):
        load_trusted_index(tmp_path, embeddings=None)
