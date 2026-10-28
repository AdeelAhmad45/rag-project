"""Ingest trusted PDF/TXT files into a local FAISS index."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List, Sequence

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(20 * 1024 * 1024)))
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def documents_dir() -> Path:
    return Path(os.getenv("DOCUMENTS_PATH", "./documents")).resolve()


def index_dir() -> Path:
    return Path(os.getenv("FAISS_INDEX_PATH", "./faiss_index")).resolve()


def index_files_present(path: Path | None = None) -> bool:
    root = path or index_dir()
    return (root / "index.faiss").is_file() and (root / "index.pkl").is_file()


def create_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)


def _safe_source_name(path: Path) -> str:
    return path.name


def iter_source_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {root}")
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            logger.info("Skipping unsupported file: %s", path)
            continue
        yield path


def load_file(path: Path) -> List[Document]:
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds size limit ({MAX_FILE_BYTES} bytes): {path.name}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
    elif suffix == ".txt":
        try:
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
        except UnicodeDecodeError:
            loader = TextLoader(str(path), encoding="latin-1")
            docs = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    for doc in docs:
        metadata = dict(doc.metadata or {})
        metadata["source"] = _safe_source_name(path)
        metadata["source_path"] = str(path)
        if "page" in metadata and metadata["page"] is not None:
            metadata["page"] = int(metadata["page"])
        doc.metadata = metadata
    return docs


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(list(documents))
    for idx, chunk in enumerate(chunks):
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata["chunk_index"] = idx
    return chunks


def save_index(vectorstore: FAISS, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(path))


def load_trusted_index(path: Path, embeddings) -> FAISS:
    """Load FAISS only from a local configured directory, never from request uploads."""
    configured = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
    if ".." in Path(configured).parts:
        raise ValueError("Refusing to load a FAISS index path that contains parent-directory segments")
    resolved = path.resolve()
    if not index_files_present(resolved):
        raise FileNotFoundError(f"Trusted FAISS index not found in {resolved}")
    # LangChain stores docstore metadata in pickle; only load indexes this service wrote.
    return FAISS.load_local(
        str(resolved),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def ingest_documents(
    documents_path: Path | None = None,
    index_path: Path | None = None,
    embeddings=None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    docs_root = Path(documents_path) if documents_path else documents_dir()
    store_root = Path(index_path) if index_path else index_dir()
    embedder = embeddings or create_embeddings()

    raw_docs: List[Document] = []
    sources: List[str] = []
    for file_path in iter_source_files(docs_root):
        loaded = load_file(file_path)
        if not loaded:
            logger.warning("No content loaded from %s", file_path)
            continue
        raw_docs.extend(loaded)
        sources.append(file_path.name)

    if not raw_docs:
        raise ValueError(f"No PDF or TXT documents found in {docs_root}")

    chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorstore = FAISS.from_documents(chunks, embedder)
    save_index(vectorstore, store_root)

    return {
        "documents": len(sources),
        "chunks": len(chunks),
        "sources": sources,
        "index_path": str(store_root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest documents into the FAISS index")
    parser.add_argument("--documents", type=Path, default=None)
    parser.add_argument("--index", type=Path, default=None)
    args = parser.parse_args(argv)
    result = ingest_documents(documents_path=args.documents, index_path=args.index)
    logger.info(
        "Ingested %s files into %s chunks at %s",
        result["documents"],
        result["chunks"],
        result["index_path"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
