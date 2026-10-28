"""FastAPI retrieval-augmented generation service."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from ingest import create_embeddings, index_dir, index_files_present, load_trusted_index

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

RETRIEVE_K = int(os.getenv("RETRIEVE_K", "4"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

app = FastAPI(title="Python RAG Service", version="1.0.0")

SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the user using only the provided context. "
    "If the context is not enough to answer, say you do not know. "
    "Do not invent facts. Mention source filenames when they support the answer."
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be empty")
        return cleaned


class Source(BaseModel):
    source: str
    page: Optional[int] = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


def _vectorstore_mtime(path: Path) -> float:
    faiss_file = path / "index.faiss"
    pkl_file = path / "index.pkl"
    return max(faiss_file.stat().st_mtime, pkl_file.stat().st_mtime)


@lru_cache(maxsize=1)
def get_embeddings():
    return create_embeddings()


_vectorstore: FAISS | None = None
_loaded_mtime: float | None = None


def get_vectorstore() -> FAISS | None:
    global _vectorstore, _loaded_mtime
    path = index_dir()
    if not index_files_present(path):
        _vectorstore = None
        _loaded_mtime = None
        return None
    mtime = _vectorstore_mtime(path)
    if _vectorstore is None or _loaded_mtime != mtime:
        _vectorstore = load_trusted_index(path, get_embeddings())
        _loaded_mtime = mtime
        logger.info("Loaded FAISS index from %s", path)
    return _vectorstore


def format_context(docs_with_scores: list[tuple[Any, float]]) -> str:
    parts: list[str] = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        location = f"{source}" + (f" (page {page})" if page is not None else "")
        parts.append(f"[{i}] Source: {location}\n{doc.page_content}")
    return "\n\n".join(parts)


def to_sources(docs_with_scores: list[tuple[Any, float]]) -> list[Source]:
    sources: list[Source] = []
    seen: set[tuple[str, Optional[int], str]] = set()
    for doc, _score in docs_with_scores:
        source_name = Path(str(doc.metadata.get("source", "unknown"))).name
        page = doc.metadata.get("page")
        page_int = int(page) if page is not None else None
        snippet = " ".join(doc.page_content.split())[:280]
        key = (source_name, page_int, snippet)
        if key in seen:
            continue
        seen.add(key)
        sources.append(Source(source=source_name, page=page_int, snippet=snippet))
    return sources


def generate_answer(question: str, context: str) -> str:
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer using only the context above."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
    except Exception as exc:
        logger.exception("Ollama answer generation failed")
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to generate an answer using Ollama. "
                "Check that Ollama is running and the model is installed."
            ),
        ) from exc

    content = response.content

    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else str(part)
            for part in content
        ).strip()

    return str(content).strip()

@app.get("/health")
def health() -> dict:
    ready = index_files_present()
    return {
        "status": "ok",
        "index_ready": ready,
        "retrieve_k": RETRIEVE_K,
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    store = get_vectorstore()
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="FAISS index is not ready. Run ingest.py against trusted documents first.",
        )

    docs_with_scores = store.similarity_search_with_score(payload.question, k=RETRIEVE_K)
    if not docs_with_scores:
        return AskResponse(
            answer="I do not know. The knowledge base did not return any relevant context.",
            sources=[],
        )

    context = format_context(docs_with_scores)
    answer = generate_answer(payload.question, context)
    return AskResponse(answer=answer, sources=to_sources(docs_with_scores))
