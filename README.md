# RAG Project

A retrieval-augmented question-answering app with:

- A Python FastAPI service using Hugging Face sentence-transformer embeddings, FAISS, and Ollama
- A Node.js Express gateway for chat requests
- PDF and text document ingestion with source citations

## How it works

```text
Client -> Node Express :3000 POST /api/chat
       -> Python FastAPI :8000 POST /ask
       -> FAISS retrieves the 4 most relevant chunks by default
       -> Ollama generates an answer using the retrieved context
```

Documents are read from `python-rag/documents/`. The FAISS index is stored in
`python-rag/faiss_index/`. By default, embeddings use
`sentence-transformers/all-MiniLM-L6-v2`; Ollama uses the `llama3.2` model.

## Prerequisites

- Python 3.11+
- Node.js 20+
- [Ollama](https://ollama.com/) installed and running
- The Ollama model pulled locally:

```powershell
ollama pull llama3.2
```

No OpenAI API key is required. The first Python ingestion may download the
embedding model from Hugging Face.

## Run locally

### 1. Install Python dependencies and ingest documents

In PowerShell, from the repository root:

```powershell
cd rag-project\python-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put `.pdf` and `.txt` files in `python-rag\documents\`. Then build or refresh
the FAISS index:

```powershell
python ingest.py
```

### 2. Start the Python API

Keep the virtual environment active:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

In another terminal, check the API:

```powershell
curl.exe http://localhost:8000/health
```

Ask a question directly:

```powershell
curl.exe -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"Where is Northwind Analytics headquartered?"}'
```

### 3. Start the Node.js gateway

In another terminal:

```powershell
cd rag-project\node-app
npm install
$env:PYTHON_RAG_URL = "http://localhost:8000"
$env:NODE_PORT = "3000"
npm start
```

Send a chat request through Node:

```powershell
curl.exe -X POST http://localhost:3000/api/chat `
  -H "Content-Type: application/json" `
  -d '{"question":"Where is Northwind Analytics headquartered?"}'
```

The request can use either a `question` or `message` field.

## Configuration

These environment variables are optional:

| Variable | Default | Description |
| --- | --- | --- |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model used to generate answers; pull it with `ollama pull <model>` first. |
| `RETRIEVE_K` | `4` | Number of FAISS chunks retrieved for each question. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face sentence-transformer model used for document and query embeddings. |
| `DOCUMENTS_PATH` | `./documents` | Directory containing source PDF and text files. |
| `FAISS_INDEX_PATH` | `./faiss_index` | Directory used to save and load the FAISS index. |
| `CHUNK_SIZE` | `1000` | Maximum characters per document chunk during ingestion. |
| `CHUNK_OVERLAP` | `200` | Character overlap between adjacent chunks. |
| `PYTHON_RAG_URL` | `http://localhost:8000` | Python API URL used by the Node gateway. |
| `NODE_PORT` | `3001` | Port used by the Node gateway. |

For example, to use another Ollama model in PowerShell, pull it and set the
variable before starting the Python API:

```powershell
ollama pull qwen2.5
$env:OLLAMA_MODEL = "qwen2.5"
```

## API

### Python service

- `GET /health` returns the service status, whether the FAISS index is ready,
  and the current retrieval count.
- `POST /ask` accepts `{ "question": "..." }` and returns an answer and source
  snippets:

```json
{
  "answer": "Example answer.",
  "sources": [
    { "source": "handbook.txt", "page": null, "snippet": "Relevant text..." }
  ]
}
```

The index must exist before asking a question. If Ollama is unavailable or the
model is not installed, answer generation fails with an error.

### Node gateway

- `GET /health` returns the gateway status.
- `POST /api/chat` accepts `{ "question": "..." }` or `{ "message": "..." }`
  and returns the answer and sources from the Python service.

The gateway validates questions and maps upstream failures and timeouts to
appropriate HTTP errors.

## Tests

Python tests (run from `python-rag` with the virtual environment active):

```powershell
pytest -q
```

Node tests (run from `node-app`):

```powershell
npm test
```

The Python tests use fake embeddings and stub answer generation; they do not
need a running Ollama server or a prebuilt FAISS index.

## Docker

The repository includes Dockerfiles and a Compose file, but the Compose
configuration has not been validated for the Ollama-based local setup. In
particular, Ollama must be running on the host and reachable from the Python
container. Use the local instructions above unless you have configured that
container-to-host connection.

## Security notes

- Only `.pdf` and `.txt` files are ingested; files over the configured size
  limit are rejected.
- FAISS `index.pkl` uses pickle-based storage. Only load an index created from
  trusted documents by this project.
- There is no public document-upload endpoint; add files to the documents
  directory yourself.

## Project layout

```text
rag-project/
  docker-compose.yml
  README.md
  node-app/
    server.js
    package.json
    Dockerfile
  python-rag/
    main.py
    ingest.py
    requirements.txt
    Dockerfile
    documents/
    faiss_index/
```
