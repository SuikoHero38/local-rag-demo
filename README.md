![CI](https://github.com/suikohero38/local-rag-demo/actions/workflows/ci.yml/badge.svg)

# Local RAG Demo (FastAPI + Ollama)

A minimal Retrieval-Augmented Generation (RAG) demo that runs fully locally:
- FastAPI REST service for upload/ingest/ask
- Chroma vector store for retrieval
- Sentence-Transformers embeddings for indexing
- Ollama as the local LLM runtime (no paid APIs)

The goal is to showcase a clean end-to-end RAG pipeline with reproducible setup, clear API contracts, and practical troubleshooting.

---

## What this demo supports

- Upload documents via API (or mount `./docs`)
- Ingest `.md` / `.txt` documents into a local vector store
- Ask questions and return:
  - a generated answer (LLM)
  - retrieved sources (citations/snippets)
- Retrieval-only mode to debug the pipeline without calling the LLM

---

## Architecture (high level)

Client -> REST -> FastAPI

FastAPI:
- Document handling: upload + local storage (`/app/docs`)
- Ingestion: chunking + embeddings -> Chroma
- Retrieval: top-k chunks from Chroma
- Generation: Ollama (local model) answers using retrieved context

---

## Prerequisites

- Docker and Docker Compose
- Optional: curl (or use Swagger UI)

---

## Run (Docker Compose)

Start services:
  docker-compose up --build

The stack runs:
- API service (FastAPI): host port 8001 -> container port 8000
- Ollama runtime: host port 11434 -> container port 11434

Open:
- Swagger UI: http://localhost:8001/docs
- OpenAPI JSON: http://localhost:8001/openapi.json
- Health check: http://localhost:8001/healthz

Note:
- Opening http://localhost:8001/ may return 404. This is expected because this is an API service.

---

## First-time setup: pull a local model

In a new terminal (while compose is running), pull an affordable/lightweight model:

  docker-compose exec ollama ollama pull llama3.2:3b

Check installed models:
  docker-compose exec ollama ollama list

You can change the model via environment variable `OLLAMA_MODEL` in docker-compose.yml.

---

## Add documents

This demo ingests markdown/text documents.

Option A (recommended): Upload via Swagger UI
1) Open http://localhost:8001/docs
2) Use POST /docs/upload and upload a .md or .txt file
3) Then call POST /ingest

Option B: Put files into ./docs
1) Create docs folder (if missing):
   mkdir -p docs
2) Add a file:
   echo "This demo uses Chroma and Ollama." > docs/sample.md

---

## Ingest

Ingestion builds the vector index from files under ./docs.

  curl -X POST "http://localhost:8001/ingest"

Example response:
  {"ingested_chunks": 12, "files": 3}

If it returns 0 chunks:
- Ensure you have .md/.txt files under ./docs
- Then run ingest again

---

## Ask (RAG)

Ask a question and receive an answer plus sources:

  curl -X POST "http://localhost:8001/ask" \
    -H "Content-Type: application/json" \
    -d '{"question":"What do you use for the local LLM runtime?"}'

Example response (shape):
  {
    "answer": "Ollama.",
    "sources": [
      {"doc":"sample.md","chunk":0,"text":"..."}
    ]
  }

---

## Retrieval-only mode (debug)

To validate retrieval without calling the LLM:

  curl -X POST "http://localhost:8001/ask" \
    -H "Content-Type: application/json" \
    -d '{"question":"What do you use for the local LLM runtime?","retrieval_only":true}'

Expected:
- `sources` should be populated
- `answer` may be empty or minimal depending on implementation

This mode is helpful when:
- the model is not downloaded yet
- you want to test chunking/retrieval quality

---

## API reference (summary)

- GET /healthz
  Returns service status.

- POST /docs/upload
  Upload a .md/.txt document.

- POST /ingest
  Chunk + embed + store in Chroma.

- POST /ask
  Request body:
    {
      "question": "string",
      "retrieval_only": false
    }
  Returns:
    {
      "answer": "string",
      "sources": [{"doc":"...", "chunk":0, "text":"..."}]
    }

Full API schema:
- http://localhost:8001/openapi.json

---

## Common issues

1) POST /ask fails or returns empty answers
- Ensure Ollama is running:
  docker-compose ps
- Ensure the model exists:
  docker-compose exec ollama ollama list
- Pull model if needed:
  docker-compose exec ollama ollama pull llama3.2:3b

2) POST /ingest returns 0 chunks
- Ensure docs exist under ./docs and are .md/.txt
- Try adding a small sample file and ingest again

3) Reset the vector store
If you want to start fresh:
  docker-compose down
  rm -rf chroma/*
  docker-compose up --build
  curl -X POST "http://localhost:8001/ingest"

---

## Why this is intentionally minimal

This demo prioritizes clarity over scale. For a production-grade system, consider adding:
- chunk deduplication and better source ranking
- hybrid retrieval (BM25 + vectors)
- access control (API keys) and rate limiting
- observability (structured logs, metrics)
- a scalable vector DB setup (or sharding)

---

## License

MIT