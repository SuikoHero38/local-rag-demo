from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from pathlib import Path

from .ingest import ingest_docs
from .rag import build_qa, retrieve_only
from .config import MOCK_LLM, DOCS_DIR

app = FastAPI(
    title="Local RAG Demo (Ollama)",
    version="0.2.0",
    description=(
        "A minimal local RAG service using Ollama (local LLM) + Chroma (vector store). "
        "Ingest documents from ./docs and answer questions with sources."
    ),
)

# -----------------------
# Schemas (Swagger-friendly)
# -----------------------
class IngestOut(BaseModel):
    ingested_chunks: int
    files: int

class SourceItem(BaseModel):
    doc: str | None
    chunk: int | None
    text: str

class AskIn(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        examples=["What vector store does the demo use?"],
        description="Natural language question to query the local knowledge base.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional override for retrieval top_k (if implemented).",
    )
    retrieval_only: bool = Field(
        default=False,
        description="If true, returns sources only (no LLM call). Useful for tests/CI.",
    )

class AskOut(BaseModel):
    answer: str
    sources: list[SourceItem]

class DocsFileItem(BaseModel):
    name: str
    size_bytes: int

class DocsListOut(BaseModel):
    files: list[DocsFileItem]

# -----------------------
# Endpoints
# -----------------------
@app.get("/healthz")
async def healthz():
    return {"status": "ok", "mock_llm": MOCK_LLM}

@app.post("/ingest", response_model=IngestOut, summary="Ingest docs from ./docs into Chroma")
async def ingest():
    res = ingest_docs()
    if isinstance(res, int):
        # backward-compatible: interpret as ingested_chunks
        res = {"ingested_chunks": res, "files": 0}
    return IngestOut(**res)


@app.post("/ask", response_model=AskOut, summary="Ask a question (RAG) and return answer + sources")
async def ask(payload: AskIn):
    q = payload.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    if payload.retrieval_only:
        sources = retrieve_only(q)
        return AskOut(answer="", sources=[SourceItem(**s) for s in sources])

    qa = build_qa()
    try:
        res = qa({"query": q})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM/RAG error: {e}")

    sources = []
    for d in res.get("source_documents", []):
        meta = d.metadata or {}
        sources.append(
            SourceItem(
                doc=meta.get("doc"),
                chunk=meta.get("chunk"),
                text=d.page_content[:400],
            )
        )
    return AskOut(answer=res.get("result", ""), sources=sources)

@app.get("/docs/files", response_model=DocsListOut, summary="List available knowledge base files in ./docs")
async def list_docs_files():
    docs_path = Path(DOCS_DIR)
    if not docs_path.exists():
        return DocsListOut(files=[])

    allowed = {".md", ".txt"}
    out = []
    for fp in sorted(docs_path.glob("**/*")):
        if fp.is_file() and fp.suffix.lower() in allowed:
            out.append(DocsFileItem(name=str(fp.relative_to(docs_path)), size_bytes=fp.stat().st_size))
    return DocsListOut(files=out)

@app.post("/docs/upload", response_model=DocsFileItem, summary="Upload a txt/md file into ./docs")
async def upload_doc(file: UploadFile = File(...)):
    docs_path = Path(DOCS_DIR)
    docs_path.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".md", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .md and .txt files are supported")

    target = docs_path / (file.filename or "uploaded.txt")
    content = await file.read()
    target.write_bytes(content)

    return DocsFileItem(name=target.name, size_bytes=target.stat().st_size)
