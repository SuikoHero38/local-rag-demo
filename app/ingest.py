from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from .config import (
    DOCS_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)

COLLECTION_NAME = "kbase"  # must be 3-63 chars for recent Chroma versions


def ingest_docs() -> dict:
    """
    Load .md/.txt files from DOCS_DIR, chunk them, embed, and store in Chroma.
    Returns a mapping compatible with IngestOut:
      {"ingested_chunks": int, "files": int}
    """
    docs_path = Path(DOCS_DIR)
    chroma_path = Path(CHROMA_DIR)

    # Ensure directories exist
    docs_path.mkdir(parents=True, exist_ok=True)
    chroma_path.mkdir(parents=True, exist_ok=True)

    allowed = {".md", ".txt"}
    files = [
        p
        for p in docs_path.glob("**/*")
        if p.is_file() and p.suffix.lower() in allowed
    ]

    # If no docs, do not crash. Return 0.
    if not files:
        return {"ingested_chunks": 0, "files": 0}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents: list[Document] = []
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        chunks = splitter.split_text(text)
        for i, ch in enumerate(chunks):
            documents.append(
                Document(
                    page_content=ch,
                    metadata={"doc": fp.name, "chunk": i},
                )
            )

    # If docs exist but all were empty after filtering
    if not documents:
        return {"ingested_chunks": 0, "files": len(files)}

    # Build embeddings + vector store
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=emb,
        persist_directory=str(chroma_path),
    )

    vs.add_documents(documents)

    # Newer Chroma auto-persists. Keep this guarded for compatibility.
    try:
        vs.persist()
    except Exception:
        pass

    return {"ingested_chunks": len(documents), "files": len(files)}
