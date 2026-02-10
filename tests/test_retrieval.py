import importlib
import pytest

@pytest.mark.asyncio
async def test_retrieve_only_direct(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    chroma_dir = tmp_path / "chroma"
    docs_dir.mkdir()
    chroma_dir.mkdir()

    (docs_dir / "a.md").write_text("Chroma is used as a local vector store.", encoding="utf-8")

    monkeypatch.setenv("DOCS_DIR", str(docs_dir))
    monkeypatch.setenv("CHROMA_DIR", str(chroma_dir))

    import app.config as config
    import app.ingest as ingest
    import app.rag as rag
    importlib.reload(config)
    importlib.reload(ingest)
    importlib.reload(rag)

    ingest.ingest_docs()
    sources = rag.retrieve_only("What vector store does the demo use?")
    assert len(sources) >= 1
    assert any("chroma" in (s["text"] or "").lower() for s in sources)
