import os
import importlib
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_healthz():
    from app.main import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/healthz")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "mock_llm" in data

@pytest.mark.asyncio
async def test_docs_files_empty(tmp_path, monkeypatch):
    # Point DOCS_DIR to a temp folder for isolation
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    monkeypatch.setenv("DOCS_DIR", str(docs_dir))

    # Reload modules that depend on config
    import app.config as config
    import app.main as main
    importlib.reload(config)
    importlib.reload(main)

    async with AsyncClient(app=main.app, base_url="http://test") as ac:
        r = await ac.get("/docs/files")
        assert r.status_code == 200
        data = r.json()
        assert "files" in data
        assert data["files"] == []

@pytest.mark.asyncio
async def test_docs_upload_and_list(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    monkeypatch.setenv("DOCS_DIR", str(docs_dir))

    import app.config as config
    import app.main as main
    importlib.reload(config)
    importlib.reload(main)

    async with AsyncClient(app=main.app, base_url="http://test") as ac:
        # upload .md
        files = {"file": ("note.md", b"# Hello\nThis is a test doc.\n", "text/markdown")}
        r = await ac.post("/docs/upload", files=files)
        assert r.status_code == 200
        up = r.json()
        assert up["name"] == "note.md"
        assert up["size_bytes"] > 0

        # list should include it
        r2 = await ac.get("/docs/files")
        assert r2.status_code == 200
        lst = r2.json()["files"]
        assert any(x["name"] == "note.md" for x in lst)

@pytest.mark.asyncio
async def test_docs_upload_rejects_wrong_extension(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    monkeypatch.setenv("DOCS_DIR", str(docs_dir))

    import app.config as config
    import app.main as main
    importlib.reload(config)
    importlib.reload(main)

    async with AsyncClient(app=main.app, base_url="http://test") as ac:
        files = {"file": ("bad.pdf", b"%PDF-1.4", "application/pdf")}
        r = await ac.post("/docs/upload", files=files)
        assert r.status_code == 400
        assert "Only .md and .txt" in r.text

@pytest.mark.asyncio
async def test_ask_retrieval_only(tmp_path, monkeypatch):
    # Ensure retrieval_only works without Ollama
    docs_dir = tmp_path / "docs"
    chroma_dir = tmp_path / "chroma"
    docs_dir.mkdir()
    chroma_dir.mkdir()

    (docs_dir / "kb.md").write_text(
        "This demo uses Chroma as a local vector store and Ollama for the local LLM.",
        encoding="utf-8",
    )

    monkeypatch.setenv("DOCS_DIR", str(docs_dir))
    monkeypatch.setenv("CHROMA_DIR", str(chroma_dir))

    import app.config as config
    import app.ingest as ingest
    import app.main as main
    importlib.reload(config)
    importlib.reload(ingest)
    importlib.reload(main)

    # Ingest first
    async with AsyncClient(app=main.app, base_url="http://test") as ac:
        r_ing = await ac.post("/ingest")
        assert r_ing.status_code == 200
        assert r_ing.json()["ingested_chunks"] > 0

        r = await ac.post("/ask", json={"question": "What vector store does the demo use?", "retrieval_only": True})
        assert r.status_code == 200
        data = r.json()
        assert "sources" in data
        assert len(data["sources"]) >= 1
        assert any("chroma" in s["text"].lower() for s in data["sources"])
