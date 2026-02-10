import os

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Storage paths
CHROMA_DIR = os.getenv("CHROMA_DIR", "/app/chroma")
DOCS_DIR = os.getenv("DOCS_DIR", "/app/docs")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Embeddings (local)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# Test/CI helper
MOCK_LLM = os.getenv("MOCK_LLM", "0") == "1"
