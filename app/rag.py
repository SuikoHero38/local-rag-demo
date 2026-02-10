from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

from .config import (
    CHROMA_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TOP_K,
    EMBEDDING_MODEL,
    MOCK_LLM,
)

def _vectorstore():
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    #vs = Chroma(
    #    collection_name="kb",
    #    embedding_function=emb,
    #    persist_directory=CHROMA_DIR,
    #)
    vs = Chroma(collection_name="kbase", embedding_function=emb, persist_directory=CHROMA_DIR)
    return vs

def retrieve_only(query: str, k: int | None = None) -> list[dict]:
    """Retrieve top-k documents without calling the LLM (useful for debugging/tests)."""
    vs = _vectorstore()
    kk = k or TOP_K
    docs = vs.as_retriever(search_kwargs={"k": kk}).get_relevant_documents(query)

    out: list[dict] = []
    for d in docs:
        meta = d.metadata or {}
        out.append(
            {
                "doc": meta.get("doc"),
                "chunk": meta.get("chunk"),
                "text": (d.page_content or "")[:500],
            }
        )
    return out

def build_qa() -> RetrievalQA:
    vs = _vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})

    if MOCK_LLM:
        # Still build the chain, but you can avoid calling it by using retrieval_only in the API.
        llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    else:
        llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa
