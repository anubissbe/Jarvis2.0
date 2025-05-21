from urllib.parse import urlparse

import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

from ..config import settings


def get_vector_store() -> Chroma:
    """Return a remote Chroma vector store connected to the configured service."""
    embeddings = OllamaEmbeddings(base_url=settings.ollama_base_url, model="llama3")
    parsed = urlparse(settings.chroma_db_url)
    client = chromadb.HttpClient(host=parsed.hostname, port=parsed.port or 80)
    return Chroma(
        client=client,
        collection_name="jarvis",
        embedding_function=embeddings,
    )


def add_memory(vector_store: Chroma, text: str) -> None:
    """Persist a text snippet to the vector store."""
    vector_store.add_texts([text])
