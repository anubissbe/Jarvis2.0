from __future__ import annotations

from urllib.parse import urlparse
import uuid

import chromadb
from langchain.embeddings import OllamaEmbeddings

from ..config import settings

EMBED_MODEL = "nomic-embed-text"


def _get_client() -> chromadb.HttpClient:
    parsed = urlparse(settings.chroma_db_url)
    return chromadb.HttpClient(host=parsed.hostname, port=parsed.port)


def get_vector_store(collection_name: str = "conversations") -> chromadb.Collection:
    client = _get_client()
    return client.get_or_create_collection(name=collection_name)


def add_conversation_snippet(
    text: str,
    metadata: dict | None = None,
    collection_name: str = "conversations",
) -> None:
    collection = get_vector_store(collection_name)
    embedder = OllamaEmbeddings(base_url=settings.ollama_base_url, model=EMBED_MODEL)
    vector = embedder.embed_query(text)
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[text],
        embeddings=[vector],
        metadatas=[metadata or {}],
    )


def query_conversation_snippets(
    query: str,
    n_results: int = 5,
    collection_name: str = "conversations",
) -> list[dict]:
    collection = get_vector_store(collection_name)
    embedder = OllamaEmbeddings(base_url=settings.ollama_base_url, model=EMBED_MODEL)
    query_vector = embedder.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    matches = []
    for rid, doc, meta, dist in zip(
        results.get("ids", [[]])[0],
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        matches.append({"id": rid, "text": doc, "metadata": meta, "distance": dist})
    return matches
