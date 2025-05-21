
import chromadb

from ..config import settings


def get_vector_store():
    """Return a ChromaDB client for storing conversation embeddings."""
    return chromadb.HttpClient(host=settings.chroma_db_url)
