from chromadb import HttpClient

from ..config import settings


def get_vector_store() -> HttpClient:
    """Return a ChromaDB HTTP client connected to the configured server."""
    return HttpClient(host=settings.chroma_db_url)
