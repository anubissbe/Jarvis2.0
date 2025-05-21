try:
    from chromadb import PersistentClient
except ImportError:  # pragma: no cover - chromadb not installed in testing env
    PersistentClient = None


def get_vector_store():
    """Return a ChromaDB client if available."""
    if PersistentClient is None:
        return None
    return PersistentClient()
