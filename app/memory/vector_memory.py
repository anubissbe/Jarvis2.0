from chromadb import HttpClient
from uuid import uuid4
from urllib.parse import urlparse

from ..config import settings


def get_vector_store():
    """Create and return a ChromaDB client."""
    url = urlparse(settings.chroma_db_url)
    host = url.hostname or "localhost"
    port = url.port or 8000
    return HttpClient(host=host, port=port)


def add_message(client, message: str):
    """Store a message in ChromaDB."""
    collection = client.get_or_create_collection("messages")
    collection.add(documents=[message], ids=[str(uuid4())])
