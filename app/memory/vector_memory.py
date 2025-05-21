from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
import chromadb

from ..config import settings


def get_vector_store():
    """Return a vector store connected to the ChromaDB service."""
    embeddings = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model="llama3",
    )
    client = chromadb.HttpClient(host=settings.chroma_db_url)
    return Chroma(
        client=client,
        collection_name="jarvis",
        embedding_function=embeddings,
    )
