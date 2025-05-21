from chromadb import HttpClient
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

from ..config import settings


def get_vector_store():
    """Return a LangChain vector store connected to the configured Chroma DB."""
    client = HttpClient(host=settings.chroma_db_url)
    embeddings = OllamaEmbeddings(base_url=settings.ollama_base_url, model="llama3")
    return Chroma(client=client, embedding_function=embeddings)
