from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
import chromadb
from urllib.parse import urlparse

from ..config import settings


def get_vector_store():
    embeddings = OllamaEmbeddings(base_url=settings.ollama_base_url, model="llama3")
    parsed = urlparse(settings.chroma_db_url)
    client = chromadb.HttpClient(host=parsed.hostname, port=parsed.port)
    return Chroma(
        client=client,
        collection_name="conversations",
        embedding_function=embeddings,
    )
