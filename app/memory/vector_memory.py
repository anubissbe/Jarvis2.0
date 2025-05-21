from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

from ..config import settings


def get_vector_store():
    embeddings = OllamaEmbeddings(base_url=settings.ollama_base_url, model="llama3")
    return Chroma(embedding_function=embeddings, persist_directory="/data/chroma")
