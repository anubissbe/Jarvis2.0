from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

from ..config import settings


def get_vector_store(persist_directory: str = "./chroma"):
    """Initialize or load a Chroma vector store."""
    embeddings = OllamaEmbeddings(
        base_url=settings.ollama_base_url, model="nomic-embed-text"
    )
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def store_interaction(store: Chroma, question: str, answer: str) -> None:
    """Persist a question/answer pair to the vector store."""
    text = f"Q: {question}\nA: {answer}"
    store.add_texts([text])
