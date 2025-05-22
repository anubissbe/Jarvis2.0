from chromadb import PersistentClient
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


def get_vector_store(path: str = "./chroma") -> Chroma:
    """Return a Chroma vector store backed by a persistent client."""
    client = PersistentClient(path=path)
    embeddings = HuggingFaceEmbeddings()
    return Chroma(client=client, embedding_function=embeddings)
