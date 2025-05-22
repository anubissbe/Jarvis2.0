# ruff: noqa: E402
import pytest

chromadb = pytest.importorskip("chromadb")

from app.memory import vector_memory


class FakeEmbeddings:
    """Deterministic dummy embeddings for testing."""

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        return [float(len(text))]


def test_add_and_retrieve_document(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_memory, "HuggingFaceEmbeddings", lambda: FakeEmbeddings())
    store = vector_memory.get_vector_store(path=str(tmp_path))
    store.add_texts(["hello world"], ids=["1"])
    results = store.similarity_search("hello", k=1)
    assert results
    assert results[0].page_content == "hello world"
    # cleanup
    store._client.reset()

