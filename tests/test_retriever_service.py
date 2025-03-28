import pytest

from galapassistant.apps.assistant.services.retriever_service import RetrieverService


class DummyDocument:
    """A dummy document with page_content attribute."""

    def __init__(self, content):
        self.page_content = content


class DummyVectorStore:
    """A dummy vector store that returns a list of dummy documents."""

    def similarity_search(self, query, k):
        return [DummyDocument(f"Content {i}") for i in range(k)]


def dummy_build_vector_database(self):
    """Dummy replacement for EmbeddingService.build_vector_database."""
    return DummyVectorStore()


def test_retrieve_returns_formatted_output(monkeypatch):
    """
    Test that retrieve returns the expected formatted string when the vector store
    returns a list of documents.
    """
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.embedding_service.EmbeddingService.build_vector_database",
        dummy_build_vector_database,
    )
    retriever = RetrieverService()
    k = 3
    query = "Test query"
    result = retriever.retrieve(query, k=k)
    expected = "\nRetrieved documents:\n" + "".join(
        [f"= Document {i} =\nContent {i}\n" for i in range(k)]
    )
    assert result == expected


def test_retrieve_empty(monkeypatch):
    """
    Test that retrieve returns only the header when the vector store returns an empty list.
    """
    class EmptyVectorStore:
        def similarity_search(self, query, k):
            return []

    def dummy_build_vector_database_empty(self):
        return EmptyVectorStore()

    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.embedding_service.EmbeddingService.build_vector_database",
        dummy_build_vector_database_empty,
    )
    retriever = RetrieverService()
    query = "Test query"
    result = retriever.retrieve(query)
    expected = "\nRetrieved documents:\n"
    assert result == expected
