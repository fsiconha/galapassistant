import pytest

from galapassistant.apps.assistant.services.retrieval import RetrieverTool


class DummyDocument:
    """A dummy document with page_content attribute."""

    def __init__(self, content):
        self.page_content = content


class MockVectorStore:
    """A dummy vector store that returns a list of dummy documents."""

    def similarity_search(self, query, k):
        return [DummyDocument(f"Content {i}") for i in range(k)]


def dummy_build_vector_database(self):
    """Dummy replacement for IndexingService.build_vector_database."""
    return MockVectorStore()


def test_retrieve_returns_formatted_output(monkeypatch):
    """
    Test that retrieve returns the expected formatted string when the vector store
    returns a list of documents.
    """
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.indexing.IndexingService.build_vector_database",
        dummy_build_vector_database,
    )
    retriever = RetrieverTool()
    k = 3
    query = "Test query"
    actual_docs = retriever(query, k=k)
    assert "Retrieved documents" in actual_docs


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
        "galapassistant.apps.assistant.services.indexing.IndexingService.build_vector_database",
        dummy_build_vector_database_empty,
    )
    retriever = RetrieverTool()
    query = "Test query"
    actual_result = retriever(query, 2)
    actual_expected = "\nRetrieved documents:\n"
    assert actual_result == actual_expected
