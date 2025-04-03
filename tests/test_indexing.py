import pytest
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument
from galapassistant.apps.assistant.services.indexing import IndexingService


class MockFAISS:
    """A mocked FAISS vector store to simulate the FAISS index."""
    def __init__(self, docs, embedding_model, distance_strategy):
        self.docs = docs
        self.embedding_model = embedding_model
        self.distance_strategy = distance_strategy

    def similarity_search(self, query, k=5):
        return self.docs[:k]


def dummy_from_documents(docs, embedding_model, distance_strategy):
    """Return a MockFAISS instance."""
    return MockFAISS(docs, embedding_model, distance_strategy)

class MockSemanticChunker:
    """A mock semantic chunker that splits each document into two chunks."""
    def split_documents(self, docs):
        chunks = []
        for doc in docs:
            text = doc.page_content
            mid = len(text) // 2
            chunk1 = LangchainDocument(page_content=text[:mid])
            chunk2 = LangchainDocument(page_content=text[mid:])
            chunks.extend([chunk1, chunk2])
        return chunks


class MockEmbeddings:
    """A mock embeddings class that bypasses heavy computations."""
    def __init__(self, model_name, multi_process, model_kwargs, encode_kwargs):
        self.model_name = model_name

    def embed_documents(self, documents):
        return [f"dummy_embedding_{i}" for i, _ in enumerate(documents)]


def test_load_knowledge_base(mock_knowledge_base):
    """
    Test that the mock_knowledge_base fixture returns a single Document
    with a page_content string.
    """
    assert isinstance(mock_knowledge_base, list)
    assert len(mock_knowledge_base) == 1
    assert hasattr(mock_knowledge_base[0], "page_content")
    assert isinstance(mock_knowledge_base[0].page_content, str)

def test_build_vector_database_with_custom_knowledge_base(monkeypatch, mock_knowledge_base):
    """
    Test that build_vector_database correctly builds a vector database when a custom
    knowledge base is provided.
    """
    service = IndexingService()

    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.indexing.SemanticChunker",
        lambda *args, **kwargs: MockSemanticChunker()
    )
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.indexing.HuggingFaceEmbeddings",
        MockEmbeddings
    )
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.indexing.FAISS",
        type("MockFAISSWrapper", (), {"from_documents": staticmethod(dummy_from_documents)})
    )
    actual_vector_store = service.build_vector_database(knowledge_base=mock_knowledge_base)
    expected_chunks = []
    for doc in mock_knowledge_base:
        text = doc.page_content
        mid = len(text) // 2
        expected_chunks.extend([
            LangchainDocument(page_content=text[:mid]),
            LangchainDocument(page_content=text[mid:])
        ])

    assert isinstance(actual_vector_store, MockFAISS)
    assert actual_vector_store.docs == expected_chunks
    assert actual_vector_store.distance_strategy == DistanceStrategy.COSINE

def test_build_vector_database_with_default_knowledge_base(monkeypatch):
    """
    Test that build_vector_database uses the default knowledge base when None is provided.
    """
    service = IndexingService()
    default_doc = LangchainDocument(page_content="Default test content.")

    monkeypatch.setattr(service, "load_knowledge_base", lambda file_path: [default_doc])
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.indexing.SemanticChunker",
        lambda *args, **kwargs: MockSemanticChunker()
    )
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.indexing.HuggingFaceEmbeddings",
        MockEmbeddings
    )
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.indexing.FAISS",
        type("MockFAISSWrapper", (), {"from_documents": staticmethod(dummy_from_documents)})
    )
    actual_vector_store = service.build_vector_database(knowledge_base=None)
    text = default_doc.page_content
    mid = len(text) // 2
    expected_chunks = [
        LangchainDocument(page_content=text[:mid]),
        LangchainDocument(page_content=text[mid:])
    ]
    assert isinstance(actual_vector_store, MockFAISS)
    assert actual_vector_store.docs == expected_chunks
    assert actual_vector_store.distance_strategy == DistanceStrategy.COSINE
