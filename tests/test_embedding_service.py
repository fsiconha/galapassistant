import os
import tempfile
import pytest

from langchain.docstore.document import Document as LangchainDocument
from galapassistant.apps.assistant.services.embedding_service import (
    EmbeddingService,
    DistanceStrategy,
)


class MockTextSplitter:
    """A mock text splitter that returns the input documents unchanged."""

    def split_documents(self, docs):
        return docs


class MockEmbeddings:
    """A mock embeddings class to bypass heavy computations."""

    def __init__(self, model_name, multi_process, model_kwargs, encode_kwargs):
        self.model_name = model_name


def mock_faiss_from_documents(docs, embedding_model, distance_strategy):
    """A mock replacement for FAISS.from_documents that returns a dict with inputs."""
    return {
        "docs": docs,
        "embedding_model": embedding_model,
        "distance_strategy": distance_strategy,
    }


def test_load_knowledge_base(tmp_path):
    """
    Test that load_knowledge_base reads a file and returns a single Document
    whose page_content matches the file content.
    """
    content = "This is a test content for the knowledge base."
    temp_file = tmp_path / "test.txt"
    temp_file.write_text(content, encoding="utf-8")

    service = EmbeddingService()
    documents = service.load_knowledge_base(str(temp_file))

    assert isinstance(documents, list)
    assert len(documents) == 1
    assert documents[0].page_content == content


def test_build_vector_database_with_custom_knowledge_base(monkeypatch):
    """
    Test that build_vector_database correctly builds a vector database when a custom
    knowledge base is provided.
    """
    service = EmbeddingService()
    test_doc = LangchainDocument(page_content="Test content for splitting.")
    knowledge_base = [test_doc]

    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.embedding_service.AutoTokenizer.from_pretrained",
        lambda name: "mock_tokenizer",
    )
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.embedding_service.RecursiveCharacterTextSplitter.from_huggingface_tokenizer",
        lambda tokenizer, chunk_size, chunk_overlap, add_start_index, strip_whitespace, separators: MockTextSplitter(),
    )
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.embedding_service.HuggingFaceEmbeddings",
        MockEmbeddings,
    )
    MockFAISS = type("MockFAISS", (), {"from_documents": staticmethod(mock_faiss_from_documents)})
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.embedding_service.FAISS", MockFAISS
    )

    result = service.build_vector_database(
        chunk_size=100, knowledge_base=knowledge_base, tokenizer_name="mock_model"
    )

    assert isinstance(result, dict)
    assert result["docs"] == knowledge_base
    assert result["distance_strategy"] == DistanceStrategy.COSINE


def test_build_vector_database_with_default_knowledge_base(monkeypatch):
    """
    Test that build_vector_database uses the default knowledge base when None is provided.
    """
    service = EmbeddingService()
    default_doc = LangchainDocument(page_content="Default test content.")
    monkeypatch.setattr(service, "load_knowledge_base", lambda file_path: [default_doc])

    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.embedding_service.AutoTokenizer.from_pretrained",
        lambda name: "mock_tokenizer",
    )
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.embedding_service.RecursiveCharacterTextSplitter.from_huggingface_tokenizer",
        lambda tokenizer, chunk_size, chunk_overlap, add_start_index, strip_whitespace, separators: MockTextSplitter(),
    )
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.embedding_service.HuggingFaceEmbeddings",
        MockEmbeddings,
    )
    MockFAISS = type("MockFAISS", (), {"from_documents": staticmethod(mock_faiss_from_documents)})
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.embedding_service.FAISS", MockFAISS
    )

    result = service.build_vector_database(
        chunk_size=100, knowledge_base=None, tokenizer_name="mock_model"
    )

    assert isinstance(result, dict)
    assert result["docs"] == [default_doc]
