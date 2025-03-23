import pytest
from galapassistant.apps.assistant.services.embedding_service import EmbeddingService

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
max_tokens = 512

def fake_embed_documents(self, texts):
    """
    Fake embedding function that returns a fixed vector (768-dimensional zero vector)
    for each input text.
    """
    return [[0.0] * 768 for _ in texts]

@pytest.mark.django_db
def test_split_data_returns_chunks(mock_knowledge_base):
    """
    Test that split_data returns one or more document chunks from the fake knowledge base.
    """
    service = EmbeddingService()
    chunks = service.split_data(knowledge_base=mock_knowledge_base)
    assert len(chunks) > 0, "No chunks returned from split_data."
    for chunk in chunks:
        token_count = len(tokenizer.tokenize(chunk.page_content))
        assert token_count <= max_tokens, f"Chunk exceeds maximum token limit: {token_count} tokens."

@pytest.mark.django_db
def test_build_vector_database_creates_index(mock_knowledge_base, monkeypatch):
    """
    Test that the FAISS vector store is created and that a similarity search returns results.
    """
    service = EmbeddingService()
    chunks = service.split_data(knowledge_base=mock_knowledge_base)
    # Monkeypatch the embed_documents method of HuggingFaceEmbeddings to avoid external API calls.
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.embedding_service.HuggingFaceEmbeddings.embed_documents",
        fake_embed_documents
    )
    vector_store = service.build_vector_database(chunks)
    results = vector_store.similarity_search("evolution", k=1)
    assert len(results) > 0, "No results returned from similarity search."
