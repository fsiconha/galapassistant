import pytest
from galapassistant.apps.assistant.services.llm_service import AssistantLLMService

class MockChain:
    """A mock chain to simulate the rag_chain's invoke method."""
    def __init__(self, response):
        self.response = response

    def __or__(self, other):
        return self

    def invoke(self, query):
        return self.response

@pytest.fixture
def mock_assistant_service(monkeypatch):
    """
    Create an instance of AssistantLLMService without running __init__
    and replace its required attributes with mock objects.
    """
    service = AssistantLLMService.__new__(AssistantLLMService)
    service.prompt = MockChain("mock response")
    service.llm = MockChain("mock response")
    service.rag_chain = MockChain("mock response")
    from galapassistant.apps.assistant.services.retriever_service import RetrieverService
    monkeypatch.setattr(RetrieverService, "retrieve", lambda self, query: "dummy retriever")
    return service

def test_get_response_returns_mock_response(mock_assistant_service):
    """
    Test that get_response returns the expected mock response.
    """
    query = "What is evolution?"
    response = mock_assistant_service.get_response(query)
    assert response == "mock response"
