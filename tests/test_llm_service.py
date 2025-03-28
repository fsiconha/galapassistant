import pytest

from galapassistant.apps.assistant.services.llm_service import AssistantLLMService

class MockChain:
    """A mock chain to simulate the rag_chain's invoke method."""
    def __init__(self, response):
        self.response = response

    def invoke(self, query):
        return self.response

class ErrorChain:
    """A mock chain that always raises an error when invoked."""
    def invoke(self, query):
        raise ValueError("Test error")

@pytest.fixture
def mock_assistant_service():
    """
    Create an instance of AssistantLLMService without running __init__
    and replace its rag_chain with a MockChain.
    """
    service = AssistantLLMService.__new__(AssistantLLMService)
    service.rag_chain = MockChain("mock response")
    return service

def test_get_response_returns_mock_response(mock_assistant_service):
    """
    Test that get_response returns the expected mock response.
    """
    query = "What is evolution?"
    response = mock_assistant_service.get_response(query)
    assert response == "mock response"

def test_get_response_propagates_error():
    """
    Test that if the underlying chain raises an error, it is propagated.
    """
    service = AssistantLLMService.__new__(AssistantLLMService)
    service.rag_chain = ErrorChain()
    with pytest.raises(ValueError, match="Test error"):
        service.get_response("Any query")
