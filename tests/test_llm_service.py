import pytest
from galapassistant.apps.assistant.services.llm_service import AssistantLLMService


def test_get_response_returns_mock_response(mock_assistant_service):
    """
    Test that get_response returns the expected mock response.
    """
    query = "What is evolution?"
    response = mock_assistant_service.get_response(query)
    assert response == "dummy response"
