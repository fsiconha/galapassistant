import pytest
from galapassistant.apps.assistant.services.llm_service import AssistantLLMService


def test_generate_answer_returns_response(mock_assistant_service):
    """
    Test that generate_answer returns the expected mock response.
    """
    query = "What is evolution?"
    response = mock_assistant_service.generate_answer(query)
    assert response == "dummy response"
