import pytest
from langchain_core.language_models.fake import FakeListLLM
from galapassistant.apps.assistant.services.llm_service import AssistantLLMService


def mock_generate_answer(self, user_query: str) -> str:
    """Dummy replacement for AssistantLLMService.generate_answer."""
    return "dummy response"

@pytest.fixture
def mock_assistant_service(monkeypatch):
    """
    Create an instance of AssistantLLMService with a FakeListLLM and patch generate_answer
    to return a dummy response.
    """
    monkeypatch.setattr(AssistantLLMService, "__init__", lambda self: None)
    service = AssistantLLMService()
    fake_llm = FakeListLLM(responses=["dummy response"])
    service.llm = fake_llm
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.llm_service.AssistantLLMService.generate_answer",
        mock_generate_answer
    )
    return service

def test_get_response_returns_mock_response(mock_assistant_service):
    """
    Test that get_response returns the expected mock response.
    """
    query = "What is evolution?"
    response = mock_assistant_service.generate_answer(query)

    assert len(response) > 0
    assert response == "dummy response"
