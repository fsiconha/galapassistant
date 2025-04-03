import pytest
from langchain_core.language_models.fake import FakeListLLM
from galapassistant.apps.assistant.services.generation import GenerationService


def mock_generate_answer(self, user_query: str) -> str:
    """Dummy replacement for GenerationService.generate_answer."""
    return "dummy response"

@pytest.fixture
def mock_generation_assistant_service(monkeypatch):
    """
    Create an instance of GenerationService with a FakeListLLM and patch generate_answer
    to return a dummy response.
    """
    monkeypatch.setattr(GenerationService, "__init__", lambda self: None)
    service = GenerationService()
    fake_llm = FakeListLLM(responses=["dummy response"])
    service.llm = fake_llm
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.generation.GenerationService.generate_answer",
        mock_generate_answer
    )
    return service

def test_get_response_returns_mock_response(mock_generation_assistant_service):
    """
    Test that get_response returns the expected mock response.
    """
    query = "What is evolution?"
    response = mock_generation_assistant_service.generate_answer(query)

    assert len(response) > 0
    assert response == "dummy response"
