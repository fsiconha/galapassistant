import pytest
from galapassistant.apps.assistant.services.llm_service import AssistantLLMService


def fake_llm(query: str) -> str:
    """
    Fake LLM function to simulate a response.

    Args:
        query (str): The input query.

    Returns:
        str: A mock response based on the query.
    """
    return "Mock response for: " + query


@pytest.fixture
def llm_service(monkeypatch: pytest.MonkeyPatch) -> AssistantLLMService:
    """
    Fixture that returns an instance of AssistantLLMService with its LLM call patched.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.

    Returns:
        AssistantLLMService: The service instance with patched LLM.
    """
    service = AssistantLLMService()
    # Monkeypatch the instance's llm attribute so that calling it returns a controlled output.
    monkeypatch.setattr(service, "llm", fake_llm)
    return service


def test_get_response(llm_service: AssistantLLMService):
    """
    Test that get_response returns the expected response from the LLM service.

    Args:
        llm_service (AssistantLLMService): The patched LLM service fixture.
    """
    query = "Hello"
    expected = "Mock response for: Hello"
    response = llm_service.get_response(query)
    assert response == expected
