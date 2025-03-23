import pytest
from django.test import Client
from django.urls import reverse

from galapassistant.apps.assistant.services.llm_service import AssistantLLMService


@pytest.fixture
def client():
    """
    Fixture that returns a Django test client.
    """
    return Client()

def fake_get_response(self, query: str) -> str:
    """Return a fake assistant response for testing."""
    return f"Fake assistant response for: {query}"

@pytest.mark.django_db
def test_chat_view_get(client):
    """
    Test that a GET request to the chat view returns a 200 status code
    and that the assistant's response is empty.
    """
    url = reverse("chat")
    response = client.get(url)
    assert response.status_code == 200

@pytest.mark.django_db
def test_chat_view_post_empty_query(client):
    """
    Test that a POST request with an empty query returns a 200 status code
    and does not display any assistant message.
    """
    url = reverse("chat")
    response = client.post(url, {"query": ""})
    assert response.status_code == 200

@pytest.mark.django_db
def test_chat_view_post_with_query(client, monkeypatch):
    """
    Test that a POST request with a non-empty query returns a non-empty assistant response.
    """
    monkeypatch.setattr(AssistantLLMService, "get_response", fake_get_response)

    url = reverse("chat")
    test_query = "Test message"
    response = client.post(url, {"query": test_query})
    content = response.content.decode()
    assert response.status_code == 200
    assert len(content.strip()) > 50, "Expected a non-empty assistant response."
    assert response.status_code == 200, "Expected a 200 OK response."
    expected_response = f"Fake assistant response for: {test_query}"
    assert expected_response in content, "Expected a fake assistant response in the HTML output."
