import pytest
from django.test import RequestFactory
from galapassistant.apps.chat.views import rag_chat_view


class MockAssistant:
    def get_response(self, query: str) -> str:
        return f"Mock response for: {query}"

@pytest.fixture
def request_factory():
    return RequestFactory()

@pytest.fixture(autouse=True)
def patch_assistant(monkeypatch):
    """
    Automatically patch the global assistant variable in the views module with a mock assistant.
    """
    mock_assistant = MockAssistant()
    monkeypatch.setattr("galapassistant.apps.chat.views", mock_assistant)

def test_rag_chat_view_get(request_factory):
    """
    Test that a GET request to the rag_chat_view returns the chat form with an empty response.
    """
    request = request_factory.get("/")
    response = rag_chat_view(request)
    assert response.status_code == 200
    if hasattr(response, "context_data"):
        assert response.context_data.get("response") == ""
    else:
        content = response.content.decode("utf-8")
        assert "Mock response" not in content

def test_rag_chat_view_post_empty_query(request_factory):
    """
    Test that a POST request with an empty query returns an empty response.
    """
    request = request_factory.post("/", data={"query": "   "})
    response = rag_chat_view(request)
    if hasattr(response, "context_data"):
        assert response.context_data.get("response") == ""
    else:
        content = response.content.decode("utf-8")
        assert "Mock response" not in content

def test_rag_chat_view_post_with_query(request_factory):
    """
    Test that a POST request with a valid query returns the expected mock response.
    """
    query = "Who was Darwin?"
    request = request_factory.post("/", data={"query": query})
    response = rag_chat_view(request)
    assert response
