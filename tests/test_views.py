import pytest
from django.test import RequestFactory
from galapassistant.apps.chat.views import rag_chat_view


class DummyAssistant:
    """
    Dummy assistant with a generate_answer method that returns a dummy response.
    """
    def generate_answer(self, query: str) -> str:
        return "dummy response"

@pytest.fixture(autouse=True)
def patch_assistant(monkeypatch):
    """
    Automatically patch the assistant in the chat view to use the dummy assistant.
    """
    monkeypatch.setattr("galapassistant.apps.chat.views.assistant", DummyAssistant())

@pytest.fixture
def request_factory():
    return RequestFactory()

def test_rag_chat_view_get(request_factory):
    """
    Test that a GET request to rag_chat_view returns the chat form with an empty response.
    """
    request = request_factory.get("/")
    response = rag_chat_view(request)
    content = response.content.decode("utf-8")
    assert response.status_code == 200
    assert "Mock response" not in content

def test_rag_chat_view_post_empty_query(request_factory):
    """
    Test that a POST request with an empty query returns an empty response.
    """
    request = request_factory.post("/", data={"query": "   "})
    response = rag_chat_view(request)
    content = response.content.decode("utf-8")
    assert response.status_code == 200
    assert "dummy response" not in content

def test_rag_chat_view_post_with_query(request_factory, patch_assistant):
    """
    Test that a POST request with an empty query returns an empty response.
    """
    request = request_factory.post("/", data={"query": "Who was Darwin?"})
    response = rag_chat_view(request)
    content = response.content.decode("utf-8")
    assert response.status_code == 200
    assert isinstance(content, str)
    assert len(content) > 0
    assert "dummy response" in content
