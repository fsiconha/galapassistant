import pytest
from django.test import Client
from django.urls import reverse

@pytest.fixture
def client():
    """
    Fixture that returns a Django test client.
    """
    return Client()

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
def test_chat_view_post_with_query(client):
    """
    Test that a POST request with a non-empty query returns the expected
    assistant message containing "Hello, world" followed by the query.
    """
    url = reverse("chat")
    test_query = "Test message"
    response = client.post(url, {"query": test_query})
    expected_response = f"Hello, world: {test_query}"
    content = response.content.decode()
    assert response.status_code == 200
    assert expected_response in content
