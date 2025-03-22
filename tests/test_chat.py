import json
import pytest
from django.urls import reverse


@pytest.mark.django_db
def test_chat_query(client):
    """
    Test the chat_query API endpoint.
    """
    url = reverse("chat_query")
    data = {"query": "What is the name of the expedition ship?"}
    response = client.post(url, data=json.dumps(data), content_type="application/json")
    assert response.status_code == 200
    json_response = response.json()
    assert "response" in json_response
