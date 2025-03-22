import json
import pytest
from channels.testing import WebsocketCommunicator
from galapassistant.asgi import application


@pytest.mark.asyncio
async def test_chat_consumer():
    """
    Test the WebSocket consumer for chat.
    """
    communicator = WebsocketCommunicator(application, "/ws/chat/")
    connected, _ = await communicator.connect()
    assert connected

    # Send a test query.
    await communicator.send_json_to({"query": "What is the name of the expedition ship?"})
    response = await communicator.receive_json_from()
    assert "response" in response

    await communicator.disconnect()
