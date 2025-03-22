import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import galapassistant.apps.chat.routing as chat_routing

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "galapassistant.settings.settings")

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            chat_routing.websocket_urlpatterns
        )
    ),
})
