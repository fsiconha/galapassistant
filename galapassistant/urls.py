from django.urls import path
from galapassistant.apps.chat.views import chat_view

urlpatterns = [
    path('', chat_view, name='chat'),
    # path('chat/', chat_view, name='chat'),
]
