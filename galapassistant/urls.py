from django.urls import path
from galapassistant.apps.chat.views import rag_chat_view

urlpatterns = [
    path('', rag_chat_view, name='chat'),
    # path('chat/', chat_view, name='chat'),
]
