import os
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from galapassistant.apps.assistant.services.generation import GenerationService


assistant = GenerationService()

def rag_chat_view(request: HttpRequest) -> HttpResponse:
    """
    View to handle chat requests using the RAG assistant.
    
    On GET, it displays the chat form.
    On POST, it processes the user query using the RAG assistant and returns the answer.
    """
    response_text = ""
    if request.method == "POST":
        query = request.POST.get("query", "").strip()
        if query:
            response_text = assistant.generate_answer(query)
    return render(request, "chat.html", {"response": response_text})
