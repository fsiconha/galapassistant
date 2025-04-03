import os
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from galapassistant.apps.assistant.services.llm_service import AssistantLLMService


assistant = AssistantLLMService()

def rag_chat_view(request: HttpRequest) -> HttpResponse:
    response_text = ""
    if request.method == "POST":
        query = request.POST.get("query", "").strip()
        if query:
            try:
                response_text = assistant.get_response(query)
            except Exception as e:
                logging.exception("Error in chat view:")
                response_text = "Something went wrong. Please try again."
    return render(request, "chat.html", {"response": response_text})
