from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from galapassistant.apps.assistant.services.llm_service import AssistantLLMService


def chat_view(request: HttpRequest) -> HttpResponse:
    """
    Render the chat page. On POST, use the AssistantLLMService to generate a response
    from the LLM based on the user's query.

    Args:
        request (HttpRequest): The incoming HTTP request.

    Returns:
        HttpResponse: The rendered chat page with the assistant's response if available.
    """
    response_text = ""
    if request.method == "POST":
        query = request.POST.get("query", "").strip()
        if query:
            assistant = AssistantLLMService()
            response_text = assistant.get_response(query)
    return render(request, "chat.html", {"response": response_text})
