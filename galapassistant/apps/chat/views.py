from django.shortcuts import render
from django.http import HttpRequest, HttpResponse


def chat_view(request: HttpRequest) -> HttpResponse:
    """
    Render the chat page. On POST, if the user provides a non-empty query,
    return a response that includes 'Hello, world' followed by the query.
    The assistant's message is only visible after the user submits a query.

    Args:
        request (HttpRequest): The incoming HTTP request.

    Returns:
        HttpResponse: The rendered chat page with the assistant's response if available.
    """
    response_text = ""
    if request.method == "POST":
        query = request.POST.get("query", "").strip()
        if query:
            response_text = f"Hello, world: {query}"
    return render(request, "chat.html", {"response": response_text})
