import pytest
from galapassistant.apps.assistant.services.llm_service import AssistantLLMService

class IdentityChain:
    """
    A chain that passes through calls to a fake LLM.
    """
    def __init__(self, llm):
        self.llm = llm

    def __or__(self, other):
        return self

    def invoke(self, chain_input):
        return self.llm(chain_input)

# def test_generate_answer_returns_response(mock_assistant_service):
#     """
#     Test that generate_answer returns the expected fake response.
#     """
#     query = "What is evolution?"
#     response = mock_assistant_service.generate_answer(query)
#     assert len(response) > 0
