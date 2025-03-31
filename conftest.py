import os
import django
import pytest
from typing import List

from langchain_core.language_models.fake import FakeListLLM
from langchain.docstore.document import Document as LangchainDocument

from galapassistant.apps.assistant.services.llm_service import AssistantLLMService


os.environ.setdefault(
    'DJANGO_SETTINGS_MODULE', 'galapassistant.settings.settings'
)
django.setup()

def load_knowledge_base(file_path: str) -> List[LangchainDocument]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [LangchainDocument(page_content=content)]

def sample_paragraphs_from_document(
    document: LangchainDocument,
    num_paragraphs: int = 8,
    chars_per_paragraph: int = 1000
) -> str:
    content = document.page_content
    total_required = num_paragraphs * chars_per_paragraph
    if len(content) < total_required:
        total_required = len(content)
    sample_text = content[:total_required]
    paragraphs = [
        sample_text[i * chars_per_paragraph: (i + 1) * chars_per_paragraph]
        for i in range(num_paragraphs)
    ]
    return "\n\n".join(paragraphs)

@pytest.fixture
def mock_knowledge_base() -> List[LangchainDocument]:
    file_path = os.path.join(
        "galapassistant",
        "apps",
        "assistant",
        ".knowledge_base",
        "the Origin of Species.txt"
    )
    documents = load_knowledge_base(file_path)
    if documents:
        sample_text = sample_paragraphs_from_document(
            documents[0],
            num_paragraphs=3,
            chars_per_paragraph=500
        )
        return [LangchainDocument(page_content=sample_text)]
    return documents

class MockChain:
    """A mock chain to simulate the rag_chain's invoke method."""
    def __init__(self, response):
        self.response = response

    def __or__(self, other):
        return self

    def invoke(self, query):
        return self.response

@pytest.fixture
def mock_assistant_service(monkeypatch):
    """
    Create an instance of AssistantLLMService with a FakeListLLM,
    and patch its chain components so that generate_answer returns
    the fake LLM response.
    """
    service = AssistantLLMService.__new__(AssistantLLMService)
    fake_llm = FakeListLLM(["dummy response"])
    service.llm = fake_llm
    service.prompt = IdentityChain(fake_llm)
    monkeypatch.setattr(
        "galapassistant.apps.assistant.services.llm_service.StrOutputParser",
        lambda: IdentityChain(fake_llm)
    )
    from galapassistant.apps.assistant.services.retriever_service import RetrieverService
    monkeypatch.setattr(RetrieverService, "retrieve", lambda self, query: "dummy context")
    service.generate_answer = service.generate_answer
    return service
