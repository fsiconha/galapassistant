import os
import django
import pytest
from typing import List

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
            num_paragraphs=8,
            chars_per_paragraph=1000
        )
        return [LangchainDocument(page_content=sample_text)]
    return documents
