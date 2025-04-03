from langchain_core.tools import tool

from galapassistant.apps.assistant.services.embedding_service import EmbeddingService


TOP_K_RETRIEVED_DOCUMENTS = 3

class RetrieverTool:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        embedding_service = EmbeddingService()
        self.knowledge_vector_database = (
            embedding_service.build_vector_database()
        )

    @tool(response_format="content_and_artifact")
    def retrieve(self, query: str):
        """Retrieve information related to a query."""
        retrieved_docs = self.knowledge_vector_database.similarity_search(
            query,
            k=TOP_K_RETRIEVED_DOCUMENTS
        )
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
