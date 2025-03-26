from smolagents import Tool
from langchain_core.vectorstores import VectorStore
from galapassistant.apps.assistant.services.embedding_service import EmbeddingService


class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        embedding_service = EmbeddingService()
        self.vector_store = embedding_service.build_vector_database()

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vector_store.similarity_search(
            query,
            k=2,
        )

        return "\nRetrieved documents:\n" + "".join(
            [f"===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )
