from langchain_core.tools import BaseTool
from galapassistant.apps.assistant.services.embedding_service import EmbeddingService

TOP_K_RETRIEVED_DOCUMENTS = 3

class RetrieverTool(BaseTool):
    name: str = "retriever"
    description: str = (
        "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    )
    inputs: dict = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        },
        "k": {
            "type": "integer",
            "description": "The number of top documents to retrieve.",
        },
    }
    output_type: str = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        embedding_service = EmbeddingService()
        self.knowledge_vector_database = embedding_service.build_vector_database()

    def _run(self, tool_input: dict) -> str:
        query = tool_input.get("query")
        k = tool_input.get("k", TOP_K_RETRIEVED_DOCUMENTS)
        if not isinstance(query, str):
            raise ValueError("Your search query must be a string")
        docs = self.knowledge_vector_database.similarity_search(query, k)
        return "\nRetrieved documents:\n" + "".join(
            [
                f"_-_ Document {i} _-_\nContent {i}:\n{doc.page_content}\n"
                for i, doc in enumerate(docs)
            ]
        )

    async def _arun(self, tool_input: dict) -> str:
        return self._run(tool_input)
