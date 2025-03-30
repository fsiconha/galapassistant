from smolagents import Tool

from galapassistant.apps.assistant.services.embedding_service import EmbeddingService


class RetrieverTool(Tool):
    """
    Tool for retrieving documents from a vector store based on a query.
    """
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        vector_store = EmbeddingService()
        self.retriever = vector_store.from_documents(
            docs, k=10
        )

    def forward(self, query: str) -> str:
        """
        Retrieves the top matching documents for the provided query.
        
        Args:
            query (str): The input query text used to perform similarity search.
            k (int, optional): The number of top documents to retrieve. Defaults to 2.
        
        Returns:
            str: A formatted string containing the retrieved documents.
        """
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )
