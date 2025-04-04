from smolagents import Tool

from galapassistant.apps.assistant.services.indexing import IndexingService


TOP_K_RETRIEVED_DOCUMENTS = 3

class RetrieverTool(Tool):
    """
    Tool for retrieving documents from a vector store based on a query.
    """
    name = "retriever"
    description = ("Uses semantic search to retrieve the parts of transformers documentation "
                   "that could be most relevant to answer your query.")
    inputs = {
        "query": {
            "type": "string",
            "description": ("The query to perform. This should be semantically close "
                            "to your target documents. Use the affirmative form "
                            "rather than a question."),
        },
        "k": {
            "type": "integer",
            "description": "The number of top documents to retrieve.",
        },
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        indexing = IndexingService()
        self.knowledge_vector_database = indexing.build_vector_database()

    def forward(self, query: str, k: int) -> str:
        """
        Retrieves the top matching documents for the provided query.

        Args:
            query (str): The input query text used to perform similarity search.
            k (int, optional): The number of top documents to retrieve. Defaults to 3.

        Returns:
            str: A formatted string containing the retrieved documents, or an error message.
        """
        try:
            assert isinstance(query, str), "Your search query must be a string"
            if self.knowledge_vector_database is None:
                return "Vector database is not available."

            docs = self.knowledge_vector_database.similarity_search(
                query,
                k=TOP_K_RETRIEVED_DOCUMENTS,
            )
            return "\nRetrieved documents:\n" + "".join(
                [f"===== Document {str(i)} =====\n" + doc.page_content
                 for i, doc in enumerate(docs)]
            )
        except Exception as e:
            error_msg = f"Error during document retrieval: {e}"
            print(error_msg)
            return error_msg
