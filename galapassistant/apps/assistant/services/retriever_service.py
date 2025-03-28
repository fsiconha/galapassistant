from galapassistant.apps.assistant.services.embedding_service import EmbeddingService


class RetrieverService:
    """
    Service class for retrieving documents from a vector store based on a query.
    """
    def __init__(self):
        service = EmbeddingService()
        knowledge_vector_database = service.build_vector_database()
        self.vector_store_retriever = knowledge_vector_database.as_retriever()

    def retrieve(self, query: str, k: int = 2) -> str:
        """
        Retrieves the top matching documents for the provided query.
        
        Args:
            query (str): The input query text used to perform similarity search.
            k (int, optional): The number of top documents to retrieve. Defaults to 2.
        
        Returns:
            str: A formatted string containing the retrieved documents.
        """
        docs = self.vector_store_retriever.similarity_search(query, k=k)
        result = "\nRetrieved documents:\n" + "".join(
            [f"= Document {i} =\n{doc.page_content}\n" for i, doc in enumerate(docs)]
        )
        return result
