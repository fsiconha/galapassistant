from langchain_core.tools import tool

from galapassistant.apps.assistant.services.embedding_service import EmbeddingService


TOP_K_RETRIEVED_DOCUMENTS = 3

class RetrieverService:
    """
    Service class for retrieving documents from a vector store based on a query.
    """
    def __init__(self):
        vector_store_service = EmbeddingService()
        self.retriever = vector_store_service.build_vector_database()

    @tool
    def retrieve(self, query: str, k: int = TOP_K_RETRIEVED_DOCUMENTS) -> str:
        """
        Retrieves the top matching documents for the provided query.
        
        Args:
            query (str): The input query text used to perform similarity search.
            k (int, optional): The number of top documents to retrieve. Defaults to 3.
        
        Returns:
            str: A formatted string containing the retrieved documents.
        """
        retrieved_docs = self.retriever.similarity_search(query, k=k)
        serialized = "\n\n".join(
            (f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized
