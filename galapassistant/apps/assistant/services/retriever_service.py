from galapassistant.apps.assistant.services.embedding_service import EmbeddingService


class RetrieverService:
    def __init__(self, query: str):
        service = EmbeddingService()
        knowledge_vector_database = service.build_vector_database()
        vector_store_retriever = knowledge_vector_database.as_retriever()

        docs = vector_store_retriever.similarity_search(
            query,
            k=2,
        )

        return "\nRetrieved documents:\n" + "".join(
            [f"= Document {str(i)} =\n" + doc.page_content for i, doc in enumerate(docs)]
        )
