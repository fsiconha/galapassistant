from langchain.chains import RetrievalQA

from galapassistant.apps.assistant.services.embedding_service import EmbeddingService


class AgentUtils:
    def __init__(self):
        from galapassistant.apps.assistant.services.llm_service import AssistantLLMService
        llm_service_instance = AssistantLLMService()
        self.llm = llm_service_instance.build_llm()

    def get_retriever(self):
        embedding_service = EmbeddingService()
        vector_store = embedding_service.build_vector_database()
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )

        return retriever

    def get_rag_chain(self):
        retriever = self.get_retriever()
        rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever
        )

        return rag_chain
