from langchain.agents import Tool

from galapassistant.apps.assistant.services.embedding_service import EmbeddingService
from galapassistant.apps.assistant.services.utils import AgentUtils


TOP_K_RETRIEVED_DOCUMENTS = 3

class RetrieverTool(Tool):
    def __init__(self):
        utils = AgentUtils()
        self.rag_chain = utils.get_rag_chain()

    def get_retriever_tool(self):
        retriever_tool = Tool(
            name="OriginOfSpeciesRetriever",
            func=self.rag_chain.run,
            description=(
                "Use this function to retrieve documents about the Origin of Species. "
                "The book Origin of Species is your knowledge base. "
                "INPUT are questions about Darwinism topics "
                "OUTPUT concise and relevant answer to the question, based on the knowledge base."
            )
        )

        return retriever_tool
