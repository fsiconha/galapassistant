import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "thenlper/gte-base"

class IndexingService:
    """
    Service for loading a knowledge base, embedding the documents and building a vector database for document retrieval.
    """
    def load_knowledge_base(self, file_path: str) -> List[LangchainDocument]:
        """
        Loads the knowledge base from a text file. Opens the file
        specified by 'file_path', reads its entire content, and returns
        a list containing a single LangChain Document with the file content.

        Args:
            file_path (str): The path to the text file containing the knowledge base.

        Returns:
            List[LangchainDocument]: A list with one Document containing the file content.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return [LangchainDocument(page_content=content)]
        except Exception as e:
            print(f"Error loading knowledge base from {file_path}: {e}")
            return []

    def build_vector_database(
        self,
        knowledge_base: Optional[List[LangchainDocument]] = None,
    ):
        """
        Builds a FAISS vector database from the provided knowledge base.

        If no knowledge base is provided, it loads the default knowledge base from a file.
        The text is split into chunks using a SemanticChunker. Then, each document chunk
        is embedded using a HuggingFace model, and the resulting embeddings are stored
        in a FAISS index for similarity search using cosine distance.

        Args:
            knowledge_base (Optional[List[LangchainDocument]], optional): A list of documents
                to build the vector database. If not provided, the default knowledge base file is used.

        Returns:
            FAISS: An instance of the FAISS vector store built from the document chunks.
        """
        if knowledge_base is None:
            file_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                ".knowledge_base",
                "the Origin of Species.txt"
            )
            knowledge_base = self.load_knowledge_base(file_path)

        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        text_splitter = (
            SemanticChunker(
                embeddings=embedding_model,
                breakpoint_threshold_type="standard_deviation",
            )
        )

        doc_chunks = []
        for doc in knowledge_base:
            doc_chunks += text_splitter.split_documents([doc])

        unique_texts = {}
        unique_doc_chunks = []
        for doc in doc_chunks:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                unique_doc_chunks.append(doc)

        knowledge_vector_database = FAISS.from_documents(
            unique_doc_chunks,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )

        return knowledge_vector_database
