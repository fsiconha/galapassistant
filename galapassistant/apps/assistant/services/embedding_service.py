import os
# Disable parallelism for tokenizers to avoid fork-related warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import multiprocessing
# Change the start method to spawn (especially useful on macOS)
multiprocessing.set_start_method("spawn", force=True)

from typing import List, Optional

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

import torch
torch.set_num_threads(1)


EMBEDDING_MODEL_NAME = "thenlper/gte-base"
CHUNK_SIKE = 500
MARKDOWN_SEPARATORS = [
    "\n\n",
    "\n",
    # ".",
    # " ",
]

class EmbeddingService:
    """
    Service for loading a knowledge base and building a vector database for document retrieval.
    """
    def load_knowledge_base(self, file_path: str) -> List[LangchainDocument]:
        """
        Loads the knowledge base from a text file.
        Opens the file specified by 'file_path', reads its entire content, and returns
        a list containing a single LangChain Document with the file content.

        Args:
            file_path (str): The path to the text file containing the knowledge base.

        Returns:
            List[LangchainDocument]: A list with one Document containing the file content.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return [LangchainDocument(page_content=content)]

    def build_vector_database(
        self,
        chunk_size: Optional[int] = CHUNK_SIKE,
        knowledge_base: Optional[List[LangchainDocument]] = None,
        tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
    ):
        """
        Builds a FAISS vector database from the provided knowledge base.

        If no knowledge base is provided, it loads the default knowledge base from a file.
        The text is split into chunks using a RecursiveCharacterTextSplitter with a specified
        chunk size and overlap. Then, each document chunk is embedded using a HuggingFace model,
        and the resulting embeddings are stored in a FAISS index for fast similarity search using
        cosine distance.

        Args:
            chunk_size (Optional[int], optional): The maximum number of tokens in each chunk.
                Defaults to CHUNK_SIKE.
            knowledge_base (Optional[List[LangchainDocument]], optional): A list of documents
                to build the vector database. If not provided, the default knowledge base file is used.
            tokenizer_name (Optional[str], optional): The name of the HuggingFace model used for
                tokenization and embeddings. Defaults to EMBEDDING_MODEL_NAME.

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

        knowledge_vector_database = FAISS.from_documents(
            doc_chunks,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )

        return knowledge_vector_database
