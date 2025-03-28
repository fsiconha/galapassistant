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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

import torch
torch.set_num_threads(1)


EMBEDDING_MODEL_NAME = "thenlper/gte-small" #"sentence-transformers/all-mpnet-base-v2"
CHUNK_SIKE = 500
MARKDOWN_SEPARATORS = [
    "\n\n",
    "\n",
    # ".",
    # " ",
]

class EmbeddingService:
    def load_knowledge_base(self, file_path: str) -> List[LangchainDocument]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return [LangchainDocument(page_content=content)]

    def build_vector_database(
        self,
        chunk_size: Optional[int] = CHUNK_SIKE,
        knowledge_base: Optional[List[LangchainDocument]] = None,
        tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
    ):
        if knowledge_base is None:
            file_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                ".knowledge_base",
                "the Origin of Species.txt"
            )
            knowledge_base = self.load_knowledge_base(file_path)

        text_splitter = (
            RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                AutoTokenizer.from_pretrained(tokenizer_name),
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size / 10),
                add_start_index=True,
                strip_whitespace=True,
                separators=MARKDOWN_SEPARATORS,
            )
        )

        doc_chunks = []
        for doc in knowledge_base:
            doc_chunks += text_splitter.split_documents([doc])

        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query("hello world")))
        # vector_store = FAISS(
        #     embedding_function=embedding_model,
        #     index=index,
        #     docstore=InMemoryDocstore(),
        #     index_to_docstore_id={}
        # )

        knowledge_vector_database = FAISS.from_documents(
            doc_chunks,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )

        return knowledge_vector_database
