import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


EMBEDDING_MODEL_NAME = "thenlper/gte-small"
CHUNK_SIKE = 512
MARKDOWN_SEPARATORS = [
    "\n\n",
    "\n",
    ".",
    " ",
]

class EmbeddingService:
    def load_knowledge_base(file_path: str) -> List[LangchainDocument]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return [LangchainDocument(page_content=content)]   

    def split_data(
        self,
        chunk_size: int = CHUNK_SIKE,
        knowledge_base: Optional[List[LangchainDocument]] = None,
        tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
    ) -> List[LangchainDocument]:
        if knowledge_base is None:
            file_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                ".knowledge_base",
                "the Origin of Species.txt"
            )
            knowledge_base = self.load_knowledge_base(file_path)

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )

        doc_chunks = []
        for doc in knowledge_base:
            doc_chunks += text_splitter.split_documents([doc])

        unique_texts = {}
        doc_chunks_unique = []
        for doc in doc_chunks:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                doc_chunks_unique.append(doc)

        return doc_chunks_unique

    def build_vector_database(self, doc_chunks):   ### PRECISA CACHEAR DE ALGUMA FORMA OU OTIMIZAR. Como usar o retriever sem buildar o vector store toda hora?
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        knowledge_vector_database = FAISS.from_documents(
            doc_chunks, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )

        return knowledge_vector_database
