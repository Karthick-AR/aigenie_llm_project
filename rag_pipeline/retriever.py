# rag_pipeline/retriever.py

from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from chromadb.config import Settings
from rag_pipeline.embeddings import get_embedding_model
from config import CHROMA_PERSIST_DIR, CONTRACT_COLLECTION_NAME


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """
    Very simple text splitter that chunks a long string into overlapping pieces.
    No dependency on langchain.text_splitter.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap  # step forward with overlap

    return chunks


def build_contract_vectorstore(contract_docs: List[Document]) -> Chroma:
    """
    Build a Chroma vector store from the uploaded contract documents.
    Each original Document is split into smaller overlapping chunks.
    """
    chunked_docs: List[Document] = []

    for idx, doc in enumerate(contract_docs):
        base_meta = dict(doc.metadata or {})
        base_meta["source_doc_index"] = idx

        for i, chunk in enumerate(_chunk_text(doc.page_content)):
            meta = dict(base_meta)
            meta["chunk_index"] = i
            chunked_docs.append(Document(page_content=chunk, metadata=meta))

    client_settings = Settings(
    anonymized_telemetry=False,
    )
    
    vs = Chroma.from_documents(
        documents=chunked_docs,
        embedding=get_embedding_model(),
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CONTRACT_COLLECTION_NAME,
        client_settings=client_settings,
    )
    vs.persist()
    return vs


def get_contract_vectorstore() -> Chroma:
    """
    Load the existing Chroma store for contract chunks.
    """
    client_settings = Settings(
    anonymized_telemetry=False,
    )

    return Chroma(
        embedding_function=get_embedding_model(),
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CONTRACT_COLLECTION_NAME,
        client_settings=client_settings,
    )
