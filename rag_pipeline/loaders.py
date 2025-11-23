from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_core.documents import Document


def load_contract(path_str: str) -> List[Document]:
    """
    Load a contract file into LangChain Documents using community loaders.
    This version relies on older langchain-core that still had BaseBlobParser.
    """
    path = Path(path_str)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    docs = loader.load()
    return docs