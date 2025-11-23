import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain_core.documents import Document
from rag_pipeline.embeddings import get_embedding_model
from config import CHROMA_PERSIST_DIR, RISK_COLLECTION_NAME

def load_risk_csv(csv_path: str = "data/risk_db.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)

def build_risk_vectorstore(csv_path: str = "data/risk_db.csv") -> Chroma:
    df = load_risk_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        metadata = {
            "id": int(row["id"]),
            "clause_type": row["clause_type"],
            "risk_level": float(row["risk_level"]),
            "notes": row.get("notes", ""),
        }
        docs.append(Document(page_content=row["text"], metadata=metadata))

    embeddings = get_embedding_model()

    client_settings = Settings(
    anonymized_telemetry=False,
    )

    vs = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=RISK_COLLECTION_NAME,
        client_settings=client_settings,
    )
    vs.persist()
    return vs

def get_risk_vectorstore() -> Chroma:
    embeddings = get_embedding_model()
    vs = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=RISK_COLLECTION_NAME,
    )
    # If the persisted collection is empty (no embeddings), rebuild it in-process
    try:
        col = getattr(vs, '_collection', None)
        if col is not None and hasattr(col, 'count') and col.count() == 0:
            # Rebuild from CSV and return the fresh vectorstore instance
            return build_risk_vectorstore()
    except Exception:
        # If anything goes wrong checking persistence, fall back to returning vs
        pass

    return vs