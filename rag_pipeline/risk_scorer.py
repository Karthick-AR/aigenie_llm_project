from typing import List, Dict
import numpy as np
from rag_pipeline.risk_db import get_risk_vectorstore
from config import ROLE_PROFILES

def score_clause(clause: Dict, top_k: int = 5) -> Dict:
    """
    Score a clause against the risk DB, deduplicating repeated matches
    so you don't see the same pattern 3 times in the UI.
    """
    vs = get_risk_vectorstore()
    text = clause["text"]

    # Some Chroma/Vectorstore wrappers expose similarity_search_with_relevance_scores
    # (returning (doc, score) tuples). If not available in this environment, fall
    # back to plain similarity_search() and synthesize a relevance score so we
    # still compute non-zero risk scores.
    try:
        results = vs.similarity_search_with_relevance_scores(text, k=top_k)
    except Exception:
        results = None

    # If the with_relevance call returned nothing, or returned plain Documents
    # (some wrappers implement a similarly-named method but return docs), fall
    # back to similarity_search() and synthesize similarity scores.
    if not results or (len(results) > 0 and not isinstance(results[0], (tuple, list))):
        docs = vs.similarity_search(text, k=top_k)
        results = [(d, 1.0) for d in docs]

    if not results:
        clause["risk_score"] = 0.0
        clause["matches"] = []
        return clause

    matches = []
    scores = []

    seen_ids = set()      # use DB id to dedupe
    seen_texts = set()    # fallback in case id missing

    for doc, sim in results:
        doc_id = doc.metadata.get("id")
        text_key = doc.page_content.strip()

        # skip duplicates using id first, then text
        if doc_id is not None and doc_id in seen_ids:
            continue
        if text_key in seen_texts:
            continue

        if doc_id is not None:
            seen_ids.add(doc_id)
        seen_texts.add(text_key)

        risk_level = float(doc.metadata.get("risk_level", 5.0))
        # sim is in 0..1 (approx). Combine similarity and risk_level to produce
        # a weighted score in 0..10. If sim is synthesized as 1.0 (fallback), this
        # will rely mainly on risk_level.
        weighted = float(sim * (risk_level))
        scores.append(weighted)

        matches.append({
            "text": doc.page_content,
            "similarity": float(sim),
            "risk_level": risk_level,
            "weighted_score": weighted,
            "clause_type": doc.metadata.get("clause_type"),
            "notes": doc.metadata.get("notes", ""),
            "id": doc_id,
        })

    if not scores:
        clause["risk_score"] = 0.0
        clause["matches"] = []
    else:
        clause["risk_score"] = float(np.mean(scores))
        clause["matches"] = matches

    return clause

def score_contract(clauses: List[Dict]) -> Dict:
    scored = [score_clause(c) for c in clauses]
    overall = float(np.mean([c["risk_score"] for c in scored])) if scored else 0.0
    return {"overall_risk_score": overall, "clauses": scored}

def classify_risk(score: float, profile_name: str) -> str:
    thresholds = ROLE_PROFILES[profile_name]
    if score <= thresholds["low"]:
        return "Low"
    elif score <= thresholds["medium"]:
        return "Medium"
    else:
        return "High"
