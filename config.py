# config.py

# Gemma 2B Instruct (causal LM)
# Make sure you've accepted the license + logged in via huggingface-cli.
LLM_MODEL_NAME = "google/gemma-2b-it"
LLM_TASK = "text-generation"  # for causal LMs

# Embeddings (unchanged)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHROMA_PERSIST_DIR = "chroma_store"
RISK_COLLECTION_NAME = "risk_clauses"
CONTRACT_COLLECTION_NAME = "contract_chunks"

CLAUSE_TYPES = [
    "indemnity",
    "termination",
    "confidentiality",
    "limitation_of_liability",
    "governing_law",
    "payment_terms",
]

ROLE_PROFILES = {
    "Legal (very conservative)": {"low": 2.5, "medium": 5.0},
    "Business (balanced)": {"low": 4.0, "medium": 7.0},
    "Sales (risk tolerant)": {"low": 5.0, "medium": 8.5},
}
