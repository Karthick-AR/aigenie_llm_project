from typing import List, Dict
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

from config import LLM_MODEL_NAME, LLM_TASK, CLAUSE_TYPES

_llm = None


def _get_llm():
    """Singleton Gemma 2B text-generation LLM for clause extraction."""
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    global _llm
    if _llm is not None:
        return _llm

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME,
    use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME,
    use_auth_token=token)

    hf_pipe = pipeline(
        LLM_TASK,             # "text-generation"
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
    )
    _llm = HuggingFacePipeline(pipeline=hf_pipe)
    return _llm


def extract_clauses(contract_text: str) -> List[Dict]:
    """
    Use Gemma to extract clauses as JSON.
    """
    llm = _get_llm()

    prompt = f"""
You are a legal assistant. Given the contract text, extract key clauses.

Return STRICT JSON ONLY with this structure:
{{
  "clauses": [
    {{
      "clause_type": "indemnity|termination|confidentiality|limitation_of_liability|governing_law|payment_terms|other",
      "title": "Short heading or summary",
      "text": "Full clause text",
      "start_hint": "First few words or position hint"
    }}
  ]
}}

Do not add explanation or extra commentary.

Contract text:
```text
{contract_text}
```"""

    raw = llm.invoke(prompt)

    if isinstance(raw, str):
        raw_text = raw
    elif hasattr(raw, "content"):
        raw_text = raw.content
    else:
        raw_text = str(raw)

    # Gemma text-gen often returns something like "<s> ... </s>": strip noise
    raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
        clauses = data.get("clauses", [])
    except Exception:
        # Fallback: try to locate JSON substring
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        clauses = []
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw_text[start : end + 1])
                clauses = data.get("clauses", [])
            except Exception:
                clauses = []

    for c in clauses:
        t = (c.get("clause_type") or "other").lower().strip()
        if t not in CLAUSE_TYPES and t != "other":
            t = "other"
        c["clause_type"] = t

    return clauses
