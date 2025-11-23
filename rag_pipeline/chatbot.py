# rag_pipeline/chatbot.py

from typing import List, Tuple
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

from config import LLM_MODEL_NAME, LLM_TASK
from rag_pipeline.retriever import get_contract_vectorstore
from rag_pipeline.risk_db import get_risk_vectorstore

_llm = None

def _get_llm():
    """Singleton Gemma (or other HF causal model) via HuggingFacePipeline."""
    global _llm
    if _llm is not None:
        return _llm

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # set this in your env

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME,
        use_auth_token=token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        use_auth_token=token,
    )

    hf_pipe = pipeline(
        LLM_TASK,            # usually "text-generation"
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
    )

    _llm = HuggingFacePipeline(pipeline=hf_pipe)
    return _llm

def _dedup_docs(docs):
    seen = set()
    unique = []
    for d in docs:
        txt = d.page_content.strip()
        if txt not in seen:
            seen.add(txt)
            unique.append(d)
    return unique

def _clean_chunk(t: str) -> str:
    """Remove very short / duplicate lines inside a chunk for cleaner context."""
    lines = t.split("\n")
    cleaned = []
    seen = set()
    for line in lines:
        l = line.strip()
        if len(l) < 25:
            continue
        if l not in seen:
            seen.add(l)
            cleaned.append(l)
    return "\n".join(cleaned)

def get_context(query: str, k_contract: int = 4, k_risk: int = 4) -> str:
    """Retrieve relevant, deduped context from contract + risk DB."""
    contract_vs = get_contract_vectorstore()
    risk_vs = get_risk_vectorstore()

    docs1 = contract_vs.similarity_search(query, k=k_contract)
    docs2 = risk_vs.similarity_search(query, k=k_risk)
    docs = _dedup_docs(docs1 + docs2)

    return "\n\n".join(_clean_chunk(d.page_content) for d in docs)

def chat_once(question: str, chat_history: List[Tuple[str, str]]) -> str:
    """
    Single-turn chat with light conversation awareness.
    chat_history: list of (role, text) where role is "user" or "assistant".
    """
    llm = _get_llm()

    # Last few turns for conversational context
    last_turns = chat_history[-6:]
    history_str = ""
    for role, text in last_turns:
        if role == "user":
            history_str += f"User: {text}\n"
        else:
            history_str += f"Assistant: {text}\n"

    context = get_context(question)

    prompt = f"""
You are a contract risk analysis assistant.

Conversation so far:
{history_str}

Context (contract clauses and risk patterns):
{context}

User question:
{question}

Requirements:
- Use the context to identify relevant clauses and risks.
- Explain in your own words.
- Do NOT copy long passages from the contract.
- Use short paragraphs or bullet points.
- Never say you are giving legal advice; just explain risk.

### Answer:
"""

    raw = llm.invoke(prompt)

    text = str(raw).strip()
    if "### Answer:" in text:
        text = text.split("### Answer:", 1)[1].strip()

    # 🔹 Deduplicate repeated paragraphs
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    seen = set()
    deduped = []
    for p in paras:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    text = "\n\n".join(deduped)

    return text

def summarize_contract_basic() -> str:
    """Basic (short) summary, less structured but quick."""
    llm = _get_llm()
    context = get_context(
        "overall contract structure incident SLAs obligations liabilities risks missing clauses",
        k_contract=5,
        k_risk=5,
    )

    cleaned_lines = []
    seen = set()
    for line in context.split("\n"):
        l = line.strip()
        if len(l) < 20:
            continue
        if l not in seen:
            cleaned_lines.append(l)
            seen.add(l)
    short_context = "\n".join(cleaned_lines[:10])

    prompt = f"""
You are a contract risk analysis assistant.

Below is contract context from multiple clauses. 
Summarize the contract in your own words, focusing on key obligations and main risks.
Do NOT copy the text below.
Do NOT echo the instructions.
Return 5–8 bullet points in Markdown, where each bullet starts with "- " and no leading spaces.

### Context:
{short_context}

### Summary:
"""

    raw = llm.invoke(prompt)
    text = str(raw).strip()

    if "### Summary:" in text:
        text = text.split("### Summary:", 1)[1].strip()

    #return text
    # 🔧 Normalize lines into real Markdown bullets
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    formatted = []
    for line in lines:
        stripped = line.strip()

        # Convert bullets like "• something" to "- something"
        if stripped.startswith("•"):
            formatted.append("- " + stripped.lstrip("•").strip())
        # Convert bullets starting with "*" or "-" (any spacing) to "- "
        elif stripped.startswith(("*", "-")):
            formatted.append("- " + stripped.lstrip("*-").strip())
        # Convert numbered bullets "1. text" into "- text"
        elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1] == ".":
            formatted.append("- " + stripped[2:].strip())
        else:
            # If it's just a sentence, still make it a bullet
            formatted.append("- " + stripped)

    return "\n".join(formatted)

def summarize_contract_smart() -> str:
    """
    SMART RISK SUMMARY:
    Structured markdown with sections:
    1. Key Obligations
    2. SLA & Incident Response
    3. Indemnity & Liability
    4. Confidentiality & Data Protection
    5. Missing Clauses & Red Flags
    6. Overall Risk Assessment
    """
    llm = _get_llm()

    context = get_context(
        "key obligations SLA incident response indemnity limitation of liability confidentiality data protection missing clauses risks",
        k_contract=8,
        k_risk=8,
    )

    cleaned_lines = []
    seen = set()
    for line in context.split("\n"):
        l = line.strip()
        if len(l) < 20:
            continue
        if l not in seen:
            cleaned_lines.append(l)
            seen.add(l)
    short_context = "\n".join(cleaned_lines[:15])

    prompt = f"""
You are a contract risk analysis assistant for Customer (AIGENIE).

Using the context below, produce a SMART RISK SUMMARY in **Markdown** with the following sections:

## 1. Key Obligations
- Bullet points describing main responsibilities of Supplier and Customer.

## 2. SLA & Incident Response
- Bullet points describing incident categories, timelines, and any risky timelines.
- Explicitly label items as **Low**, **Medium**, or **High** risk where appropriate.

## 3. Indemnity & Liability
- Bullet points on indemnification obligations and liability caps/exclusions.
- Highlight any one-sided or unusually narrow supplier obligations.

## 4. Confidentiality & Data Protection
- Bullet points on confidentiality, data security, and breach notification.
- Note any missing or weak protections as risk items.

## 5. Missing Clauses & Red Flags
- List clauses that appear to be missing or weak (e.g., indemnity, audit rights, insurance, data protection).
- Clearly label these as **High** or **Medium** risk where appropriate.

## 6. Overall Risk Assessment
- 2–3 bullet points summarizing overall risk level for AIGENIE (e.g., ""Overall: Medium–High risk due to weak SLA timelines and missing audit rights"").

Rules:
- Do NOT copy long passages from the context.
- Do NOT echo these instructions.
- Use concise bullet points.
- If a section has no clear info, write ""- No significant points identified."" under that section.

### Context:
{short_context}

### SmartSummary:
"""

    raw = llm.invoke(prompt)
    text = str(raw).strip()

    if "### SmartSummary:" in text:
        text = text.split("### SmartSummary:", 1)[1].strip()

    return text