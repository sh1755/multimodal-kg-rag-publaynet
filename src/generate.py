"""Answer generation with citations.

Three backends, auto-selected:

  1. Anthropic Claude (if ANTHROPIC_API_KEY is in env). Uses the messages API.
  2. Local HuggingFace flan-T5 (CPU OK).
  3. Extractive fallback: stitches together the top retrieved chunks. Always
     available so the pipeline never breaks during demos.

The prompt includes retrieved chunks tagged with their page_id so the model is
encouraged to cite, and the KG triples for the top page so it can ground
relational claims.
"""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass


SYSTEM_PROMPT = (
    "You are a careful scientific QA assistant. Answer the user's question using ONLY the "
    "provided context. Cite the page_id of every fact in square brackets, e.g. "
    "[PMC1234_00002]. If the context does not contain the answer, say so explicitly."
)


@dataclass
class GenerationResult:
    answer: str
    backend: str
    cited_pages: list[str]


def _format_context(chunks: list[dict], kg_triples: list[str]) -> str:
    parts = ["# Retrieved evidence"]
    for c in chunks:
        parts.append(f"[{c['page_id']}] {c.get('text', '')[:600]}")
    if kg_triples:
        parts.append("\n# Knowledge-graph facts")
        parts.extend(f"- {t}" for t in kg_triples)
    return "\n\n".join(parts)


def _extract_citations(answer: str, candidate_pages: list[str]) -> list[str]:
    found = []
    for p in candidate_pages:
        if p in answer and p not in found:
            found.append(p)
    return found


# ---------------------------------------------------------------- backends ---

def _gen_anthropic(prompt: str) -> str:
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError("anthropic package not installed") from e
    client = Anthropic()
    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in msg.content if hasattr(block, "text"))


_FLAN = None


def _gen_flan(prompt: str) -> str:
    global _FLAN
    if _FLAN is None:
        from transformers import pipeline

        _FLAN = pipeline("text2text-generation", model="google/flan-t5-base")
    full = f"{SYSTEM_PROMPT}\n\n{prompt}\n\nAnswer:"
    out = _FLAN(full, max_new_tokens=256, do_sample=False)
    return out[0]["generated_text"].strip()


def _gen_extractive(prompt: str, chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant context was retrieved."
    bullets = "\n".join(
        f"- [{c['page_id']}] {c.get('text','')[:280].strip()}" for c in chunks[:3]
    )
    return (
        "Extractive answer (no LLM available). Top supporting passages:\n" + bullets
    )


# --------------------------------------------------------------- public API ---

def generate_answer(
    query: str,
    chunks: list[dict],
    kg_triples: list[str] | None = None,
    backend: str = "auto",
) -> GenerationResult:
    kg_triples = kg_triples or []
    context = _format_context(chunks, kg_triples)
    prompt = textwrap.dedent(
        f"""
        {context}

        # Question
        {query}

        # Instructions
        Write a concise, factual answer (≤120 words). Cite every claim with the page_id
        in square brackets. If the evidence is insufficient, say so.
        """
    ).strip()

    chosen = backend
    if backend == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            chosen = "anthropic"
        else:
            chosen = "extractive"  # safest default; flan needs ~1GB download

    try:
        if chosen == "anthropic":
            answer = _gen_anthropic(prompt)
        elif chosen == "flan":
            answer = _gen_flan(prompt)
        else:
            answer = _gen_extractive(prompt, chunks)
            chosen = "extractive"
    except Exception as exc:
        print(f"[warn] {chosen} backend failed: {exc}. Falling back to extractive.")
        answer = _gen_extractive(prompt, chunks)
        chosen = "extractive"

    cited = _extract_citations(answer, [c["page_id"] for c in chunks])
    return GenerationResult(answer=answer, backend=chosen, cited_pages=cited)
