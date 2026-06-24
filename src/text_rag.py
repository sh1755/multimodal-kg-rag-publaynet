"""Baseline text-only RAG. MiniLM embeddings + FAISS cosine search.

Returns chunk-level results, then aggregates to page-level scores so the
evaluation is apples-to-apples with the multimodal retriever (which works at
page granularity).
"""
from __future__ import annotations

from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .utils import OUTPUTS_DIR

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TextRAG:
    def __init__(self) -> None:
        self.index = faiss.read_index(str(OUTPUTS_DIR / "text.faiss"))
        self.chunks = pd.read_csv(OUTPUTS_DIR / "text_chunks.csv")
        self.model = SentenceTransformer(MODEL_NAME)

    def _encode(self, query: str) -> np.ndarray:
        v = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return v.astype("float32")

    def search_chunks(self, query: str, top_k: int = 20) -> list[dict]:
        scores, idx = self.index.search(self._encode(query), top_k)
        out: list[dict] = []
        for rank, (s, i) in enumerate(zip(scores[0], idx[0])):
            if i < 0:
                continue
            row = self.chunks.iloc[i]
            out.append(
                {
                    "rank": rank + 1,
                    "page_id": row["page_id"],
                    "category": row["category"],
                    "score": float(s),
                    "text": str(row["text"])[:800],
                }
            )
        return out

    def search_pages(self, query: str, top_k: int = 10, chunk_k: int = 40) -> list[dict]:
        """Aggregate chunk scores to a per-page score (max-pool)."""
        chunks = self.search_chunks(query, top_k=chunk_k)
        page_score: dict[str, float] = defaultdict(lambda: -1.0)
        page_text: dict[str, str] = {}
        for c in chunks:
            if c["score"] > page_score[c["page_id"]]:
                page_score[c["page_id"]] = c["score"]
                page_text[c["page_id"]] = c["text"]
        ranked = sorted(page_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "rank": r + 1,
                "page_id": pid,
                "score": score,
                "text": page_text[pid],
            }
            for r, (pid, score) in enumerate(ranked)
        ]
