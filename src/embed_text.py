"""Build a FAISS index over OCR text using sentence-transformers MiniLM.

We index *region-level* chunks (cleaner than full-page OCR) and fall back to
full-page text where regions are unavailable.
"""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .utils import OUTPUTS_DIR

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build() -> None:
    regions_path = OUTPUTS_DIR / "ocr_regions.csv"
    pages_path = OUTPUTS_DIR / "ocr_pages.csv"

    if regions_path.exists():
        df = pd.read_csv(regions_path)
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.len() >= 30].reset_index(drop=True)
        texts = df["text"].tolist()
        meta = df[["page_id", "region_idx", "category"]].to_dict("records")
        source = "regions"
    else:
        df = pd.read_csv(pages_path)
        df = df[df["ocr_text"].str.len() >= 50].reset_index(drop=True)
        texts = df["ocr_text"].tolist()
        meta = [
            {"page_id": r["page_id"], "region_idx": -1, "category": "page"}
            for _, r in df.iterrows()
        ]
        source = "pages"

    print(f"Embedding {len(texts)} {source} chunks with {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")

    # Inner-product index = cosine similarity since vectors are normalised.
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, str(OUTPUTS_DIR / "text.faiss"))
    np.save(OUTPUTS_DIR / "text_embeddings.npy", emb)
    pd.DataFrame(meta).assign(text=texts).to_csv(
        OUTPUTS_DIR / "text_chunks.csv", index=False
    )
    print(f"Saved text index ({emb.shape[0]} × {emb.shape[1]}) → outputs/text.faiss")


if __name__ == "__main__":
    build()
