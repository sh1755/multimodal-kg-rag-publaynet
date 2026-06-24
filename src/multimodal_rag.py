"""CLIP-based visual retrieval over PubLayNet pages."""
from __future__ import annotations

import numpy as np
import open_clip
import torch

from .utils import OUTPUTS_DIR


class MultimodalRAG:
    def __init__(self) -> None:
        self.embeddings = np.load(OUTPUTS_DIR / "clip_embeddings.npy")
        self.page_ids = np.load(OUTPUTS_DIR / "clip_page_ids.npy", allow_pickle=True)
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model.eval()

    def search_pages(self, query: str, top_k: int = 10) -> list[dict]:
        with torch.no_grad():
            tokens = self.tokenizer([query])
            feat = self.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        q = feat.squeeze(0).cpu().numpy()
        scores = self.embeddings @ q  # cosine, since both are L2-normalised
        idx = np.argsort(-scores)[:top_k]
        return [
            {
                "rank": r + 1,
                "page_id": str(self.page_ids[i]),
                "score": float(scores[i]),
            }
            for r, i in enumerate(idx)
        ]
