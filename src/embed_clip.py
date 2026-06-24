"""CLIP ViT-B-32 image embeddings for full pages.

We embed one image vector per page. Query-time, we encode the text query with
the CLIP text tower and rank pages by cosine similarity. This captures visual
characteristics (presence of tables, figures, dense vs sparse layout) that OCR
text alone misses.
"""
from __future__ import annotations

import argparse

import numpy as np
import open_clip
import torch
from datasets import load_dataset
from tqdm import tqdm

from .utils import OUTPUTS_DIR


def build(limit: int = 100) -> None:
    print("Loading CLIP ViT-B-32 (openai)…")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    dataset = load_dataset(
        "lhoestq/small-publaynet-wds", split=f"train[:{limit}]"
    )

    vectors, ids = [], []
    with torch.no_grad():
        for sample in tqdm(dataset, desc="CLIP"):
            img = preprocess(sample["png"]).unsqueeze(0).to(device)
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            vectors.append(feat.squeeze(0).cpu().numpy())
            ids.append(sample["__key__"])

    emb = np.vstack(vectors).astype("float32")
    np.save(OUTPUTS_DIR / "clip_embeddings.npy", emb)
    np.save(OUTPUTS_DIR / "clip_page_ids.npy", np.array(ids))
    print(f"Saved CLIP embeddings ({emb.shape}) → outputs/clip_embeddings.npy")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args()
    build(limit=args.limit)
