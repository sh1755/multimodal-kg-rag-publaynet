"""Ingest the small PubLayNet subset.

For each page we:
  1. Read the rendered PNG and its layout JSON.
  2. OCR the full page (full-page text feeds the baseline RAG).
  3. Also OCR each *text/title/list* region by cropping to its bbox — these
     give us cleaner, region-level chunks that the enhanced retriever uses.
  4. Record counts of every PubLayNet category (text/title/list/table/figure)
     for layout-aware boosting and KG construction downstream.

Outputs:
  outputs/ocr_pages.csv     — one row per page (page-level OCR + counts)
  outputs/ocr_regions.csv   — one row per text region (region-level chunks)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from .utils import OUTPUTS_DIR, configure_tesseract

CATEGORIES = ["text", "title", "list", "table", "figure"]


def _ocr(image, pytesseract):
    try:
        return pytesseract.image_to_string(image).strip()
    except Exception as exc:  # pragma: no cover
        print(f"[warn] OCR failed: {exc}")
        return ""


def build_ocr_dataset(limit: int = 100) -> None:
    configure_tesseract()
    import pytesseract  # imported after configure for cmd to take effect

    print(f"Loading first {limit} samples of small-publaynet-wds…")
    dataset = load_dataset(
        "lhoestq/small-publaynet-wds",
        split=f"train[:{limit}]",
    )

    page_rows: list[dict] = []
    region_rows: list[dict] = []

    for sample in tqdm(dataset, desc="OCR"):
        page_id = sample["__key__"]
        image = sample["png"]
        annotations = sample["json"]["annotations"]

        # ---- page-level ---------------------------------------------------
        full_text = _ocr(image, pytesseract)
        counts = {f"num_{c}": 0 for c in CATEGORIES}
        for a in annotations:
            key = f"num_{a['category_name']}"
            if key in counts:
                counts[key] += 1

        page_rows.append(
            {
                "page_id": page_id,
                "ocr_text": full_text,
                **counts,
                "num_annotations": len(annotations),
            }
        )

        # ---- region-level -------------------------------------------------
        # Crop and OCR each text-bearing region for cleaner chunks.
        for i, ann in enumerate(annotations):
            cat = ann["category_name"]
            if cat not in {"text", "title", "list"}:
                continue
            # COCO bbox: [x, y, w, h]
            x, y, w, h = ann["bbox"]
            try:
                crop = image.crop((x, y, x + w, y + h))
            except Exception:
                continue
            text = _ocr(crop, pytesseract)
            if len(text) < 20:
                continue
            region_rows.append(
                {
                    "page_id": page_id,
                    "region_idx": i,
                    "category": cat,
                    "bbox": [float(v) for v in ann["bbox"]],
                    "text": text,
                }
            )

    pages_df = pd.DataFrame(page_rows)
    regions_df = pd.DataFrame(region_rows)

    pages_path = OUTPUTS_DIR / "ocr_pages.csv"
    regions_path = OUTPUTS_DIR / "ocr_regions.csv"
    pages_df.to_csv(pages_path, index=False)
    regions_df.to_csv(regions_path, index=False)

    print(f"Saved {len(pages_df)} pages   → {pages_path}")
    print(f"Saved {len(regions_df)} regions → {regions_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args()
    build_ocr_dataset(limit=args.limit)
