"""Quantitative evaluation: baseline (text-only) vs enhanced (text + CLIP + KG).

Because the small PubLayNet subset has no QA labels, we construct silver-
standard relevance judgements automatically from the eval questions.

A page is relevant to a question if:
1. its OCR text contains at least one of the question keywords, and
2. its layout satisfies any structural requirement such as table or figure.

Metrics:
- Recall@k
- MRR
- nDCG@10
"""

from __future__ import annotations

import json
import math

import pandas as pd

from .fusion import fuse
from .kg_rag import KGRAG
from .multimodal_rag import MultimodalRAG
from .text_rag import TextRAG
from .utils import EVAL_DIR, OUTPUTS_DIR, read_jsonl


def build_gold(questions: list[dict], pages_df: pd.DataFrame) -> dict[str, set[str]]:
    gold: dict[str, set[str]] = {}

    for q in questions:
        keywords = [k.lower() for k in q["keywords"]]
        relevant: set[str] = set()

        for _, row in pages_df.iterrows():
            text = str(row.get("ocr_text", "")).lower()

            if not text.strip():
                continue

            hits = sum(1 for kw in keywords if kw in text)

            if hits == 0:
                continue

            if q.get("needs_table") and int(row.get("num_table", 0) or 0) < 1:
                continue

            if q.get("needs_figure") and int(row.get("num_figure", 0) or 0) < 1:
                continue

            relevant.add(row["page_id"])

        gold[q["qid"]] = relevant

    return gold


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")

    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return float("nan")

    for i, page_id in enumerate(retrieved, 1):
        if page_id in relevant:
            return 1.0 / i

    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return float("nan")

    dcg = 0.0

    for i, page_id in enumerate(retrieved[:k], 1):
        if page_id in relevant:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(k, len(relevant))

    idcg = sum(
        1.0 / math.log2(i + 1)
        for i in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg > 0 else 0.0


def _run_baseline(query: str, text_rag: TextRAG, k: int = 10) -> list[str]:
    results = text_rag.search_pages(query, top_k=k)
    return [r["page_id"] for r in results]


def _run_enhanced(
    query: str,
    text_rag: TextRAG,
    multimodal_rag: MultimodalRAG,
    kg_rag: KGRAG,
    k: int = 10,
) -> list[str]:

    text_results = text_rag.search_pages(query, top_k=20)
    visual_results = multimodal_rag.search_pages(query, top_k=20)
    kg_results = kg_rag.search_pages(query, top_k=20)

    fused_results = fuse(
        query,
        text_results,
        visual_results,
        kg_results,
        top_k=k
    )

    return [r["page_id"] for r in fused_results]


def main() -> None:
    questions = read_jsonl(EVAL_DIR / "questions.jsonl")
    pages_df = pd.read_csv(OUTPUTS_DIR / "ocr_pages.csv")

    gold = build_gold(questions, pages_df)

    keep = [
        q for q in questions
        if len(gold[q["qid"]]) > 0
    ]

    dropped = len(questions) - len(keep)

    print(
        f"Eval: {len(keep)}/{len(questions)} questions have at least one relevant page "
        f"({dropped} dropped due to no matches in this {len(pages_df)}-page subset)."
    )

    text_rag = TextRAG()
    multimodal_rag = MultimodalRAG()
    kg_rag = KGRAG()

    rows: list[dict] = []
    aggregate = {
        "baseline": [],
        "enhanced": []
    }

    for q in keep:
        relevant = gold[q["qid"]]

        baseline_results = _run_baseline(
            q["question"],
            text_rag,
            k=10
        )

        enhanced_results = _run_enhanced(
            q["question"],
            text_rag,
            multimodal_rag,
            kg_rag,
            k=10
        )

        for system_name, retrieved in (
            ("baseline", baseline_results),
            ("enhanced", enhanced_results)
        ):

            metric_row = {
                "qid": q["qid"],
                "system": system_name,
                "question": q["question"],
                "n_relevant": len(relevant),
                "recall@3": recall_at_k(retrieved, relevant, 3),
                "recall@5": recall_at_k(retrieved, relevant, 5),
                "recall@10": recall_at_k(retrieved, relevant, 10),
                "mrr": mrr(retrieved, relevant),
                "ndcg@10": ndcg_at_k(retrieved, relevant, 10),
                "top1": retrieved[0] if retrieved else "",
                "top1_hit": int(bool(retrieved and retrieved[0] in relevant)),
            }

            rows.append(metric_row)
            aggregate[system_name].append(metric_row)

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUTS_DIR / "metrics_per_query.csv",
        index=False,
        encoding="utf-8"
    )

    summary = {}

    for system_name, metrics in aggregate.items():
        summary[system_name] = {
            "n_queries": len(metrics),
            "recall@3": float(pd.Series([m["recall@3"] for m in metrics]).mean()),
            "recall@5": float(pd.Series([m["recall@5"] for m in metrics]).mean()),
            "recall@10": float(pd.Series([m["recall@10"] for m in metrics]).mean()),
            "mrr": float(pd.Series([m["mrr"] for m in metrics]).mean()),
            "ndcg@10": float(pd.Series([m["ndcg@10"] for m in metrics]).mean()),
            "top1_accuracy": float(pd.Series([m["top1_hit"] for m in metrics]).mean()),
        }

    with open(
        OUTPUTS_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    cols = [
        "recall@3",
        "recall@5",
        "recall@10",
        "mrr",
        "ndcg@10",
        "top1_accuracy"
    ]

    lines = [
        "# Retrieval metrics: baseline vs enhanced\n",
        "| System | " + " | ".join(cols) + " |",
        "|--------|" + "|".join("---" for _ in cols) + "|"
    ]

    for system_name in ("baseline", "enhanced"):
        scores = summary[system_name]

        lines.append(
            f"| {system_name} | "
            + " | ".join(f"{scores[c]:.3f}" for c in cols)
            + " |"
        )

    delta = {
        c: summary["enhanced"][c] - summary["baseline"][c]
        for c in cols
    }

    lines.append(
        "| **Delta (enhanced - baseline)** | "
        + " | ".join(f"{delta[c]:+.3f}" for c in cols)
        + " |"
    )

    markdown_table = "\n".join(lines) + "\n"

    (OUTPUTS_DIR / "metrics_table.md").write_text(
        markdown_table,
        encoding="utf-8"
    )

    print("\n" + markdown_table)
    print(f"Per-query metrics saved to {OUTPUTS_DIR / 'metrics_per_query.csv'}")


if __name__ == "__main__":
    main()