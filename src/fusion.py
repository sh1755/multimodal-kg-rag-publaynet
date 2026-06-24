"""Late-fusion combiner: text + CLIP visual + KG entity overlap.

Each component returns a list of {page_id, score}. Scores are min-max
normalised within each component and then combined using a weighted sum.

For visually-oriented queries such as figure, chart, plot, or bar chart,
the visual CLIP component is given more weight. A small layout-aware boost is
also added when the query explicitly asks for tables, figures, or titles.
"""

from __future__ import annotations

import pandas as pd

from .utils import OUTPUTS_DIR


def _minmax(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}

    vals = list(d.values())
    lo, hi = min(vals), max(vals)

    if hi - lo < 1e-9:
        return {k: 1.0 for k in d}

    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def _layout_bonus(query: str, page_id: str, pages_df: pd.DataFrame) -> float:
    q = query.lower()

    row = pages_df[pages_df["page_id"] == page_id]

    if row.empty:
        return 0.0

    r = row.iloc[0]

    bonus = 0.0

    if any(word in q for word in ["table", "tables"]):
        bonus += 0.08 * min(int(r.get("num_table", 0) or 0), 3)

    if any(word in q for word in ["figure", "figures", "chart", "charts", "bar", "plot", "plots"]):
        bonus += 0.08 * min(int(r.get("num_figure", 0) or 0), 3)

    if any(word in q for word in ["title", "heading", "section"]):
        bonus += 0.03 * min(int(r.get("num_title", 0) or 0), 3)

    return bonus


def _choose_weights(query: str) -> tuple[float, float, float]:
    q = query.lower()

    visual_terms = [
        "figure",
        "figures",
        "chart",
        "charts",
        "bar",
        "plot",
        "plots",
        "image",
        "visual",
        "diagram",
    ]

    table_terms = [
        "table",
        "tables",
    ]

    if any(term in q for term in visual_terms):
        return 0.35, 0.50, 0.15

    if any(term in q for term in table_terms):
        return 0.45, 0.30, 0.25

    return 0.55, 0.25, 0.20


def fuse(
    query: str,
    text_results: list[dict],
    visual_results: list[dict],
    kg_results: list[dict],
    weights: tuple[float, float, float] | None = None,
    top_k: int = 10,
) -> list[dict]:

    if weights is None:
        weights = _choose_weights(query)

    w_t, w_v, w_k = weights

    text = _minmax({
        r["page_id"]: float(r.get("score", 0.0))
        for r in text_results
    })

    vis = _minmax({
        r["page_id"]: float(r.get("score", 0.0))
        for r in visual_results
    })

    kg = _minmax({
        r["page_id"]: float(r.get("score", 0.0))
        for r in kg_results
    })

    candidates = set(text) | set(vis) | set(kg)

    pages_df = pd.read_csv(OUTPUTS_DIR / "ocr_pages.csv")

    fused: list[dict] = []

    for pid in candidates:
        text_score = text.get(pid, 0.0)
        visual_score = vis.get(pid, 0.0)
        kg_score = kg.get(pid, 0.0)
        layout_score = _layout_bonus(query, pid, pages_df)

        score = (
            w_t * text_score
            + w_v * visual_score
            + w_k * kg_score
            + layout_score
        )

        fused.append({
            "page_id": pid,
            "score": score,
            "text_score": text_score,
            "visual_score": visual_score,
            "kg_score": kg_score,
            "layout_bonus": layout_score,
        })

    fused.sort(key=lambda x: x["score"], reverse=True)

    for rank, item in enumerate(fused[:top_k], 1):
        item["rank"] = rank

    return fused[:top_k]