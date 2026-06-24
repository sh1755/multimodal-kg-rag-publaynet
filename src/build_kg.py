"""Build a knowledge graph from OCR text + layout metadata.

Node types:
  • page        — one per document page
  • entity      — extracted named entities (spaCy NER, lowercased lemma)
  • layout      — one node per layout category present on a page (table/figure/…)

Edge types:
  • (page) -[MENTIONS]-> (entity)         from NER
  • (page) -[CONTAINS]-> (layout)         from PubLayNet annotations
  • (entity) -[CO_OCCURS_WITH]-> (entity) entities appearing on the same page

Falls back gracefully when spaCy model isn't installed: uses a regex /
noun-chunk-ish heuristic so the pipeline still runs.
"""
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

import networkx as nx
import pandas as pd

from .utils import OUTPUTS_DIR

# Entity types we keep from spaCy's default NER.
KEEP_LABELS = {"ORG", "PERSON", "GPE", "PRODUCT", "WORK_OF_ART", "EVENT", "NORP"}
MIN_ENTITY_LEN = 3
MAX_ENTITY_LEN = 60


def _load_spacy():
    try:
        import spacy

        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("[warn] en_core_web_sm not found; run "
                  "`python -m spacy download en_core_web_sm` for richer KGs.")
            return None
    except ImportError:
        return None


def _fallback_entities(text: str) -> list[str]:
    """Regex fallback when spaCy isn't available: capitalised n-grams."""
    cands = re.findall(r"\b([A-Z][a-zA-Z0-9\-]{2,}(?:\s+[A-Z][a-zA-Z0-9\-]{2,}){0,3})\b", text)
    return [c.strip() for c in cands]


def _extract_entities(text: str, nlp) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    text = text[:5000]  # cap per-page to keep things fast
    if nlp is None:
        ents = _fallback_entities(text)
    else:
        doc = nlp(text)
        ents = [e.text for e in doc.ents if e.label_ in KEEP_LABELS]
    cleaned = []
    seen = set()
    for e in ents:
        e_norm = re.sub(r"\s+", " ", e).strip().lower()
        if MIN_ENTITY_LEN <= len(e_norm) <= MAX_ENTITY_LEN and e_norm not in seen:
            seen.add(e_norm)
            cleaned.append(e_norm)
    return cleaned


def build() -> nx.MultiDiGraph:
    pages = pd.read_csv(OUTPUTS_DIR / "ocr_pages.csv")
    nlp = _load_spacy()
    g = nx.MultiDiGraph()

    entity_freq: Counter[str] = Counter()
    page_entities: dict[str, list[str]] = {}

    for _, row in pages.iterrows():
        pid = row["page_id"]
        g.add_node(pid, type="page")

        # Layout edges.
        for cat in ("text", "title", "list", "table", "figure"):
            n = int(row.get(f"num_{cat}", 0) or 0)
            if n > 0:
                layout_node = f"layout::{cat}"
                g.add_node(layout_node, type="layout", category=cat)
                g.add_edge(pid, layout_node, relation="CONTAINS", count=n)

        # Entity edges.
        ents = _extract_entities(row.get("ocr_text", ""), nlp)
        page_entities[pid] = ents
        for e in ents:
            entity_freq[e] += 1
            ent_node = f"entity::{e}"
            g.add_node(ent_node, type="entity", name=e)
            g.add_edge(pid, ent_node, relation="MENTIONS")

    # Co-occurrence edges (capped to entities that appear ≥2 times to avoid noise).
    common = {e for e, c in entity_freq.items() if c >= 2}
    for pid, ents in page_entities.items():
        kept = [e for e in ents if e in common]
        for a, b in combinations(sorted(set(kept)), 2):
            g.add_edge(
                f"entity::{a}", f"entity::{b}",
                relation="CO_OCCURS_WITH", page=pid,
            )

    out_path = OUTPUTS_DIR / "kg.graphml"
    nx.write_graphml(g, out_path)
    print(f"KG: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges → {out_path}")
    print(f"     pages:   {sum(1 for _, d in g.nodes(data=True) if d.get('type')=='page')}")
    print(f"     entities:{sum(1 for _, d in g.nodes(data=True) if d.get('type')=='entity')}")
    print(f"     layout:  {sum(1 for _, d in g.nodes(data=True) if d.get('type')=='layout')}")
    return g


if __name__ == "__main__":
    build()
