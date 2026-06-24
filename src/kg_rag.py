"""KG-aware retrieval.

Uses the knowledge graph in two ways:
  1. *Entity overlap scoring*: extract entities from the query (NER or
     fallback), then score pages by how many of those entities they mention.
  2. *Graph expansion*: for each query entity, walk CO_OCCURS_WITH edges to
     find related entities and include the pages that mention them too.
This component is meant to be combined with text/visual scores in fusion.py.
"""
from __future__ import annotations

import re
from collections import defaultdict

import networkx as nx

from .utils import OUTPUTS_DIR
from .build_kg import _extract_entities, _load_spacy


class KGRAG:
    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.read_graphml(OUTPUTS_DIR / "kg.graphml")
        self.nlp = _load_spacy()
        # Pre-compute: page -> set(entity_name)
        self.page_to_entities: dict[str, set[str]] = defaultdict(set)
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "MENTIONS":
                self.page_to_entities[u].add(v.replace("entity::", ""))
        # entity name -> entity node id
        self.entity_nodes = {
            d["name"]: n
            for n, d in self.graph.nodes(data=True)
            if d.get("type") == "entity"
        }

    # ------------------------------------------------------------------ utils
    def _query_entities(self, query: str) -> set[str]:
        ents = set(_extract_entities(query, self.nlp))
        # also throw in lowercased content words ≥ 4 chars so short queries
        # still produce something (NER often whiffs on lower-cased queries)
        for tok in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", query.lower()):
            if tok in self.entity_nodes:
                ents.add(tok)
        return ents

    def _expand(self, entities: set[str], hops: int = 1) -> set[str]:
        out = set(entities)
        frontier = set(entities)
        for _ in range(hops):
            next_frontier: set[str] = set()
            for e in frontier:
                node = self.entity_nodes.get(e)
                if node is None:
                    continue
                for nb in self.graph.neighbors(node):
                    nb_data = self.graph.nodes[nb]
                    if nb_data.get("type") == "entity":
                        name = nb_data.get("name", nb.replace("entity::", ""))
                        if name not in out:
                            next_frontier.add(name)
            out |= next_frontier
            frontier = next_frontier
        return out

    # -------------------------------------------------------------- retrieval
    def search_pages(self, query: str, top_k: int = 10, hops: int = 1) -> list[dict]:
        q_ents = self._query_entities(query)
        expanded = self._expand(q_ents, hops=hops) if q_ents else set()

        scores: dict[str, float] = defaultdict(float)
        for page, ents in self.page_to_entities.items():
            if not ents:
                continue
            direct = len(ents & q_ents)
            indirect = len(ents & (expanded - q_ents))
            # Direct hits weighted more than 1-hop neighbours.
            score = direct * 1.0 + indirect * 0.4
            if score > 0:
                scores[page] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "rank": r + 1,
                "page_id": pid,
                "score": float(s),
                "matched_entities": sorted(self.page_to_entities[pid] & (q_ents | expanded))[:8],
            }
            for r, (pid, s) in enumerate(ranked)
        ]

    def explain(self, page_id: str, query: str | None = None, max_items: int = 8) -> list[str]:
        """Return human-readable triples explaining why this page is relevant."""
        if page_id not in self.graph:
            return ["No KG evidence for this page."]
        q_ents = self._query_entities(query) if query else set()
        lines: list[str] = []
        for nb in self.graph.neighbors(page_id):
            data = self.graph.nodes[nb]
            t = data.get("type")
            if t == "entity":
                name = data.get("name", nb.replace("entity::", ""))
                tag = " [matches query]" if name in q_ents else ""
                lines.append(f"{page_id} —MENTIONS→ {name}{tag}")
            elif t == "layout":
                lines.append(f"{page_id} —CONTAINS→ {data.get('category')}")
        # Put query-matching entities first.
        lines.sort(key=lambda s: 0 if "[matches query]" in s else 1)
        return lines[:max_items]
