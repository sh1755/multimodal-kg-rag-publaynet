"""Interactive end-to-end demo: retrieval + generation + KG explanation."""
from __future__ import annotations

import argparse

from .fusion import fuse
from .generate import generate_answer
from .kg_rag import KGRAG
from .multimodal_rag import MultimodalRAG
from .text_rag import TextRAG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="Natural-language question.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backend", choices=["auto", "anthropic", "flan", "extractive"],
                        default="auto")
    args = parser.parse_args()

    query = args.query or input("Ask your question: ")

    text_rag = TextRAG()
    mm_rag = MultimodalRAG()
    kg_rag = KGRAG()

    t = text_rag.search_pages(query, top_k=15)
    v = mm_rag.search_pages(query, top_k=15)
    g = kg_rag.search_pages(query, top_k=15)
    fused = fuse(query, t, v, g, top_k=args.top_k)

    # Attach the best chunk text from text RAG for context.
    text_by_page = {r["page_id"]: r["text"] for r in t}
    for r in fused:
        r["text"] = text_by_page.get(r["page_id"], "")

    print("\n" + "=" * 72)
    print(f"Query: {query}")
    print("=" * 72)
    print("\nTop retrieved pages (fused score):")
    for r in fused:
        print(f"  #{r['rank']}  {r['page_id']}  "
              f"score={r['score']:.3f}  "
              f"[text={r['text_score']:.2f} vis={r['visual_score']:.2f} kg={r['kg_score']:.2f}]")

    top_page = fused[0]["page_id"] if fused else None
    kg_triples = kg_rag.explain(top_page, query=query) if top_page else []

    print("\nKG evidence for top page:")
    for line in kg_triples:
        print("  •", line)

    result = generate_answer(query, fused, kg_triples=kg_triples, backend=args.backend)
    print(f"\nAnswer (backend={result.backend}):")
    print(result.answer)
    if result.cited_pages:
        print(f"\nCited pages: {', '.join(result.cited_pages)}")


if __name__ == "__main__":
    main()
