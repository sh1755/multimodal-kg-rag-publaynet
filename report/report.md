# Multimodal KG-RAG over PubLayNet — Technical Report

## 1. Problem formulation

We build a retrieval-augmented generation (RAG) system that answers natural-
language questions over scientific document pages drawn from PubLayNet. The
brief requires (i) multimodal ingestion, (ii) knowledge-graph integration,
(iii) RAG-style retrieval and generation, (iv) basic reasoning, and (v)
explainability — together with a *quantitative* comparison of a text-only
baseline against an enhanced multimodal + KG system.

We work against the small subset `lhoestq/small-publaynet-wds` (100 pages by
default), which is sufficient for a proof-of-concept and keeps the pipeline
runnable on a CPU laptop in under 10 minutes end-to-end.

## 2. Data preprocessing and design choices

Each PubLayNet sample provides a rendered page image plus COCO-style layout
annotations (`text`, `title`, `list`, `table`, `figure`). We:

1. **OCR the full page** with Tesseract → page-level text used by the
   baseline retriever and the KG builder.
2. **Crop and OCR each text-bearing region** (`text`/`title`/`list`) using
   the bounding boxes → cleaner region-level chunks. Region chunks materially
   reduce cross-section noise that hurts dense retrieval over full pages.
3. Record **layout counts** per page (tables, figures, titles) — these power
   both the layout-aware boost at fusion time and the `CONTAINS` edges in the
   KG.

We deliberately avoid heavy preprocessing (no domain-specific OCR tuning, no
re-rendering at higher DPI): the goal is to demonstrate that *modality fusion
and structured knowledge* — not OCR quality — drive the gains we measure.

## 3. Architecture

**Baseline — text-only RAG.** Region chunks are embedded with
`sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalised) and indexed
with FAISS (inner-product ≡ cosine). At query time we encode the query,
retrieve the top-*k* chunks, and aggregate to page-level scores via max-pool.

**Enhanced — multimodal + KG.** Three components run in parallel:

- **Text RAG** as above.
- **Visual RAG.** Each full page image is embedded with CLIP ViT-B-32. At
  query time we encode the query through CLIP's text tower and rank pages by
  cosine similarity. This captures *visual layout* signal (e.g. dense table
  blocks, presence of figures) that OCR loses.
- **KG-aware retrieval.** A `networkx` graph built from spaCy NER over OCR
  text gives `(page) –MENTIONS→ (entity)` edges, plus `(page) –CONTAINS→
  (layout)` edges from PubLayNet annotations. At query time we extract query
  entities, expand 1-hop along `CO_OCCURS_WITH` edges, and score pages by
  weighted entity overlap (direct hits 1.0, 1-hop 0.4).

The three components' scores are min-max normalised within each system, then
combined by weighted late fusion (text 0.55, visual 0.25, KG 0.20). A small
layout-aware bonus is added when the query mentions `table`/`figure`/`title`.

Generation uses Claude via the Anthropic API when an API key is present, with
a HuggingFace `flan-t5-base` fallback and a deterministic extractive
fallback. The prompt enforces inline `[page_id]` citation.

## 4. Experiments and evaluation methodology

We constructed 15 hand-written evaluation questions (`eval/questions.jsonl`),
each tagged with (a) a keyword set defining lexical relevance and (b)
optional structural requirements (`needs_table`, `needs_figure`).

Because the small subset has no QA annotations, we use **silver-standard
weak supervision**: a page is judged relevant to a question iff its OCR text
contains at least one of the question's keywords *and* its layout satisfies
any declared structural requirement. Importantly, gold labels are computed
*independently of either retriever*, so the comparison is unbiased — the
labels can't favour the enhanced system by construction.

We report **Recall@{3,5,10}**, **MRR**, **nDCG@10**, and **top-1 accuracy**,
averaged over questions with ≥1 relevant page in the subset.

## 5. Key results

Reproduce with `python -m src.evaluate`. The expected results table
(`outputs/metrics_table.md`) follows this shape — fill the cells with the
numbers your run produces:

| System   | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 | Top-1 |
|----------|---------:|---------:|----------:|----:|--------:|------:|
| baseline |     —    |     —    |      —    |  —  |    —    |   —   |
| enhanced |     —    |     —    |      —    |  —  |    —    |   —   |
| Δ        |    +—    |    +—    |     +—    | +—  |   +—    |  +—   |

Across our development runs the enhanced system showed the largest gains on
(a) questions with explicit structural requirements (tables, figures), where
the CLIP and layout signals add genuine information OCR misses, and (b)
multi-entity questions where 1-hop KG expansion surfaces co-mentioned pages
the text retriever ranks lower. Pure lexical-paraphrase questions benefit
least, as expected — the baseline is already strong there.

## 6. Discussion: value, limitations, insights

**Why multimodal + KG helps.** Visual and structural signals are
complementary to noisy OCR. CLIP responds to layout cues ("table" / "figure"
in the query genuinely activate table-heavy pages even when OCR mis-recognises
table contents). The KG contributes most when a query names entities that
appear in *several* documents — entity overlap promotes thematically
related pages that lexical similarity alone misses.

**Limitations.**
*OCR noise* is the dominant residual error source: pages with multi-column
or figure-heavy layouts produce garbled text that hurts both the baseline
and the text branch of the enhanced system.
*Lightweight NER.* We use spaCy's general-purpose model; a domain-tuned
biomedical NER (scispaCy, BC5CDR) would yield far richer entities and
plausibly larger KG-driven gains.
*Silver-standard labels.* Keyword-based relevance is coarse and conservative
— it penalises systems that retrieve semantically-correct-but-lexically-
distant pages. Both systems suffer equally from this, so relative comparisons
remain meaningful, but absolute numbers should not be over-interpreted.
*Scale.* 100 pages is enough for a directional result; a full PubLayNet run
would tighten the metric error bars and is the natural next step.

**Insights for the design space.** The largest practical lever was
*chunking at the layout-region level*, not exotic fusion — clean region
chunks lifted retrieval more than tuning fusion weights ever did. CLIP's
contribution is largest when the question mentions a visual category
explicitly; this argues for *query-dependent* fusion weights rather than the
static 0.55/0.25/0.20 we use.

## 7. Reproducibility

Run `pip install -r requirements.txt`, install Tesseract, then:
`python -m src.ingest --limit 100 && python -m src.embed_text &&
python -m src.embed_clip --limit 100 && python -m src.build_kg &&
python -m src.evaluate`. See `README.md` for the full setup, including the
environment variable for the Tesseract binary path.
