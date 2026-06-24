# Multimodal KG-RAG over PubLayNet

A retrieval-augmented generation (RAG) system for scientific document understanding using the Small PubLayNet dataset. The system combines text embeddings, CLIP visual embeddings, document layout information, and a lightweight knowledge graph to answer natural-language queries and provide explainable evidence.

The project includes a quantitative comparison between a baseline Text-RAG system and an enhanced Multimodal Knowledge Graph RAG system.

**Dataset:** `lhoestq/small-publaynet-wds` (HuggingFace)

---

## Architecture

```text
                   ┌──────────────────────────┐
   PubLayNet ───▶  │ ingest.py (OCR + layout) │ ──▶ ocr_pages.csv
                   └──────────────────────────┘
                               │
            ┌──────────────────┼──────────────────────┐
            ▼                  ▼                      ▼

   text embeddings     CLIP image embeds        spaCy NER
   (MiniLM + FAISS)    (ViT-B-32)               → networkx KG

            │                  │                      │
            └──────── fused retrieval ────────────────┘
                               │
                               ▼

                generate.py (answer synthesis
                     + cited evidence)

                               │
                               ▼

                     evaluate.py
                 (Recall@K, MRR, nDCG)
```

---

## Setup

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt

python -m spacy download en_core_web_sm
```

### Install Tesseract OCR

Ubuntu:

```bash
sudo apt-get install tesseract-ocr
```

macOS:

```bash
brew install tesseract
```

Windows:

Install from:

https://github.com/UB-Mannheim/tesseract/wiki

Add `tesseract.exe` to PATH or set the `TESSERACT_CMD` environment variable.

---

## Reproduce Results

### 1. OCR Extraction

```bash
python -m src.ingest --limit 100
```

### 2. Build Text Embeddings

```bash
python -m src.embed_text
```

### 3. Build CLIP Visual Embeddings

```bash
python -m src.embed_clip --limit 100
```

### 4. Build Knowledge Graph

```bash
python -m src.build_kg
```

### 5. Evaluate Baseline vs Enhanced System

```bash
python -m src.evaluate
```

### 6. Run Interactive Query

```bash
python -m src.run_query "Which pages discuss logistic regression results in tables?"
```

Generated outputs are stored in:

```text
outputs/
```

Evaluation results are stored in:

```text
outputs/metrics.json
outputs/metrics_table.md
```

---

## Example Query

**Query**

```text
Which pages discuss HIV testing and logistic regression results with tables?
```

**Example Retrieval Evidence**

```text
PMC4991227_00003

contains_table
mentions_hiv
mentions_regression
mentions_result
mentions_analysis
mentions_testing
```

The enhanced Multimodal KG-RAG system retrieves relevant scientific pages and provides graph-based explanations for retrieval decisions.

---

## Evaluation

### Baseline System

Text-only Retrieval-Augmented Generation

Components:

* OCR Text
* SentenceTransformer Embeddings
* FAISS Retrieval

### Enhanced System

Multimodal Knowledge Graph RAG

Components:

* OCR Text
* SentenceTransformer Embeddings
* CLIP Visual Embeddings
* Layout Metadata
* Knowledge Graph Reasoning
* Retrieval Fusion

Evaluation Metrics:

* Recall@K
* Mean Reciprocal Rank (MRR)
* Normalized Discounted Cumulative Gain (nDCG)

---

## Project Layout

```text
src/
  ingest.py
  embed_text.py
  embed_clip.py
  build_kg.py
  text_rag.py
  multimodal_rag.py
  kg_rag.py
  fusion.py
  generate.py
  evaluate.py
  run_query.py
  utils.py

eval/
  questions.jsonl

outputs/

report/
  report.md
```

---

## Notes and Limitations

* OCR over rendered scientific page images introduces recognition errors.
* The knowledge graph is intentionally lightweight and constructed using spaCy entity extraction and layout relations.
* The evaluation dataset contains a small set of benchmark queries.
* Visual retrieval operates at page level rather than figure or region level.

---

## Future Work

* Replace OCR with LayoutLMv3-based document understanding.
* Integrate SciSpaCy for biomedical entity extraction.
* Extend the knowledge graph using Neo4j.
* Scale experiments to the full PubLayNet dataset.
* Introduce figure-level and table-level visual retrieval.

---

## Author

**Dr.Sajjad Hussain**
Doctoral Researcher in Robotics and Artificial Intelligence
University of Brighton, United Kingdom
