# Multimodal KG-RAG over PubLayNet — Technical Report

## 1. Problem Formulation

This project presents a Multimodal Knowledge Graph Retrieval-Augmented Generation (KG-RAG) system for scientific document retrieval using the PubLayNet dataset. The objective is to answer natural-language questions over scientific documents by combining textual information, visual document content, and structured knowledge graph evidence.

Traditional retrieval systems rely primarily on text and often ignore visual information such as tables, figures, charts, and document layout. Scientific documents contain important information distributed across multiple modalities; therefore, integrating visual and structural information has the potential to improve retrieval quality and explainability.

The project compares a baseline text-only retrieval system against an enhanced multimodal retrieval framework that combines OCR text, CLIP visual embeddings, and knowledge graph reasoning.

A subset of 100 document pages from the Small PubLayNet dataset was used to demonstrate the complete pipeline while maintaining practical execution time on a standard CPU-based system.

---

## 2. Data Preprocessing and Design Choices

Each PubLayNet sample contains a rendered document page image together with layout annotations describing text blocks, titles, figures, tables, and lists.

The preprocessing pipeline consists of three stages:

1. Full-page OCR extraction using Tesseract OCR.
2. Region-level OCR extraction using PubLayNet bounding boxes.
3. Layout metadata extraction including counts of tables, figures, and titles.

Full-page OCR provides page-level text used by both the baseline retriever and the knowledge graph. Region-level OCR generates cleaner retrieval chunks and reduces noise introduced by unrelated document sections.

The ingestion process successfully produced:

* 100 document pages
* 713 document regions
* OCR text content
* Layout metadata

The design intentionally avoids complex OCR optimisation or domain-specific preprocessing. The focus of the work is to evaluate the contribution of multimodal fusion and structured knowledge rather than OCR quality alone.

---

## 3. System Architecture

### 3.1 Baseline Text-RAG

The baseline system uses only textual information.

OCR text chunks are embedded using:

* SentenceTransformer all-MiniLM-L6-v2
* 384-dimensional embeddings

The embeddings are indexed using FAISS and retrieved using cosine similarity.

At query time:

1. Query embedding is generated.
2. Similar chunks are retrieved.
3. Page-level scores are aggregated.

### 3.2 Visual Retrieval

The visual retrieval module uses CLIP ViT-B/32.

Each page image is encoded into a 512-dimensional visual embedding.

At query time:

1. The query is encoded using the CLIP text encoder.
2. Visual similarity is computed.
3. Pages are ranked according to image-text similarity.

This component captures visual information such as:

* Tables
* Figures
* Charts
* Document layout

which may not be fully represented in OCR text.

### 3.3 Knowledge Graph Retrieval

A lightweight knowledge graph is constructed using:

* spaCy Named Entity Recognition
* NetworkX

The graph contains:

* Page nodes
* Entity nodes
* Layout nodes

Relationships include:

* MENTIONS
* CONTAINS
* HAS_TABLE
* HAS_TITLE
* HAS_FIGURE

The final graph contains:

* 2154 nodes
* 2665 edges

Knowledge graph retrieval ranks pages according to entity overlap between the query and graph entities.

### 3.4 Multimodal Fusion

The final retrieval score combines text, visual, and graph scores using weighted late fusion:

[
Score = w_tT + w_vV + w_kK
]

where:

* (T) = Text retrieval score
* (V) = Visual retrieval score
* (K) = Knowledge graph score

The weights used in this project are:

* Text: 0.55
* Visual: 0.25
* Knowledge Graph: 0.20

A small layout-aware bonus is added when queries explicitly mention tables, figures, or titles.

### 3.5 Answer Generation

Answer generation is implemented using an extractive approach.

The highest-ranked retrieved passages are combined and returned together with page-level citations. This ensures that generated answers remain grounded in retrieved evidence and avoids hallucinated information.

The architecture can be extended in future work with large language models for abstractive answer generation.

---

## 4. Experimental Evaluation

A set of 15 manually designed evaluation questions was created.

The questions cover:

* Logistic regression analysis
* Clinical results
* HIV testing
* Statistical significance
* Tables
* Figures
* Charts
* Demographic information
* Patient outcomes

Because the Small PubLayNet subset does not provide question-answer annotations, a weakly supervised evaluation strategy was adopted.

A page is considered relevant when:

1. OCR text contains one or more query keywords.
2. Layout requirements are satisfied (table or figure when required).

The following retrieval metrics were used:

* Recall@3
* Recall@5
* Recall@10
* Mean Reciprocal Rank (MRR)
* nDCG@10
* Top-1 Accuracy

---

## 5. Experimental Results

The quantitative evaluation results are shown below.

| System            | Recall@3 | Recall@5 | Recall@10 |    MRR | nDCG@10 | Top-1 Accuracy |
| ----------------- | -------: | -------: | --------: | -----: | ------: | -------------: |
| Baseline Text-RAG |    0.061 |    0.074 |     0.143 |  0.507 |   0.351 |          0.333 |
| Enhanced KG-RAG   |    0.131 |    0.151 |     0.206 |  0.578 |   0.397 |          0.400 |
| Improvement       |   +0.070 |   +0.077 |    +0.063 | +0.071 |  +0.047 |         +0.067 |

The enhanced multimodal system consistently outperformed the text-only baseline across all evaluation metrics.

Key observations include:

* Recall@5 increased from 0.074 to 0.151.
* MRR increased from 0.507 to 0.578.
* Top-1 Accuracy improved from 0.333 to 0.400.
* Visual retrieval contributed strongly for figure and chart related queries.

For visual queries such as:

"Find pages containing figures or bar charts"

CLIP produced visual similarity scores close to 1.0, demonstrating successful visual retrieval.

---

## 6. Discussion

### Why Multimodal Retrieval Helps

Scientific documents contain information beyond text. Tables, figures, and page structure provide important contextual information that traditional text retrieval systems cannot fully exploit.

The CLIP retrieval module captures visual patterns such as:

* Table-heavy pages
* Figure-rich pages
* Structured layouts

even when OCR text quality is poor.

### Why Knowledge Graphs Help

The knowledge graph improves explainability and retrieval quality.

Instead of retrieving pages solely through semantic similarity, the graph enables retrieval through:

* Entity overlap
* Structural relationships
* Shared document concepts

This provides additional reasoning capability and improves transparency.

### Limitations

The main limitation is OCR noise.

Because OCR operates on rendered document images, some extracted text contains recognition errors. These errors affect:

* Text retrieval
* Entity extraction
* Knowledge graph quality

A second limitation is the use of general-purpose spaCy NER rather than domain-specific biomedical entity extraction.

Finally, the evaluation uses weak supervision based on keywords rather than manually labelled relevance judgements.

---

## 7. Conclusion

This project demonstrates a complete Multimodal KG-RAG framework for scientific document retrieval using the PubLayNet dataset.

The proposed system combines:

* OCR text retrieval
* CLIP visual retrieval
* Knowledge graph reasoning
* Late fusion ranking

Experimental results show consistent improvements over a text-only baseline across Recall, MRR, nDCG, and Top-1 Accuracy.

The findings indicate that multimodal retrieval and structured knowledge integration can significantly improve both retrieval effectiveness and explainability for scientific document search systems.

---

## 8. Future Work

Future improvements include:

* Integration of stronger OCR models.
* Use of biomedical NER models such as SciSpaCy.
* Figure-level and table-level retrieval.
* Query-dependent fusion weighting.
* Deployment on the complete PubLayNet dataset.
* Integration with graph databases such as Neo4j.

---

## 9. References

[1] Lewis, P. et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” NeurIPS, 2020.

[2] Radford, A. et al., “Learning Transferable Visual Models From Natural Language Supervision,” ICML, 2021.

[3] Johnson, J., Douze, M., and Jégou, H., “Billion-Scale Similarity Search with GPUs,” IEEE Transactions on Big Data, 2021.

[4] Zhong, X., Tang, J., and Yepes, A. J., “PubLayNet: Largest Dataset Ever for Document Layout Analysis,” ICDAR, 2019.

[5] Reimers, N. and Gurevych, I., “Sentence-BERT: Sentence Embeddings using Siamese BERT Networks,” EMNLP, 2019.

[6] Smith, R., “An Overview of the Tesseract OCR Engine,” ICDAR, 2007.

[7] Honnibal, M. and Montani, I., “spaCy: Industrial-Strength Natural Language Processing in Python,” 2023.

[8] Hagberg, A., Swart, P., and Chult, D., “Exploring Network Structure, Dynamics, and Function Using NetworkX,” SciPy Conference, 2008.

---

### Source Code Repository

https://github.com/sh1755/multimodal-kg-rag-publaynet
