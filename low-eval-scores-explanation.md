# Why Retrieval Evaluation Scores Are Low

## Observed Problem

All retrieval evaluation metrics (MRR@10, NDCG@10, Recall@K) are very small across **all** ablation modes (full_system, dpr_only, chroma_only, hybrid_no_mmr, hybrid_no_reranker).

## Root Cause: Citation Ground Truth ≠ Semantic Similarity

The test set was built automatically by extracting **citations** from judgments — if Case A cites Case B, B is marked "relevant to A". But **citation relevance and semantic retrieval relevance are fundamentally different things**.

A real retrieval test confirms the mismatch:

| | Count |
|---|---|
| Relevant (cited) cases for query | 11 |
| Retrieved by semantic search | 10 |
| **Overlap** | **1** |

### Why Citation ≠ Retrieval

- A murder case might cite a procedural case about evidence admissibility — they share zero semantic content
- A property dispute might cite a constitutional case about fundamental rights — topically unrelated
- DPR finds cases with similar *language and facts*; citations link cases by *legal reasoning chains*

## Contributing Factors

### 1. Query Text Is Raw Judgment Text

The test set uses the first 2000 chars of the judgment as `query_text`, which includes case numbers, judge names, dates, and procedural boilerplate. This is noisy input for a semantic encoder.

### 2. DPR Domain Mismatch

The DPR model (`facebook/dpr-question_encoder-single-nq-base`) was trained on Wikipedia Q&A, not Indian legal text. It doesn't understand legal vocabulary well.

### 3. Small Test Set

Only 24 queries with 26 unique relevant IDs. Each miss disproportionately tanks the average.

### 4. All Modes Are Equally Affected

Scores are small across all ablation modes because the problem isn't which retrieval pipeline you use (DPR vs ChromaDB vs hybrid) — it's that the *definition of relevance* in the test set doesn't match what any embedding-based retriever can find.

## This Is Expected in Legal IR

This is a known challenge in legal information retrieval research. Citation-based ground truth is the standard approach (because manual annotation is expensive), but it systematically underestimates semantic retrieval quality. The system may be retrieving *actually useful* cases that just aren't among the cited ones.

## What Would Improve Scores

| Fix | Impact | Effort |
|-----|--------|--------|
| **Manual annotation** of relevant cases | High — ground truth matches task | Very high |
| **Fine-tune DPR** on Indian legal corpus | High — better embeddings | Medium |
| **Use judgment facts section** as query text instead of raw first-2000-chars | Medium — cleaner queries | Low |
| **Index more years** (currently only 2024) | Medium — larger candidate pool | Low (just time) |
| **Hybrid ground truth** (citations + same legal topic/sections) | Medium — broader relevance | Medium |

## Conclusion

The low scores don't mean the system is broken — they mean the evaluation's definition of "correct answer" doesn't align with what semantic retrieval produces. When tested manually with natural-language queries (e.g., "criminal appeal IPC 302 murder"), the system returns relevant criminal cases with matching IPC sections and appropriate fact patterns.
