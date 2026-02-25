# COUNTERCASE - Lessons Learned

## Phase 1

- **Conda env required**: The `torch2` conda environment must be activated before running any pipeline scripts. The base Python env is missing `tiktoken`, `langchain-text-splitters`, and other dependencies.
- **Nested f-string quoting**: When running Python one-liners via PowerShell, nested quotes in f-strings (e.g., `meta.get("key")`) cause `SyntaxError`. Use a script file instead of inline `-c` for complex tests.
- **DPR pooler weights warning**: The DPR models from HuggingFace emit an `UNEXPECTED` key warning for `pooler.dense.weight/bias`. This is benign and can be ignored -- the pooler is not used by DPR encoders.
- **ChromaDB cosine distance**: ChromaDB returns distances (0=identical, 2=opposite) not similarities. Must convert: `score = 1.0 - distance / 2.0`.
- **Metadata schema richness**: The ADX parquet metadata for 2024 includes: `title`, `petitioner`, `respondent`, `judge`, `author_judge`, `citation`, `case_id`, `cnr`, `decision_date`, `disposal_nature`, `court`, `path`, `year`, etc. Much richer than expected -- `author_judge` has 782 nulls, all other key fields are fully populated.

## Phase 2

- **ChromaDB embeddings are numpy arrays**: `result.get("embeddings") or []` triggers `ValueError: The truth value of an array with more than one element is ambiguous`. Must check `result.get("ids")` first and access embeddings without `or` fallback: use `emb_raw = result.get("embeddings") if result else None` then check `emb_raw is not None`.
- **Cross-encoder lazy loading matters**: First call to `CrossEncoderReranker.rerank()` takes ~25s (model download + load). Subsequent calls take <1s. The pass-through mode (`model_name="none"`) correctly bypasses loading.
- **Cross-encoder scores are unbounded**: MS-MARCO MiniLM cross-encoder outputs raw logits, not probabilities. Scores range from -10 to +4 in practice. Higher = more relevant. Do not treat as [0,1].
- **DPR index must be explicitly saved in Phase 1**: The Phase 1 smoke test indexes only into ChromaDB (which auto-persists) but didn't call `dpr.save()`. Fixed by adding auto-load in `DualIndex.__init__()` — if `dpr.faiss` exists on disk, it's loaded automatically (mirroring ChromaDB's `PersistentClient` behavior).- **Deduplicate ranked case_ids before computing IR metrics**: Multiple chunks from the same case produce duplicate case_ids in ranked results. Without deduplication, NDCG and Recall exceed 1.0 (e.g., NDCG=2.13, Recall=3.33). Standard IR eval counts each document once at its highest rank position. Always deduplicate (preserving first-occurrence order) before passing to MRR/NDCG/Recall functions.

## Phase 3

- **extract_directory processes ALL PDFs**: The Phase 1 `extract_directory` function loads every PDF in a folder into memory before returning. For Phase 3 pipeline with `--max-cases 10`, this wastes time processing 780+ PDFs. Solution: iterate per-PDF with `extract_pdf` and `break` when enough cases are collected.
- **Regex monetary amount empty group**: Pattern `Rs\.?\s*([\d,]+)` can match `Rs,` where group(1) is empty string. Always check `if not raw: continue` before `float()` conversion.
- **Mock LLM is sufficient for testing**: The regex-based mock_llm_fn produces valid Pydantic-validated FactSheet objects by extracting IPC sections, evidence types, monetary amounts, and ages directly from text. This lets the full pipeline run without any LLM dependency.
- **Section locator 3-strategy design**: Priority order: (1) pre-detected sections from Phase 1, (2) heading-pattern heuristic, (3) first 20% fallback. All 10 test judgments had a detected Facts heading, yielding 100% detection rate on Supreme Court of India judgments.