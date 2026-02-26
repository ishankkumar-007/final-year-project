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

## Phase 4

- **Pure perturbation functions avoid side effects**: All perturbation functions (perturb_numerical, perturb_section, etc.) are pure -- they deep-copy the fact sheet and return new instances. This makes tree construction safe from aliasing bugs.
- **Pydantic model_dump for deep copy**: Use `fs.model_dump()` + `FactSheet.model_validate(data)` for reliable deep copies of Pydantic models rather than `copy.deepcopy()` which can struggle with Pydantic v2 internals.
- **Section adjacency requires normalised keys**: The adjacency map keys must match the exact normalised format in FactSheet.sections_cited. If NER span text is "Section 302 of IPC" but fact sheet stores "IPC-302", a normalisation step (`_find_matching_section`) is needed to bridge them.
- **Mock validator accepts all by default**: When no real LLM is available, the validator accepts all perturbations. This is correct for testing tree structure but means Phase 5+ must test with a real or more selective mock to verify rejection filtering.

## Phase 5

- **BFS expansion with displacement-based pruning**: expand_tree uses a deque frontier processed layer-by-layer. Nodes with mean_displacement below threshold are pruned, which prevents expanding irrelevant branches. With retriever=None all displacements are 0.0 so no pruning occurs -- confirmed that pruning logic works correctly with injected mock results.
- **TYPE_CHECKING guard for circular imports**: sensitivity.py imports PerturbationTree only for type hints. Using `TYPE_CHECKING` guard avoids circular import (perturbation_tree.py imports sensitivity.py for compute_diff in expand_tree).
- **Sensitivity scores need retrieval results**: All four dimension scores are 0.0 when retriever is None. With mock injected results, Numerical=0.70 while others=0.0 since only Numerical edges were created. This is correct behavior.
- **Streamlit session_state for tree persistence**: The tree must be stored in st.session_state to survive page switches. Using @st.cache_resource only for heavy objects like the retriever.
- **Streamlit sys.path fix required**: Streamlit runs scripts from its own process and does not inherit the project directory on sys.path. Must add `sys.path.insert(0, project_root)` at the top of the app file before any `countercase` imports, using `Path(__file__).resolve().parents[2]` to locate the project root.

## Phase 6

- **Arrow mixed types in DataFrames**: Streamlit/pyarrow rejects columns mixing int and str. When a rank column may have int values and "-" for missing, convert all values to `str()` consistently.
- **Path resolution with parents[]**: `Path(__file__).resolve().parents[0]` is the file's own directory, `parents[1]` is one level up. Easy to mix up — always verify by printing the resolved path.

## Phase 7

- **Evaluation data files must be created before running eval CLIs**: The eval harness, counterfactual eval, and explanation eval all load JSON/CSV input files that don't exist by default. Each needs a generation step first: test_set.json from pipeline queries, explanations.json from the retrieval+explanation pipeline, cf_eval_set.json with annotated fact sheets, and human_eval_filled.csv with Likert scores.
- **PerturbationTree API mismatch**: `PerturbationTree.__init__` takes `(retriever, top_k)`, not `root_fact_sheet`. Must call `build_root(fact_sheet)` then `expand_tree(validator=..., max_depth=..., max_children_per_node=...)` separately. The validator requires importing `PerturbationValidator` + `mock_validation_llm_fn` from `llm_validator`.
- **Faithfulness scoring requires metadata as source texts**: Template-based explanations reference fact sheet metadata (party types, sections) not chunk text. The source_texts list must include both chunk text AND metadata strings for word-overlap grounding to work. Without metadata sources, faithfulness is 0.0.
- **PowerShell f-string quoting**: Inline Python f-strings with dict access like `result["key"]` break in PowerShell due to quote escaping. Use `%` formatting or write to a script file instead.