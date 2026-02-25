# COUNTERCASE Development Workflow -- Copilot Agent Prompts

This document contains sequential prompts to give to a Copilot agent to build the COUNTERCASE system phase by phase. Each phase is a self-contained prompt. Complete one phase before starting the next. Phases build on the outputs of prior phases.

Read `plan.md` in this repository before starting any phase. It is the authoritative design document.

---

## Workspace Conventions

All source code lives under `countercase/` at the repository root. The directory structure is:

```
countercase/
    config/
        settings.py
    ingestion/
        __init__.py
        pdf_extractor.py
        metadata_extractor.py
        section_detector.py
        noise_filter.py
    preprocessing/
        __init__.py
        chunker.py
        section_tagger.py
    indexing/
        __init__.py
        dpr_index.py
        chroma_index.py
        dual_index.py
    retrieval/
        __init__.py
        rrf.py
        mmr.py
        reranker.py
        hybrid_retriever.py
    fact_extraction/
        __init__.py
        section_locator.py
        fact_sheet_extractor.py
        schema.py
        ner_tagger.py
    counterfactual/
        __init__.py
        perturbation_rules.py
        section_adjacency.py
        llm_validator.py
        perturbation_tree.py
        sensitivity.py
    explanation/
        __init__.py
        per_result.py
        counterfactual_summary.py
        output_formatter.py
    evaluation/
        __init__.py
        metrics.py
        eval_harness.py
    app/
        __init__.py
        streamlit_app.py
    tests/
        ...
```

Python version: 3.10+. Package manager: pip with a `requirements.txt`. No emojis in code, comments, docstrings, or outputs. All modules must include type hints. Use Google-style docstrings.

The judgment data is already available locally under `judgments-data/`. The structure is:

```
judgments-data/
    data/
        tar/
            year={YYYY}/
                english/
                    english.tar        <-- contains judgment PDFs
                    english.index.json <-- index of contents
    metadata/
        parquet/
            year={YYYY}/
                metadata.parquet      <-- per-year metadata
        tar/
            year={YYYY}/
                metadata.tar
                metadata.index.json
```

The tar files under `data/tar/year={YYYY}/english/` contain the judgment PDFs. The parquet files under `metadata/parquet/year={YYYY}/` contain per-case metadata. Use the `extract-judgments.ps1` script to extract tar files before processing. The metadata parquet files are directly readable with pandas.

---

## Phase 1 -- Data Infrastructure and Baseline Retrieval

### Prompt

You are building Phase 1 of the COUNTERCASE project, a legal case retrieval system for Indian Supreme Court judgments. Read `plan.md` for the full architectural specification.

Phase 1 deliverables:

**1.1 -- Project scaffolding.** Create the directory structure under `countercase/` as specified in `workflow.md` (the workspace conventions section). Create `requirements.txt` with all dependencies needed for Phase 1: `pdfplumber`, `chromadb`, `sentence-transformers`, `transformers`, `langchain`, `langchain-text-splitters`, `tiktoken`, `pandas`, `pyarrow`, `tqdm`, `faiss-cpu`. Create `countercase/config/settings.py` with a configuration dataclass holding all paths and parameters: `DATA_DIR` pointing to `judgments-data/data/tar/`, `METADATA_DIR` pointing to `judgments-data/metadata/parquet/`, `CHUNK_SIZE` (1024), `CHUNK_OVERLAP` (128), `EMBEDDING_MODEL` (`all-MiniLM-L6-v2`), `DPR_QUESTION_MODEL` (`facebook/dpr-question_encoder-single-nq-base`), `DPR_CONTEXT_MODEL` (`facebook/dpr-ctx_encoder-single-nq-base`), `CHROMA_PERSIST_DIR`, `DPR_INDEX_DIR`, `TOP_K` (50), `RRF_K` (60).

**1.2 -- Metadata ingestion.** Write `countercase/ingestion/metadata_extractor.py`. This module reads parquet files from `judgments-data/metadata/parquet/year={YYYY}/metadata.parquet` for a configurable year range. Load each parquet file with pandas. Inspect the columns and log them. Build a unified metadata DataFrame indexed by a case identifier. The parquet files contain whatever metadata the ADX dataset provides natively. Write a function `load_metadata(start_year: int, end_year: int) -> pd.DataFrame` that returns the combined DataFrame. Write a function `inspect_metadata_schema(start_year: int, end_year: int) -> dict` that returns column names, dtypes, null counts, and sample values for each column, so we can understand what is natively available versus what must be extracted from PDFs. Save inspection results to `countercase/data/metadata_inspection.json`.

**1.3 -- PDF text extraction.** Write `countercase/ingestion/pdf_extractor.py`. This module extracts text from judgment PDFs using `pdfplumber`. The input is a path to a directory of PDFs (extracted from the tar files). The output is a list of `PageText` objects, each containing `pdf_path`, `page_number` (1-indexed), and `text`. Handle multi-column layouts by sorting text blocks (words) by their top coordinate first, then left coordinate, to reconstruct reading order. Detect and remove headers and footers: any text block whose top is in the top 5% of the page or whose bottom is in the bottom 5% of the page is classified as boilerplate and removed. Write a function `extract_pdf(pdf_path: str) -> list[PageText]` for a single PDF and `extract_directory(dir_path: str) -> dict[str, list[PageText]]` for batch processing.

**1.4 -- Noise filtering.** Write `countercase/ingestion/noise_filter.py`. Implement cause-list noise removal: detect text matching patterns like "Diary No.", "ITEM NO.", "COURT NO.", or "Reportable" / "Non-Reportable" headers at the start of the document. Strip these. Also implement a function to collapse excessive whitespace and normalize unicode characters (replace curly quotes, em-dashes, non-breaking spaces with their ASCII equivalents).

**1.5 -- Section detection.** Write `countercase/ingestion/section_detector.py`. Implement a function `detect_sections(full_text: str) -> list[Section]` where `Section` is a dataclass with `section_type` (enum: Facts, Issues, Submissions, Analysis, Held, Ratio, Obiter, Unknown), `start_char`, `end_char`, and `text`. Use regex patterns to match headings like "FACTS", "FACTUAL BACKGROUND", "STATEMENT OF THE CASE", "ISSUES", "SUBMISSIONS", "ANALYSIS", "HELD", "ORDER", "RATIO DECIDENDI", "OBITER DICTA". Match all-caps headings, numbered headings ("I. FACTS"), and bold-marked headings. If no headings are found, classify the first 20% of text as Facts and the remainder as Unknown.

**1.6 -- Chunking.** Write `countercase/preprocessing/chunker.py`. Use LangChain `RecursiveCharacterTextSplitter` with `chunk_size=1024` tokens, `chunk_overlap=128` tokens, and custom separators: `["\n\n", "\n", ". ", " ", ""]` with legal-aware separators inserted before the character fallback (patterns like `r"\n\d+\.\s"`, `r"\n\([a-z]\)\s"`, `r"\n\([ivx]+\)\s"`). Use `tiktoken` with `cl100k_base` for token counting. Each output chunk is a dataclass `Chunk` with `chunk_id`, `text`, `source_pdf`, `page_number`, `section_type`, `char_start`, `char_end`. The `chunk_id` is `{case_id}_chunk_{zero_padded_index}`. Write `countercase/preprocessing/section_tagger.py` that tags each chunk with the section type from the section detector, using majority overlap.

**1.7 -- Dual indexing.** Write `countercase/indexing/dpr_index.py` implementing a DPR (Dense Passage Retrieval) index wrapper. Use the `facebook/dpr-question_encoder-single-nq-base` model for encoding queries and `facebook/dpr-ctx_encoder-single-nq-base` for encoding passages, loaded via `transformers` (`DPRQuestionEncoder`, `DPRQuestionEncoderTokenizer`, `DPRContextEncoder`, `DPRContextEncoderTokenizer`). On indexing, encode all chunks with the context encoder and build a FAISS `IndexFlatIP` (inner product) index over the resulting embeddings. Store a mapping from FAISS internal index to `chunk_id`. Implement `query(query_text: str, top_k: int) -> list[tuple[str, float]]` that encodes the query with the question encoder and searches the FAISS index, returning `(chunk_id, score)` pairs. Serialize the FAISS index and mappings to disk with `faiss.write_index` and pickle respectively. The DPR models must be lazy-loaded (loaded on first call, not at import time) to avoid slow startup. Write `countercase/indexing/chroma_index.py` implementing a ChromaDB wrapper. The ChromaDB collection stores embeddings from `all-MiniLM-L6-v2` with metadata fields: `year` (int), `bench_type` (str), `act_sections` (str, comma-separated), `section_type` (str), `outcome_label` (str), `source_pdf` (str), `page_number` (int), `case_id` (str). Support `query(query_text: str, top_k: int, where: dict | None) -> list[tuple[str, float]]` with optional metadata pre-filtering via the `where` parameter. Write `countercase/indexing/dual_index.py` that combines both indexes and exposes a `query(query_text: str, top_k: int, metadata_filters: dict | None) -> tuple[list, list]` returning the DPR and ChromaDB result lists separately.

**1.8 -- RRF fusion.** Write `countercase/retrieval/rrf.py`. Implement Reciprocal Rank Fusion: `rrf_fuse(ranked_lists: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]`. For each chunk appearing in any list, compute `score = sum(1 / (k + rank))` where rank is 1-indexed. Return the merged list sorted by RRF score descending.

**1.9 -- End-to-end pipeline script.** Write `countercase/pipeline_phase1.py` that ties everything together: given a year range, extracts PDFs, processes text, chunks, indexes into both DPR and ChromaDB, and runs a sample query with RRF fusion to verify the pipeline works. Print the top 10 results with their chunk text, source PDF, page number, and RRF score. This is the smoke test for Phase 1.

**1.10 -- Metadata-driven extraction.** Write `countercase/ingestion/metadata_extractor.py` to also include a function `extract_metadata_from_text(full_text: str) -> dict` that extracts the following from judgment text using regex: `case_id` (pattern: case type + number + / + year, like "Criminal Appeal 1234/2015"), `year` (4-digit year from the date line), `act_sections` (all patterns matching "Section \d+ \w+" or "Article \d+"), `judge_names` (names following "Hon'ble" or "Justice" or "J." patterns), `bench_type` (count of judge names: 1=Single, 2=Division, 3=Three-Judge, 5+=Constitution), `outcome_label` (search final 20% of text for "appeal allowed", "appeal dismissed", "petition disposed", classify accordingly).

Technical constraints: no emojis anywhere. All functions must have type hints and Google-style docstrings. Handle errors gracefully with logging, never crash on a single bad PDF. Use `logging` module, not print statements, for all diagnostic output.

### Verification

After completing Phase 1, verify by running `pipeline_phase1.py` on a small year range (e.g., 2024-2025). Confirm:
- PDFs are extracted and text is readable
- Metadata inspection JSON is generated and shows column schema
- Chunks are generated with correct section types
- DPR and ChromaDB indexes are populated
- RRF fusion returns ranked results for a sample query
- Each result has source_pdf, page_number, and section_type attached

---

## Phase 2 -- Retrieval Enhancements

### Prompt

You are building Phase 2 of the COUNTERCASE project. Phase 1 is complete: PDF extraction, chunking, dual DPR+ChromaDB indexing, and RRF fusion are working. Read `plan.md` sections 5 and 6 for the full retrieval architecture.

Phase 2 deliverables:

**2.1 -- MMR implementation.** Write `countercase/retrieval/mmr.py`. Implement Maximal Marginal Relevance selection: `mmr_select(candidates: list[tuple[str, float]], embeddings: dict[str, list[float]], top_k: int, lambda_mult: float = 0.6) -> list[tuple[str, float]]`. The function takes the RRF-fused candidate list with scores, a dictionary mapping chunk_id to its embedding vector, the number of results to return, and the lambda parameter. At each step, select the candidate that maximizes `lambda_mult * relevance_score - (1 - lambda_mult) * max_cosine_similarity_to_already_selected`. Return the selected list in order. Use numpy for cosine similarity computation. The `embeddings` dict is populated from ChromaDB at query time.

**2.2 -- Cross-encoder re-ranking.** Write `countercase/retrieval/reranker.py`. Implement a re-ranker wrapper class `CrossEncoderReranker` with a configurable model name (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`). The class loads the cross-encoder from `sentence_transformers.CrossEncoder` on init. Implement `rerank(query: str, candidates: list[tuple[str, str]], top_k: int) -> list[tuple[str, float]]` where candidates are `(chunk_id, chunk_text)` pairs. The re-ranker scores each (query, chunk_text) pair and returns the top_k sorted by cross-encoder score. The class must be swappable: accept any model name at init, and provide a no-op pass-through mode that returns candidates in their original order (for when re-ranking should be disabled).

**2.3 -- Hybrid retriever.** Write `countercase/retrieval/hybrid_retriever.py`. This is the main retrieval interface. Class `HybridRetriever` composes the dual index, RRF fusion, MMR, and cross-encoder re-ranker into a single `retrieve(query: str, top_k: int = 10, metadata_filters: dict | None = None, lambda_mult: float = 0.6, use_reranker: bool = True) -> list[RetrievalResult]` method. `RetrievalResult` is a dataclass with: `chunk_id`, `text`, `source_pdf`, `page_number`, `section_type`, `case_id`, `year`, `bench_type`, `act_sections`, `outcome_label`, `rrf_score`, `mmr_score`, `reranker_score`, `final_rank`. The method executes the six-stage pipeline: (1) metadata pre-filter, (2) DPR + ChromaDB ANN search with top_k=50 each, (3) RRF fusion, (4) MMR selection with the given lambda, (5) cross-encoder re-ranking if enabled, (6) source attribution metadata attached. Log timing for each stage.

**2.4 -- Evaluation harness.** Write `countercase/evaluation/metrics.py` with functions: `mrr_at_k(ranked_results: list[str], relevant_ids: set[str], k: int) -> float`, `ndcg_at_k(ranked_results: list[str], relevant_ids: set[str], k: int) -> float`, `recall_at_k(ranked_results: list[str], relevant_ids: set[str], k: int) -> float`. Write `countercase/evaluation/eval_harness.py` that runs evaluation over a test set. The test set format is a JSON file where each entry has `query_case_id`, `query_text`, and `relevant_case_ids` (list of case IDs judged relevant). The harness runs each query through the `HybridRetriever`, collects the ranked result case IDs, computes metrics, and writes results to a JSON report. Include ablation modes: DPR-only, ChromaDB-only, hybrid-no-MMR, hybrid-no-reranker, full-system.

**2.5 -- Update requirements.txt.** Add `numpy` and `scipy` if not already present (for cosine similarity in MMR and evaluation metrics).

**2.6 -- Pipeline script.** Write `countercase/pipeline_phase2.py` that demonstrates the full retrieval pipeline: load the indexes from Phase 1, run a sample query through all six stages, print each stage's timing and output, and show the final ranked results with all metadata fields. Also run a comparison between DPR-only, ChromaDB-only, and full hybrid to show the difference in results.

Technical constraints: same as Phase 1. The cross-encoder model must be lazy-loaded (loaded on first call, not at import time) to avoid slow startup. MMR must handle the case where the embeddings dict does not contain a candidate chunk_id (skip that candidate with a warning). All retrieval stages must be independently testable.

### Verification

After completing Phase 2, verify:
- MMR produces diverse results (no two consecutive chunks from the same case and section)
- Cross-encoder re-ranking changes the order of results compared to RRF-only
- `pipeline_phase2.py` prints timing per stage and full results with metadata
- Evaluation harness runs on a small test set (even if hand-constructed with 5 queries) and produces MRR, NDCG, Recall numbers
- Ablation modes produce different results (DPR-only should differ from ChromaDB-only)

---

## Phase 3 -- Fact Sheet Extraction and NER

### Prompt

You are building Phase 3 of the COUNTERCASE project. Phases 1-2 are complete: the full hybrid retrieval pipeline with RRF, MMR, cross-encoder re-ranking, and evaluation harness are working. Read `plan.md` sections 4 and 7 (Stage 1 of the perturbation pipeline) for the fact sheet extraction and NER specifications.

Phase 3 deliverables:

**3.1 -- Fact sheet schema.** Write `countercase/fact_extraction/schema.py`. Define Pydantic models for the fact sheet:

```python
class PartyInfo(BaseModel):
    petitioner_type: str | None  # closed vocab: Individual, Corporation, State, UnionOfIndia, Minor, etc.
    respondent_type: str | None

class EvidenceItem(BaseModel):
    evidence_type: str  # DyingDeclaration, Confession, MedicalReport, Document, Witness, FIR, etc.
    description: str

class NumericalFacts(BaseModel):
    amounts: list[dict]    # each: {"value": float, "unit": str, "context": str}
    ages: list[dict]       # each: {"value": int, "descriptor": str}
    durations: list[dict]  # each: {"value": float, "unit": str, "context": str}

class FactSheet(BaseModel):
    case_id: str
    parties: PartyInfo
    evidence_items: list[EvidenceItem]
    sections_cited: list[str]  # normalized: "IPC-302", "Constitution-Article-21"
    numerical_facts: NumericalFacts
    outcome: str | None
```

Include validation: sections_cited must match a regex pattern. Party types must be from a closed vocabulary list defined as a constant.

**3.2 -- Section locator.** Write `countercase/fact_extraction/section_locator.py`. Implement `locate_facts_section(full_text: str, sections: list[Section] | None = None) -> str`. If sections are provided (from the section detector in Phase 1), return the text of the section with type Facts. If no Facts section exists, use the two-stage heuristic from plan.md: search for heading patterns, extract the text between the facts heading and the next heading (Issues or Submissions). Fallback: if no headings are found, return the first 20% of the judgment text. The function returns the raw text of the facts section.

**3.3 -- LLM-based fact sheet extraction.** Write `countercase/fact_extraction/fact_sheet_extractor.py`. Implement class `FactSheetExtractor` that takes a facts section text and produces a `FactSheet` object. The extractor uses an LLM to populate the schema. Design it with a pluggable LLM backend: accept a callable `llm_fn(prompt: str) -> str` at init. Provide two implementations in the same file: (a) `local_llm_fn` that calls a local Mistral-7B model via `llama-cpp-python` or `transformers`, and (b) `api_llm_fn` that calls an OpenAI-compatible API endpoint (configurable URL and API key from env vars). The extraction prompt is:

```
You are a legal analyst specializing in Indian Supreme Court cases. Extract a structured fact sheet from the following case facts section. Return ONLY valid JSON matching this exact schema, with no additional text:

{schema_json}

Rules:
- Only include information explicitly stated in the text.
- For sections_cited, normalize to format "ACT-Section" (e.g., "IPC-302", "Constitution-Article-21").
- For party types, use one of: Individual, Corporation, State, UnionOfIndia, Minor, Partnership, ForeignNational, PublicSector, PrivateSector, Employee, Contractor, Tenant, Licensee, Unknown.
- If a field is not present in the text, use null or an empty list.

Case facts:
{facts_text}
```

Parse the LLM output as JSON, validate against the Pydantic schema. If parsing fails, retry once with an example-augmented prompt that includes a filled-out example fact sheet. If it fails again, return None and log the failure.

**3.4 -- NER tagger for perturbation candidates.** Write `countercase/fact_extraction/ner_tagger.py`. Implement `tag_perturbation_candidates(fact_text: str) -> list[TaggedSpan]` where `TaggedSpan` has `text`, `start`, `end`, `entity_type` (enum: MONETARY_AMOUNT, AGE, DURATION, LEGAL_SECTION, PARTY_ROLE, EVIDENCE_TYPE). Start with a rule-based NER implementation using regex patterns:
- MONETARY_AMOUNT: patterns like "Rs.", "rupees", followed by numbers, "lakh", "crore"
- AGE: patterns like "aged \d+", "\d+ years old", "minor", "major"
- DURATION: patterns like "\d+ years", "\d+ months", "\d+ days", "period of"
- LEGAL_SECTION: patterns like "Section \d+", "S. \d+", "Article \d+", "IPC \d+", "CPC \d+"
- PARTY_ROLE: patterns matching the closed party type vocabulary
- EVIDENCE_TYPE: patterns matching "dying declaration", "confession", "FIR", "post-mortem", "medical report", "witness"

Design this as a swappable component: the interface is the same whether using regex or a fine-tuned BERT NER model. Later phases can swap in a trained model.

**3.5 -- Fact sheet database.** Write a simple JSON-file-based storage in `countercase/fact_extraction/fact_store.py`: `save_fact_sheet(case_id: str, fact_sheet: FactSheet)`, `load_fact_sheet(case_id: str) -> FactSheet | None`, `list_fact_sheets() -> list[str]`. Store each fact sheet as `{FACT_STORE_DIR}/{case_id}.json`.

**3.6 -- Pipeline script.** Write `countercase/pipeline_phase3.py`. Given a year range (small, like 2023-2024), extract PDFs, detect sections, locate the facts section, run the LLM-based fact sheet extractor on 10 sample cases, run the NER tagger on the facts text, and print: the extracted fact sheet JSON, the tagged spans with entity types, and any extraction failures. Also store the successful fact sheets in the fact store.

**3.7 -- Update requirements.txt.** Add `pydantic`, `llama-cpp-python` (optional, for local LLM), `openai` (for API calls), `requests`.

Technical constraints: the LLM call must have a configurable timeout (default 60 seconds). If the LLM is unavailable, the pipeline must continue and log the failure rather than crash. The NER tagger must not import any ML model at module level; it should work with pure regex by default.

### Verification

After completing Phase 3, verify:
- Pydantic schema validates correctly on hand-written example fact sheets
- Section locator finds the facts section in at least 80% of test cases from 2023-2024
- LLM extractor produces valid FactSheet JSON for at least 5 of 10 sample cases (depends on LLM availability)
- NER tagger finds LEGAL_SECTION spans in cases that cite IPC/CPC sections
- Fact sheets are saved to and loaded from the fact store correctly

---

## Phase 4 -- Perturbation Logic and Single-Level Tree

### Prompt

You are building Phase 4 of the COUNTERCASE project. Phases 1-3 are complete: hybrid retrieval pipeline, fact sheet extraction with LLM, and NER tagging are working. Read `plan.md` section 7 for the full counterfactual reasoning module specification.

Phase 4 deliverables:

**4.1 -- Perturbation rules.** Write `countercase/counterfactual/perturbation_rules.py`. Implement perturbation functions for each fact type:

`perturb_numerical(span: TaggedSpan, fact_sheet: FactSheet) -> list[FactSheet]`: For MONETARY_AMOUNT, shift the value above and below legally meaningful thresholds. For AGE, shift across boundaries: 7 (criminal responsibility), 12 (POCSO), 18 (majority), 21 (certain contract acts), 60 (retirement). For DURATION, double and halve the value, or shift across limitation periods (1 year, 3 years, 6 years, 12 years under the Limitation Act). Return a list of perturbed fact sheets, each with exactly one numerical fact changed.

`perturb_section(span: TaggedSpan, fact_sheet: FactSheet, adjacency_map: dict) -> list[FactSheet]`: Replace the cited section with each of its neighbors in the adjacency map. Return a list of perturbed fact sheets, each with exactly one section substituted.

`perturb_party_type(fact_sheet: FactSheet) -> list[FactSheet]`: For each party (petitioner, respondent), swap along predefined axes: Individual<->Minor, Employee<->Contractor, Tenant<->Licensee, Individual<->Corporation, State<->UnionOfIndia. Return perturbed fact sheets.

`perturb_evidence(fact_sheet: FactSheet) -> list[FactSheet]`: For each evidence item, create a version with that item removed. Also create versions with common evidence types added (DyingDeclaration, Confession, MedicalReport) if not already present. Return perturbed fact sheets.

Each function must record the perturbation metadata: `PerturbationEdge` dataclass with `fact_type` (enum: Numerical, Section, PartyType, Evidence), `original_value`, `perturbed_value`, `description` (human-readable, like "Age changed from 25 to 17 (crossed majority boundary)").

**4.2 -- Section adjacency map.** Write `countercase/counterfactual/section_adjacency.py`. Define a dictionary mapping normalized section strings to their adjacent sections. Cover at minimum:
- IPC: 299<->300, 300<->302, 302<->304, 304<->304A, 304B, 306, 307, 376, 376A-D, 420, 498A, 34, 120B, 149
- CPC: Order 7 Rule 11, Order 39, Section 9, Section 10, Section 11, Section 151
- Constitution: Articles 14, 19, 21, 32, 136, 226, 227, 300A, 368
- Evidence Act: Sections 3, 24, 25, 26, 27, 32, 45, 65B

The map is a `dict[str, list[str]]` where keys and values are normalized section strings. Include a function `get_adjacent_sections(section: str) -> list[str]`.

**4.3 -- LLM validation filter.** Write `countercase/counterfactual/llm_validator.py`. Implement class `PerturbationValidator` that takes a pluggable `llm_fn` (same interface as Phase 3). For each perturbed fact sheet, call the LLM with the validation prompt from plan.md section 7. Parse the response as JSON with fields `plausible`, `operative`, `reasoning`. Return a boolean accept/reject. Implement caching: store validation results in a dict keyed by `(perturbation_type, original_value, perturbed_value)` so structurally similar perturbations are not re-validated. Log accept/reject counts for monitoring.

**4.4 -- Single-level perturbation tree.** Write `countercase/counterfactual/perturbation_tree.py`. Define `TreeNode` dataclass: `node_id` (int), `fact_sheet` (FactSheet), `parent_id` (int | None), `edge` (PerturbationEdge | None), `retrieval_results` (list[RetrievalResult] | None), `children_ids` (list[int]). Define `PerturbationTree` class that holds a dict of `node_id -> TreeNode`. Implement:
- `build_root(fact_sheet: FactSheet) -> int`: create root node, run retrieval, store results, return node_id
- `expand_node(node_id: int, retriever: HybridRetriever, validator: PerturbationValidator, max_children: int = 5) -> list[int]`: generate all perturbations from the node's fact sheet using the four perturbation functions, validate each with the LLM validator, keep top `max_children` by plausibility, create child nodes, run retrieval for each child, store results, return child node_ids
- `get_node(node_id: int) -> TreeNode`
- `get_children(node_id: int) -> list[TreeNode]`
- `to_json() -> dict`: serialize entire tree to JSON

For Phase 4, only expand the root to depth 1 (single-level children).

**4.5 -- Basic diff view.** Write `countercase/counterfactual/sensitivity.py`. Implement:
- `compute_diff(parent_results: list[RetrievalResult], child_results: list[RetrievalResult], k: int = 10) -> DiffResult`: where `DiffResult` has `dropped_cases` (list of case_ids that were in parent top-K but not child), `new_cases` (list of case_ids in child but not parent), `stable_cases` (in both), `rank_displacements` (dict mapping case_id to displacement), `mean_displacement` (float).
- Rank displacement: for case in both sets, displacement = |rank_parent - rank_child|. For case in one set only, displacement = |rank - (K+1)|.

**4.6 -- Pipeline script.** Write `countercase/pipeline_phase4.py`. Given a single case (by case_id or by providing text), extract the fact sheet, build the root of the perturbation tree, expand to depth 1, compute diffs for each parent-child pair, and print: original fact sheet, each perturbation with its edge description, the diff (dropped/new/stable cases), and the mean displacement score. This demonstrates single-level counterfactual reasoning end-to-end.

**4.7 -- Update requirements.txt.** Add any new dependencies.

Technical constraints: perturbation functions must be pure functions (no side effects). The tree must be serializable to JSON for persistence. The LLM validator must gracefully handle LLM unavailability by accepting all perturbations with a warning log.

### Verification

After completing Phase 4, verify:
- Perturbation rules generate at least 3 perturbations for a typical case with multiple sections and party types
- Section adjacency map returns neighbors for IPC 302 (should include 300, 304, 304A)
- LLM validator accepts plausible perturbations and rejects obviously implausible ones (if LLM is available)
- Perturbation tree root has children after expansion
- Diff computation correctly identifies dropped and new cases
- Tree serializes to and from JSON without data loss

---

## Phase 5 -- Multi-Level Tree, Sensitivity Scoring, and UI

### Prompt

You are building Phase 5 of the COUNTERCASE project. Phases 1-4 are complete: single-level perturbation tree with diff computation is working. Read `plan.md` sections 7 (tree architecture, user interaction, sensitivity scoring) and 9 (output to user) for the full specification.

Phase 5 deliverables:

**5.1 -- Multi-level tree expansion.** Modify `countercase/counterfactual/perturbation_tree.py`. Add method `expand_tree(retriever: HybridRetriever, validator: PerturbationValidator, max_depth: int = 3, max_children_per_node: int = 5, min_displacement_threshold: float = 1.0)`. This method performs breadth-first expansion from the root:
- Expand root to depth 1 (already implemented)
- For each Level 1 node, compute the diff vs root. If `mean_displacement < min_displacement_threshold`, skip expansion (prune). Otherwise, expand to depth 2.
- Repeat for depth 3 if configured.
- Track timing per level and total node count. Log progress.
- The expansion must be interruptible: if a keyboard interrupt is received, stop expanding and return the partial tree.

**5.2 -- Aggregate sensitivity scoring.** Extend `countercase/counterfactual/sensitivity.py`. Add:
- `compute_sensitivity_scores(tree: PerturbationTree, k: int = 10) -> dict[str, float]`: for each fact type (Numerical, Section, PartyType, Evidence), collect all edges in the tree where that fact type was perturbed, compute the mean rank displacement across those edges, and return a dict mapping fact type name to its aggregate sensitivity score.
- `compute_per_case_sensitivity(tree: PerturbationTree, case_id: str, k: int = 10) -> dict`: for a specific case that appears in multiple nodes' results, compute how its rank changes across different perturbation paths.

**5.3 -- Manual node editing.** Add to `PerturbationTree`:
- `add_manual_node(parent_id: int, edited_fact_sheet: FactSheet, retriever: HybridRetriever, description: str = "Manual edit") -> int`: create a new child node from a user-edited fact sheet, run retrieval, compute diff vs parent, return node_id. This is the mechanism for user interaction.

**5.4 -- Streamlit UI.** Write `countercase/app/streamlit_app.py`. Build a Streamlit application with the following views:

Page 1 -- Query Input:
- Text area for pasting case text, OR text input for case_id lookup
- Year range sliders for metadata filtering
- Button "Extract Fact Sheet" that runs the fact sheet extractor and displays the structured fact sheet as an editable JSON form
- Button "Build Perturbation Tree" that triggers tree expansion

Page 2 -- Retrieval Results:
- Display the root node's top-K retrieval results in a table: rank, case_id, section_type, source_pdf, page_number, RRF score, snippet of chunk text
- Each row expandable to show full chunk text

Page 3 -- Perturbation Tree:
- Display the tree structure as an indented list or simple tree visualisation
- Each node shows: node_id, edge description (fact that changed), number of results
- Click a node to see its retrieval results
- Click an edge to see the diff view (dropped/new/stable cases)

Page 4 -- Diff View:
- Select two nodes (parent and child)
- Side-by-side table showing: case_id, rank in parent, rank in child, status (stable/dropped/new)
- Color coding: stable=grey, dropped=red, new=green
- Show the edge description at the top

Page 5 -- Sensitivity Dashboard:
- Bar chart of aggregate sensitivity scores per fact dimension
- Table of fact types ranked by sensitivity score
- Interpretation text: "The most legally operative fact for this case is {fact_type} with a sensitivity score of {score}"

Page 6 -- Manual Edit:
- Select a node from the tree
- Display its fact sheet as an editable form (text inputs for each field)
- Button "Apply Edit and Re-retrieve" that calls `add_manual_node` and refreshes the tree view

**5.5 -- Update requirements.txt.** Add `streamlit`, `plotly` (for charts).

**5.6 -- Pipeline script.** Write `countercase/pipeline_phase5.py` that builds a full depth-3 tree for a sample case, computes sensitivity scores, and exports the tree to JSON. Also include instructions for launching the Streamlit app.

Technical constraints: the Streamlit app must not block on tree expansion. Use `st.spinner` for long operations. Cache the retriever and indexes using `@st.cache_resource`. The tree must be stored in `st.session_state` to persist across Streamlit reruns. The UI must work on localhost without any external dependencies beyond the Python packages.

### Verification

After completing Phase 5, verify:
- Tree expands to depth 2 with pruning (some Level 1 nodes are skipped)
- Sensitivity scores differ across fact dimensions for a test case
- Manual node editing creates a new child and triggers retrieval
- Streamlit app launches and displays all six pages
- Diff view correctly color-codes dropped/stable/new cases
- Sensitivity dashboard shows a bar chart

---

## Phase 6 -- Explanation Engine and Output Format

### Prompt

You are building Phase 6 of the COUNTERCASE project. Phases 1-5 are complete: multi-level perturbation tree, sensitivity scoring, and Streamlit UI are working. Read `plan.md` section 8 for the explanation engine specification.

Phase 6 deliverables:

**6.1 -- Per-result explanation.** Write `countercase/explanation/per_result.py`. Implement `explain_result(query_fact_sheet: FactSheet, result: RetrievalResult, dpr_index: DPRIndexWrapper) -> str`. The function generates a one to two sentence explanation of why this result is relevant:
- Check for shared `act_sections` between the query fact sheet and the result metadata. If found: "This case involves {section}, which is also cited in your case."
- Check for matching party types. If found: "This case also involves a {party_type} as the {petitioner/respondent}."
- Compute DPR embedding similarity: retrieve the DPR embeddings for the query and the result chunk, compute cosine similarity, and identify the top 3 overlapping key terms between the query fact sheet text and the result chunk text using TF-IDF weighting (build a lightweight TF-IDF vectorizer over the corpus at init). If found: "This case shares similar factual patterns involving {term1}, {term2}, {term3}."
- If none of the above produce a grounded explanation: "This case was retrieved based on overall semantic similarity to your case."
- Never fabricate details not present in the result chunk or metadata.

**6.2 -- Counterfactual explanation.** Write `countercase/explanation/counterfactual_summary.py`. Implement `explain_edge(parent_node: TreeNode, child_node: TreeNode, diff: DiffResult) -> str`. Generate a one-paragraph summary:
- State the fact change: "When {edge.description}..."
- State dropped cases: "{N} precedents dropped out of the top results, including {case_ids}."
- For each dropped case, check if its metadata contains the original value of the changed fact (e.g., original section). If so: "These cases cited {original_section}, which is no longer relevant after the change."
- State new cases: "{N} new precedents became applicable, including {case_ids}."
- For new cases, check if their metadata contains the perturbed value. If so: "These cases involve {perturbed_value}, which matches the altered facts."
- If no dropped or new cases: "This fact change did not significantly alter the retrieval results, suggesting it is not a legally operative fact for precedent applicability."

**6.3 -- JSON output format.** Write `countercase/explanation/output_formatter.py`. Implement `format_node_output(node: TreeNode, diff: DiffResult | None, explanations: dict[str, str], sensitivity_scores: dict[str, float] | None) -> dict`. Returns a structured dict matching the JSON schema from plan.md section 8:

```json
{
    "node_id": 0,
    "fact_sheet_state": { ... },
    "retrieval_results": [ { "chunk_id": "...", "source_pdf": "...", "page_number": 1, "section_type": "Held", "rank": 1, "rrf_score": 0.03, "reranker_score": 0.85, "case_id": "...", "year": 2020, "explanation": "..." } ],
    "diff_vs_parent": { "dropped_cases": [...], "newly_appeared_cases": [...], "stable_cases": [...], "rank_displacements": {...} },
    "sensitivity_scores": { "Numerical": 3.2, "Section": 8.5, "PartyType": 1.0, "Evidence": 5.3 },
    "counterfactual_summary": "When ..."
}
```

Implement `format_tree_output(tree: PerturbationTree, ...) -> dict` that formats the entire tree as a nested JSON structure.

**6.4 -- Text summary generator.** Implement `generate_text_summary(tree_output: dict) -> str` in the same file. Deterministically project the JSON into a human-readable Markdown report:
- Section 1: Query case fact sheet (formatted as a table)
- Section 2: Top retrieved precedents with per-result explanations
- Section 3: Counterfactual analysis with a table of perturbations, dropped/new counts, and sensitivity scores
- Section 4: Conclusion summarizing which facts are most legally operative
This is a pure string-formatting function. No LLM calls. The text must be derivable from the JSON alone.

**6.5 -- Integrate explanations into Streamlit UI.** Update `countercase/app/streamlit_app.py`:
- In the Retrieval Results page, add per-result explanations below each result
- In the Diff View page, add the counterfactual summary paragraph
- Add an "Export" button that generates the JSON output and the Markdown text summary, and provides download links for both

**6.6 -- Pipeline script.** Write `countercase/pipeline_phase6.py`. Run end-to-end on a sample case: extract fact sheet, build tree to depth 2, generate all explanations, format as JSON and Markdown, write both to files in `countercase/output/`.

Technical constraints: explanation generation must be fast (under 100ms per result, no LLM calls). The JSON output must be valid and parseable. The Markdown summary must render correctly in any standard Markdown viewer. No emojis in explanations or output.

### Verification

After completing Phase 6, verify:
- Per-result explanations reference shared sections or factual patterns from the actual result metadata
- Counterfactual summaries correctly name dropped and new cases
- JSON output validates against the schema
- Markdown summary is readable and contains all sections
- Streamlit export button produces downloadable files
- No explanation contains fabricated information not present in the source data

---

## Phase 7 -- Evaluation, Ablation, and Research Writeup Support

### Prompt

You are building Phase 7 of the COUNTERCASE project. Phases 1-6 are complete: the full system is operational with retrieval, counterfactual reasoning, explanations, and UI. Read `plan.md` sections 10 and 11 for the evaluation plan.

Phase 7 deliverables:

**7.1 -- Retrieval evaluation.** Extend `countercase/evaluation/eval_harness.py`. Create a comprehensive evaluation script that:
- Loads the full index (all years indexed)
- Runs all five ablation modes on the test set: DPR-only, ChromaDB-only, hybrid-RRF-no-MMR, hybrid-RRF-MMR-no-reranker, full-system
- Computes MRR@10, NDCG@10, Recall@5, Recall@10, Recall@20 for each mode
- Computes paired t-tests between the full system and each baseline for MRR@10 and NDCG@10
- Writes results to `countercase/evaluation/results/retrieval_eval.json` with a LaTeX-compatible table format
- Generates a comparison bar chart (saved as PNG) showing all metrics across all modes

**7.2 -- Counterfactual evaluation.** Write `countercase/evaluation/counterfactual_eval.py`. Implement:
- Load a set of evaluation cases (hand-curated JSON file) where each case has `case_id`, `dispositive_facts` (list of fact types that are known to matter), `non_dispositive_facts` (list of fact types that should not matter)
- For each case, build the perturbation tree to depth 2
- Compute sensitivity scores for each fact dimension
- Check: is the sensitivity score for dispositive facts higher than for non-dispositive facts?
- Compute Spearman rank correlation between human-annotated fact importance ordering and system sensitivity ordering
- Log the LLM validation filter accept/reject rate
- Write results to `countercase/evaluation/results/counterfactual_eval.json`

**7.3 -- Explanation evaluation tooling.** Write `countercase/evaluation/explanation_eval.py`. Implement:
- `compute_faithfulness(explanation: str, source_chunk: str, source_metadata: dict) -> float`: for each sentence in the explanation, check if it can be grounded in the source. A sentence is grounded if: (a) it references an act section that is in the source metadata, (b) it references a factual term that appears in the source chunk text, or (c) it is the fallback "retrieved based on overall semantic similarity" statement. Return the fraction of grounded sentences.
- Run faithfulness evaluation over a sample of 100 explanations from a test run and report the mean faithfulness score.
- Write results to `countercase/evaluation/results/explanation_eval.json`
- Generate a template for human evaluation: a CSV file with columns `case_id`, `explanation_text`, `clarity_score` (blank, 1-5), `accuracy_score` (blank, 1-5), `usefulness_score` (blank, 1-5) for legal experts to fill in.

**7.4 -- Evaluation test set creation tool.** Write `countercase/evaluation/create_test_set.py`. A utility script that helps create the evaluation test set:
- Given a list of case_ids, extract their fact sheets and retrieval results
- Prompt the user to annotate: for each query case, which retrieved cases are relevant? (interactive CLI)
- Save annotations to the test set JSON format expected by the eval harness
- Include a mode to auto-generate partial ground truth by using citation analysis: extract case citations from each judgment and treat cited cases as relevant (weak supervision)

**7.5 -- Results aggregation.** Write `countercase/evaluation/aggregate_results.py`. Combine all evaluation results into a single summary:
- Retrieval metrics table (all baselines vs full system)
- Counterfactual correlation scores
- Explanation faithfulness scores
- Generate LaTeX tables for the research paper
- Generate summary figures as PNGs

**7.6 -- Update requirements.txt.** Add `matplotlib`, `scipy` (for statistical tests), `tabulate` (for LaTeX table generation).

Technical constraints: evaluation scripts must be runnable independently. All results must be saved as JSON for reproducibility. Statistical tests must report p-values. Charts must be publication-quality (no emojis, clear labels, appropriate font sizes).

### Verification

After completing Phase 7, verify:
- Retrieval evaluation produces a table with all five modes and all metrics
- Paired t-tests report p-values for each comparison
- Counterfactual evaluation computes Spearman correlation (even if on a small test set)
- Explanation faithfulness score is computed and reported
- Human evaluation CSV template is generated with correct format
- LaTeX tables are syntactically valid
- All results are saved as reproducible JSON files

---

## Post-Phase Checklist

After all seven phases are complete, verify the following end-to-end:

1. Run `pipeline_phase1.py` on year range 2020-2024 to build the full index
2. Run `pipeline_phase2.py` to verify retrieval quality
3. Run `pipeline_phase3.py` to extract fact sheets for sample cases
4. Run `pipeline_phase4.py` to build a single-level perturbation tree
5. Run `pipeline_phase5.py` to build a depth-3 tree with sensitivity scores
6. Run `pipeline_phase6.py` to generate explanations and export JSON + Markdown
7. Run `pipeline_phase7.py` (eval harness) to produce evaluation results
8. Launch Streamlit app with `streamlit run countercase/app/streamlit_app.py` and verify all pages work
9. Verify that the JSON output from step 6 contains all required fields
10. Verify that the Markdown report from step 6 renders correctly
11. Verify that evaluation results from step 7 include all metrics, statistical tests, and figures

---

## Notes for the Agent

- Always read the relevant section of `plan.md` before starting a phase. The plan contains specific design decisions, parameter values, and architectural constraints that must be followed exactly.
- Each phase produces pipeline scripts (`pipeline_phaseN.py`) that serve as smoke tests. Run them after implementation to verify correctness.
- If a phase requires an LLM (Phases 3, 4) and no LLM is available locally, implement the code with the API backend and test with mock responses. The code must be correct even if the LLM is not reachable.
- The judgment data is under `judgments-data/`. Tar files must be extracted first using `extract-judgments.ps1`. Parquet metadata files are directly readable.
- Never use emojis anywhere: not in code, comments, docstrings, log messages, UI text, or output files.
- All paths in config should be relative to the repository root, not absolute.
- Prefer composition over inheritance. Each module should be independently importable and testable.
- When in doubt about a design decision, follow `plan.md`. If `plan.md` does not cover it, choose the simpler option.
