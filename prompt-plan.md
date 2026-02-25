# Copilot Agent Prompt — COUNTERCASE RAG Pipeline Plan

---

## Prompt

You are a senior ML systems architect specializing in production-grade RAG (Retrieval-Augmented Generation) pipelines. Your task is to produce a comprehensive, opinionated technical plan for a project called **COUNTERCASE** — a legal case retrieval system with counterfactual reasoning, targeting the Indian Supreme Court corpus.

Write the full plan to a file named `plan.md`.

---

## Project Context

COUNTERCASE has three core modules:

1. **Legal Case Retrieval** — hybrid search (BM25 + dense embeddings) to find legally relevant precedents by both factual and legal-issue similarity
2. **Counterfactual Reasoning** — alter key facts in a query case and observe how retrieved precedents shift; the primary user-facing value is showing how a change in case facts changes which precedents are applicable
3. **Explanation Engine** — text-grounded justifications linking query facts to retrieved precedents and showing counterfactual impact via a side-by-side diff view with sensitivity scores

The input is Indian Supreme Court judgment PDFs sourced from the **AWS Marketplace / AWS Data Exchange (ADX) — Indian Supreme Court Judgments** dataset. The text in these PDFs is selectable and extractable; no OCR is required.

The delivery target is **both**: a working end-to-end prototype and a research writeup where novelty and evaluation rigour matter.

---

## Technical Constraints and Preferences

- **Text extraction**: PDFs have selectable text; use `pdfplumber` or `PyMuPDF` — do not use OCR
- **Chunking**: Use `LangChain RecursiveCharacterTextSplitter`; favour long chunks with meaningful overlap (legal text is dense with context)
- **Embeddings**: `all-MiniLM-L6-v2` for dense retrieval; consider `InLegalBERT` or `LegalBERT` for domain-specific re-ranking
- **Vector store**: `ChromaDB`
- **No emojis anywhere in the codebase or documentation**
- **Metadata filtering is a first-class concern** — pre-filter on court, year, jurisdiction, bench type before vector search; do not post-filter on high-selectivity fields
- **MMR (Maximal Marginal Relevance)** must be used at retrieval time to balance relevance with diversity — every retrieved chunk must earn its place by adding new information
- **Source attribution**: every chunk returned to the user must reference its source PDF and page number
- **Compute**: the architecture must be designed to be flexible — it should work on a single GPU (T4/V100 class) for fine-tuning small models, but should degrade gracefully to CPU-only if needed. Do not hard-depend on multi-GPU infrastructure

---

## What the Plan Must Cover

Structure `plan.md` with the following sections. Write in clear technical prose — avoid vague language and avoid bullet-point padding. Use prose paragraphs with inline technical specifics.

---

### 1. Executive Summary

One paragraph: what COUNTERCASE does, why it is novel, and what gap it fills in existing Indian legal AI systems. Be specific — name the gap (retrieval systems that are fact-similarity-only, with no mechanism to reason about which facts causally drive precedent applicability).

---

### 2. Novelty and Prior Work Gap

Be precise. Discuss what existing systems do and what they do not do:

- **OpenNyAI / ILDC**: covers Indian legal NLP but focuses on judgment outcome prediction, not retrieval sensitivity analysis
- **FIRE / SemEval legal tracks**: retrieval benchmarks but no counterfactual component
- **Standard RAG over legal corpora**: retrieves by semantic similarity but cannot answer "would these precedents still apply if this fact changed?"

Articulate clearly that the novel contribution is the **perturbation tree framework** — a structured, chained counterfactual exploration where each node in the tree represents a perturbed fact state and its corresponding retrieval result set. This has not been done in the Indian legal NLP domain. Also note that the combination of auto-parsed structured fact sheets as input, hybrid NER-rule-LLM perturbation, and rank-diff sensitivity scoring is a distinct architectural contribution.

---

### 3. Dataset

The corpus is the **AWS Data Exchange — Indian Supreme Court Judgments** dataset. Describe:

- How to access and ingest it programmatically via the ADX API or S3 sync
- Expected data characteristics: judgment PDFs, varying length (short orders to 100+ page constitutional bench decisions), inconsistent formatting across decades
- What metadata is natively available vs. what must be extracted (case number, year, bench composition, statutes cited, outcome)
- Any known quality issues: scanning artifacts in older judgments (pre-2000), inconsistent citation formats

This is the **single dataset** used for both RAG indexing and counterfactual reasoning. No separate annotated counterfactual dataset is needed because the counterfactual module operates on the retrieval layer — it re-queries the same index with perturbed fact representations and compares result sets. Make this architectural decision explicit and justify it: it means the system requires zero counterfactual-specific annotation, which is a practical advantage and a point of novelty.

---

### 4. Data and Preprocessing Pipeline

- PDF ingestion with `pdfplumber` or `PyMuPDF`; handle multi-column layouts, headers/footers, cause-list noise, and page numbering inconsistencies
- **Metadata extraction schema**: `{case_id, court, bench_type, year, month, act_sections, citation_string, judge_names, outcome_label, pdf_path, page_range}`
- **Structured fact sheet extraction**: each ingested case must also produce a structured fact sheet — a parsed representation of the key legally operative facts. This fact sheet is the input unit to the counterfactual module. Design a fact schema: `{parties: {petitioner_type, respondent_type}, evidence_items: [...], sections_cited: [...], numerical_facts: {amounts, ages, durations}, outcome: ...}`. Describe how this is extracted: a combination of rule-based section detection (locating the "Facts" or "Background" section of a judgment) and an LLM prompt over that section to populate the schema
- **Chunking strategy**: chunk size of 512-1024 tokens with 128-token overlap, using `RecursiveCharacterTextSplitter` with legal-aware separators (split on paragraph breaks and section headers before character boundaries). Justify: legal sentences are long and co-referential; short chunks lose the antecedent of pronouns and defined terms
- Distinguish structured sections (Facts, Issues, Held, Ratio Decidendi) from unstructured prose; tag chunks with their section type as metadata

---

### 5. Indexing Architecture

- **Dual-index design**: BM25 (via `rank_bm25` or Elasticsearch) for sparse lexical retrieval + ChromaDB with `all-MiniLM-L6-v2` embeddings for dense semantic retrieval
- **Metadata schema in ChromaDB**: include `year`, `court`, `bench_type`, `act_sections`, `section_type` (facts/held/ratio), `outcome_label` as filterable fields
- **Staged hybrid filtering**:
  - Stage 1 (pre-filter, indexed): `year_range`, `act_sections`, `bench_type` — reduces the candidate set before ANN search. Apply pre-filter when field selectivity is below 10% of the corpus
  - Stage 2 (ANN vector search): HNSW over the filtered subset in ChromaDB
  - Stage 3 (post-filter, optional): lightweight refinement on non-indexed fields such as `word_count` or `judge_name`
- Justify the selectivity rule explicitly: pre-filtering on `year_range` when querying for a 5-year window over a 70-year corpus eliminates roughly 93% of documents before vector search — this is the correct place to filter

---

### 6. Retrieval Module

- **Hybrid fusion**: Reciprocal Rank Fusion (RRF) over BM25 and dense retrieval scores; explain why RRF is preferable to weighted score combination (score scales are incommensurable across BM25 and cosine similarity)
- **MMR**: apply Maximal Marginal Relevance post-fusion with a tunable `lambda_mult` parameter; for legal retrieval, recommend starting at `lambda_mult=0.6` (slightly relevance-heavy) because diversity is important but a precedent from the same IPC section must not be penalised for thematic overlap
- **Re-ranking**: cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2` or a fine-tuned `InLegalBERT` cross-encoder) over top-K candidates; the re-ranker scores query-chunk pairs directly and does not rely on embedding geometry
- **Source attribution**: each result chunk carries `{source_pdf, page_number, chunk_id, section_type, metadata}` — surface this to the user alongside the retrieved text
- **Fine-tuning decision point**: describe the tradeoff between zero-shot cross-encoder re-ranking (fast to deploy, no labelled data needed) and fine-tuning on Indian SC relevance judgements (higher precision, requires annotation effort). Since compute is not yet fixed, design the system so the re-ranker is a swappable component

---

### 7. Counterfactual Reasoning Module

This is the novel core of the system. Design it with full architectural specificity.

**Input**: a structured fact sheet auto-parsed from a query case (see Section 4). The fact sheet schema is `{parties, evidence_items, sections_cited, numerical_facts, outcome}`. This is what gets perturbed — not raw text. The query case can be one of the indexed SC judgments or a new case provided by the user.

**Fact types in scope for perturbation** (all four must be handled):
- Numerical and quantitative facts (monetary amounts, ages, durations, quantum of evidence)
- Legal sections cited (e.g., IPC 302 to IPC 304)
- Party type or relationship (employee vs. contractor, tenant vs. licensee, minor vs. adult)
- Presence or absence of a piece of evidence (boolean toggle on an evidence item)

**Fact perturbation — hybrid three-stage pipeline**:

Stage 1 is NER-based identification: a legal NER model (fine-tuned on Indian legal text, using available Indian legal NER corpora as a starting point, e.g., from OpenNyAI) tags legally operative spans in the fact sheet — monetary amounts, IPC/CPC section numbers, party role labels, and evidence mentions. These tagged spans become the perturbation candidates.

Stage 2 is rule-based perturbation: for each tagged span type, apply deterministic perturbation rules. Numerical facts are scaled by a legally meaningful factor — for example, compensation amounts are shifted above or below a statutory threshold, or the age of a party is moved across the 18-year majority boundary. Legal sections are substituted with adjacent sections in the same chapter using a hand-curated section adjacency map (e.g., IPC 302 to IPC 304, IPC 304 to IPC 304A). Party type is swapped along a predefined axis from a closed vocabulary. Evidence presence is toggled as a boolean.

Stage 3 is LLM validation: before a perturbed fact sheet is accepted into the perturbation tree, it is passed to a lightweight LLM (e.g., `Mistral-7B` quantized, or a hosted API call) with a prompt that checks two things — whether the perturbation is legally plausible (does a case with these facts plausibly exist in Indian jurisprudence?) and whether the perturbation modifies a legally operative fact rather than a stylistic one. Perturbations that fail either check are discarded before tree expansion.

**Perturbation tree architecture**:

The counterfactual module produces a **perturbation tree**, not a flat list of alternatives. Each node in the tree is a fact sheet state (original or perturbed). Each directed edge represents a single fact change applied to the parent node. The tree is constructed as follows:

- The root node is the original fact sheet
- Level 1 children are each single-fact perturbation that passes LLM validation
- Level 2 children are valid single-fact perturbations applied on top of a Level 1 node — meaning two facts have been changed from the original
- Tree depth is capped (recommended default: depth 3, branching factor configurable per deployment) to prevent combinatorial explosion

For each node in the tree, the full hybrid retrieval pipeline (pre-filter, ANN, RRF, MMR, re-rank) is executed and the top-K result set is stored at that node.

**User interaction**: the user can manually edit the fact sheet at any node — modifying a field value directly in the UI — and the system re-runs retrieval from that node and attaches the result as a new child, extending the tree. Both auto-perturbation and manual editing produce nodes in the same tree data structure, giving the user full control to explore directions the auto-perturbation did not take.

**Comparison and sensitivity scoring**:

For any parent-child pair of nodes in the tree, the system surfaces: (a) cases that dropped out of top-K between parent and child — these precedents were sensitive to the changed fact; (b) cases that newly appeared in the child's top-K — these are precedents applicable only under the perturbed fact state; (c) a rank displacement score per case, defined as the absolute change in rank position across the two result sets, where cases absent from one set receive a rank of K+1; (d) an aggregate fact sensitivity score per fact dimension, computed as the mean rank displacement across all tree edges where that fact type was perturbed.

The side-by-side diff view presents the two result sets with cases colour-coded as stable (present in both at similar rank), dropped, or newly appeared, with explicit rank positions in each set.

---

### 8. Explanation Engine

- **Per-result explanation**: for each retrieved case, generate a plain-language justification grounded strictly in the retrieved chunk text — shared act sections, matching factual patterns, similar party roles. Use token-level similarity (BM25 term overlap or cross-encoder attention weights) to identify which spans drove the match. The explanation must not assert anything not present in the retrieved chunk.
- **Counterfactual explanation**: for each edge in the perturbation tree, generate a one-paragraph natural language summary covering which fact changed, which cases dropped out and why (what in those cases depended on the original fact), and which cases appeared and what in them matches the new fact state.
- **Output format**: structured JSON `{node_id, fact_sheet_state, retrieval_results, diff_vs_parent, sensitivity_scores, explanations}` plus a human-readable text summary derived from the JSON. The JSON is the ground truth; the text summary is generated from it to prevent hallucination through unconstrained generation.

---

### 9. System Architecture — End-to-End Data Flow

Describe and render the full pipeline as an ASCII diagram:

```
PDF Corpus (ADX — Indian SC Judgments)
        |
        v
[Ingestion + Text Extraction] (pdfplumber / PyMuPDF)
        |
        v
[Preprocessing] (section detection, noise removal, metadata extraction)
        |
        +---> [Fact Sheet Extraction] (rule-based section detection + LLM) --> stored per case
        |
        v
[Chunking] (RecursiveCharacterTextSplitter, 512-1024 tokens, 128 overlap, section-type tagged)
        |
        v
[Dual Indexing]
    BM25 Index (rank_bm25 / Elasticsearch)
    ChromaDB (all-MiniLM-L6-v2, HNSW, metadata fields)
        |
        v
[Query Input] (raw case text OR user-provided case)
        |
        v
[Auto Fact Sheet Parsing] --> [Root Node of Perturbation Tree]
        |
        v
[Hybrid Retrieval per Node]
    Stage 1: Metadata pre-filter (year, act_sections, bench_type)
    Stage 2: BM25 + ANN vector search (HNSW on filtered subset)
    Stage 3: RRF fusion
    Stage 4: MMR re-ranking (lambda_mult=0.6 default)
    Stage 5: Cross-encoder re-ranking (swappable component)
    Stage 6: Source attribution attached to each result chunk
        |
        v
[Perturbation Tree Expansion]
    NER tagging of fact sheet spans
    Rule-based perturbation per fact type
    LLM plausibility validation
    Auto-expand to depth D OR user manually edits a node
        |
        v
[Per-node Retrieval] (full retrieval pipeline re-run for each tree node)
        |
        v
[Diff + Sensitivity Scoring]
    Dropped cases, newly appeared cases, rank displacement per case
    Aggregate fact sensitivity score per fact dimension
        |
        v
[Explanation Engine]
    Per-result chunk-grounded justification
    Per-edge counterfactual summary paragraph
    JSON output + derived text summary
        |
        v
[Output to User]
    Side-by-side diff view, perturbation tree visualisation
    Sensitivity scores, source-attributed retrieved chunks
    Explanations in JSON and human-readable text
```

---

### 10. Evaluation Plan

**Retrieval quality**: measure MRR@10, NDCG@10, and Recall@K on a held-out subset of SC judgments with manually annotated relevant precedents. Baselines to beat: BM25-only, dense-only without hybrid, hybrid without MMR, and hybrid with MMR but without cross-encoder re-ranking.

**Counterfactual module**: for a set of cases where the legal outcome is known to hinge on a specific fact (for example, cases decided on the age of majority or on the presence of a dying declaration), verify that perturbing that fact produces a high rank displacement score while perturbing an irrelevant fact produces a low score. Separately, measure the LLM validation filter's accept/reject rate and manually audit a sample of accepted perturbations for legal plausibility. Measure perturbation tree coverage: what fraction of the legally meaningful fact variations does the tree explore at depth 2 versus depth 3.

**Explanation quality**: measure faithfulness as the fraction of explanation sentences that can be directly traced to a source span in the retrieved chunk. Separately, conduct a human evaluation with legal domain experts using a 5-point Likert scale for explanation clarity, legal accuracy, and usefulness of the counterfactual summary.

---

### 11. Implementation Roadmap

Phase the work into clear milestones with deliverables:

- **Phase 1**: ADX dataset ingestion, PDF extraction, metadata schema, BM25 and ChromaDB dual indexing, baseline hybrid retrieval with RRF
- **Phase 2**: MMR integration, cross-encoder re-ranking, source attribution, evaluation harness with MRR and NDCG metrics
- **Phase 3**: Structured fact sheet extraction pipeline — rule-based section detection plus LLM population of the fact schema, NER-based perturbation candidate tagging
- **Phase 4**: Rule-based perturbation rules for all four fact types, LLM validation filter, single-level perturbation and basic diff view
- **Phase 5**: Perturbation tree at depth 2 and 3, user manual editing of tree nodes, aggregate sensitivity scoring, side-by-side diff UI
- **Phase 6**: Explanation engine, JSON output format, text summary generation from JSON
- **Phase 7**: Ablation studies, evaluation against all baselines, research writeup

---

### 12. Key Risks and Mitigations

Address each of the following with a mitigation strategy:

- **Fact sheet extraction quality**: if the NLP pipeline mis-parses the facts section, all downstream counterfactuals are wrong. Expose the parsed fact sheet to the user for correction before the tree is built; include a manual override interface.
- **LLM validation cost and latency**: calling an LLM per perturbation candidate multiplies latency at tree depth. Mitigate by caching validation results keyed by perturbation type and section citation; batch API calls; use a small quantized local model rather than a hosted API where compute allows.
- **Perturbation tree explosion**: at depth 3 with four fact types and multiple values per type, the tree can become very large. Cap branching factor; prune nodes where the sensitivity score delta from the parent is below a configurable threshold — if a perturbation had no effect on retrieval, it is not worth exploring further.
- **Hallucination in explanations**: constrain the explanation generator to only use extractive or tightly anchored generative methods over retrieved chunk text; do not allow free-form generation without a source span anchor.
- **Dataset coverage gaps**: the ADX SC Judgments corpus covers the Supreme Court only. Scope the system explicitly to this court; document the boundary clearly in the research writeup so reviewers do not expect subordinate court coverage.
- **Compute uncertainty**: design all model components as swappable with lightweight defaults (CPU-compatible `all-MiniLM-L6-v2`, zero-shot cross-encoder) and documented upgrade paths to GPU-dependent alternatives (fine-tuned InLegalBERT cross-encoder, hosted LLM API for validation).

---

### 13. Open Research Questions

Identify these as genuine open questions that could become research contributions in the writeup:

- What is the right granularity for legal fact perturbation — entity span level versus structured schema field level — and does granularity affect the legal plausibility of generated counterfactuals?
- How should the perturbation tree be pruned without losing legally meaningful branches? A retrieval-similarity-based pruning criterion may miss cases where a small textual change has a large legal effect.
- Is rank displacement a valid proxy for legal sensitivity, or do legal experts find that rank changes are sometimes caused by lexical artefacts rather than genuine legal distinction? This is an empirical question that the human evaluation in Section 10 is designed to answer.
- Can the topology of the perturbation tree itself — its depth, branching density, and the distribution of sensitivity scores across edges — be used to characterise how legally robust a case is as a precedent? A case whose retrieval results are insensitive to all perturbations may be a more stable and universally applicable precedent than one that is highly sensitive.

---

## Writing Instructions

- Write in clear, technical prose. Use prose paragraphs with inline technical specifics — do not use bullet-point padding anywhere in the plan.
- Be opinionated: if there is a clearly better design choice, say so and justify it.
- Do not include emojis anywhere in the plan, in code examples, or in config snippets.
- Where a design decision is genuinely contested, present the tradeoff explicitly rather than hedging.
- The audience is the project team (ML engineers and legal domain experts) plus academic reviewers evaluating novelty.
- Target length: 3,000 to 5,000 words in `plan.md`.