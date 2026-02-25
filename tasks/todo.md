# COUNTERCASE - Task Tracking

## Phase 1 -- Data Infrastructure and Baseline Retrieval

- [x] 1.1 Project scaffolding (directory structure, requirements.txt, settings.py)
- [x] 1.2 Metadata ingestion (load_metadata, inspect_metadata_schema, metadata_inspection.json)
- [x] 1.3 PDF text extraction (pdfplumber, multi-column, header/footer removal)
- [x] 1.4 Noise filtering (cause-list removal, unicode normalization, whitespace collapse)
- [x] 1.5 Section detection (regex-based heading matching, fallback heuristic)
- [x] 1.6 Chunking (RecursiveCharacterTextSplitter, tiktoken cl100k_base, section tagging)
- [x] 1.7 Dual indexing (DPR FAISS + ChromaDB with metadata)
- [x] 1.8 RRF fusion (reciprocal rank fusion)
- [x] 1.9 End-to-end pipeline script (pipeline_phase1.py)
- [x] 1.10 Metadata-driven extraction (extract_metadata_from_text regex)
- [x] Verification: Smoke test on 10 PDFs from 2024 -- all checks pass

## Phase 2 -- Retrieval Enhancements
- [x] 2.1 MMR implementation (countercase/retrieval/mmr.py)
- [x] 2.2 Cross-encoder re-ranking (countercase/retrieval/reranker.py)
- [x] 2.3 Hybrid retriever (countercase/retrieval/hybrid_retriever.py)
- [x] 2.4 Evaluation harness (countercase/evaluation/metrics.py, eval_harness.py)
- [x] 2.5 Update requirements.txt (added scipy)
- [x] 2.6 Pipeline script (countercase/pipeline_phase2.py)
- [x] Verification: Full pipeline on 159 indexed chunks -- MMR diversifies, cross-encoder reorders, eval harness runs 5 queries × 3 modes

## Phase 3 -- Fact Sheet Extraction
- [x] 3.1 Pydantic fact sheet schema (schema.py: FactSheet, PartyInfo, EvidenceItem, NumericalFacts)
- [x] 3.2 Section locator (section_locator.py: 3-strategy facts section extraction)
- [x] 3.3 LLM-based fact sheet extractor (fact_sheet_extractor.py: mock/local/API backends)
- [x] 3.4 NER tagger for perturbation candidates (ner_tagger.py: 6 entity types, regex-based)
- [x] 3.5 Fact store (fact_store.py: JSON file storage with round-trip verification)
- [x] 3.6 Pipeline script (pipeline_phase3.py: end-to-end with early termination)
- [x] 3.7 Requirements update (pydantic, openai, requests)
- [x] Verification: 10/10 facts sections found (100%), 10/10 extractions OK, 215+ NER spans, fact store round-trip OK
