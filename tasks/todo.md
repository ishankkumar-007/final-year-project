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
- [ ] 2.1 MMR implementation
- [ ] 2.2 Cross-encoder re-ranking
- [ ] 2.3 Hybrid retriever
- [ ] 2.4 Evaluation harness
- [ ] 2.5 Update requirements.txt
- [ ] 2.6 Pipeline script (pipeline_phase2.py)
