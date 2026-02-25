# COUNTERCASE -- Command Reference

Quick-reference for running each phase of the project.

## Environment Setup

```powershell
conda activate torch2
```

## Phase 1 -- Data Infrastructure and Baseline Retrieval

### Extract judgment PDFs from tar archives

```powershell
powershell -File .\extract-judgments.ps1 -StartYear 2024 -EndYear 2025
```

### Run full Phase 1 pipeline (extract, chunk, index, query)

```powershell
python -m countercase.pipeline_phase1 --start-year 2024 --end-year 2025
```

### Quick smoke test (10 PDFs, fast)

```powershell
python -m countercase.tests.test_phase1_smoke
```

### Inspect metadata schema for a year range

```powershell
python -c "from countercase.ingestion.metadata_extractor import inspect_metadata_schema; inspect_metadata_schema(2024, 2024)"
```

Output: `countercase/data/metadata_inspection.json`

## Phase 2 -- Retrieval Enhancements

### Run full Phase 2 pipeline (six-stage retrieval + ablation comparison + mini evaluation)

```powershell
python -m countercase.pipeline_phase2
```

### Run with a custom query

```powershell
python -m countercase.pipeline_phase2 "writ petition Article 21 right to life"
```

### Run evaluation harness on a test set

```powershell
python -m countercase.evaluation.eval_harness --test-set countercase/data/test_set.json --output countercase/data/eval_report.json
```

### Run specific ablation modes only

```powershell
python -m countercase.evaluation.eval_harness --test-set countercase/data/test_set.json --modes full_system chroma_only dpr_only
```
