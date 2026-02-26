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

## Phase 3 -- Fact Sheet Extraction and NER

### Run full Phase 3 pipeline (section locator, LLM extraction, NER tagging, fact store)

```powershell
python -m countercase.pipeline_phase3
```

### Process a specific year range

```powershell
python -m countercase.pipeline_phase3 --start-year 2024 --end-year 2025
```

### Limit number of cases processed

```powershell
python -m countercase.pipeline_phase3 --max-cases 5
```

Output: fact sheets saved to `countercase/data/fact_store/`

## Phase 4 -- Perturbation Logic and Single-Level Tree

### Run Phase 4 pipeline with sample fact sheet

```powershell
python -m countercase.pipeline_phase4
```

### Load a fact sheet from the fact store by case ID

```powershell
python -m countercase.pipeline_phase4 --case-id "Criminal Appeal 1031/2024"
```

### Extract fact sheet from a text file and build tree

```powershell
python -m countercase.pipeline_phase4 --text path/to/judgment.txt
```

### Control maximum child perturbations per node

```powershell
python -m countercase.pipeline_phase4 --max-children 8
```

Output: perturbation tree saved to `countercase/output/phase4_tree.json`
