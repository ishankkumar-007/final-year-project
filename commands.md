# COUNTERCASE -- Command Reference

Quick-reference for running each phase of the project.

## Environment Setup

```powershell
conda activate torch2
```

## Helper Scripts

### Extract judgment PDFs from tar archives

```powershell
# Windows
powershell -File .\extract-judgments.ps1 -StartYear 2024 -EndYear 2025

# Linux/macOS
./extract-judgments.sh 2024 2025
```

### Run Phase 7 evaluation suite

```powershell
# Windows -- all steps
.\run-phase7.ps1

# Windows -- specific steps only
.\run-phase7.ps1 -Steps "retrieval,counterfactual"

# Linux/macOS -- all steps
./run-phase7.sh

# Linux/macOS -- specific steps only
./run-phase7.sh --steps retrieval,counterfactual
```

---

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

## Phase 5 -- Multi-Level Tree, Sensitivity Scoring, and UI

### Run Phase 5 pipeline with default settings (depth 3)

```powershell
python -m countercase.pipeline_phase5
```

### Load a fact sheet from the fact store

```powershell
python -m countercase.pipeline_phase5 --case-id "Criminal Appeal 1031/2024"
```

### Control tree depth and branching

```powershell
python -m countercase.pipeline_phase5 --max-depth 2 --max-children 3
```

### Set displacement threshold for pruning

```powershell
python -m countercase.pipeline_phase5 --min-displacement 0.5
```

### Launch the Streamlit UI

```powershell
streamlit run countercase/app/streamlit_app.py
```

Output: perturbation tree saved to `countercase/output/phase5_tree.json`

## Phase 6 -- Explanation Engine and Output Format

### Run Phase 6 pipeline (explanations, JSON + Markdown export)

```powershell
python -m countercase.pipeline_phase6
```

### Load a specific case from the fact store

```powershell
python -m countercase.pipeline_phase6 --case-id "Criminal Appeal 1031/2024"
```

### Control tree depth

```powershell
python -m countercase.pipeline_phase6 --max-depth 2 --max-children 5
```

### Run via helper script (Windows)

```powershell
.\run-phase6.ps1
.\run-phase6.ps1 -CaseId "Criminal Appeal 1031/2024" -MaxDepth 3 -LaunchUI
```

### Launch the Streamlit UI (includes export buttons)

```powershell
streamlit run countercase/app/streamlit_app.py
```

Output: `countercase/output/phase6_tree.json`, `countercase/output/phase6_report.md`

## Phase 7 -- Evaluation, Ablation, and Research Writeup

### Run all Phase 7 steps via helper script

```powershell
# Windows
.\run-phase7.ps1

# Linux/macOS
./run-phase7.sh
```

### Run specific evaluation steps only

```powershell
# Windows
.\run-phase7.ps1 -Steps "retrieval,counterfactual,aggregate"

# Linux/macOS
./run-phase7.sh --steps retrieval,counterfactual,aggregate
```

### Run full retrieval evaluation with t-tests, LaTeX, and chart

```powershell
python -c "
from countercase.evaluation.eval_harness import EvalHarness
h = EvalHarness()
ts = h.load_test_set('countercase/evaluation/data/test_set.json')
h.run_full_evaluation(ts, output_dir='countercase/evaluation/results')
"
```

Output: `countercase/evaluation/results/retrieval_eval.json`, `retrieval_table.tex`, `retrieval_chart.png`

### Run counterfactual module evaluation

```powershell
python -m countercase.evaluation.counterfactual_eval --eval-set countercase/evaluation/data/cf_eval_set.json --output countercase/evaluation/results/counterfactual_eval.json
```

### Run explanation faithfulness evaluation

```powershell
python -m countercase.evaluation.explanation_eval faithfulness --input countercase/evaluation/data/explanations.json --output countercase/evaluation/results/faithfulness.json
```

### Generate human evaluation CSV template

```powershell
python -m countercase.evaluation.explanation_eval gen-template --input countercase/evaluation/data/explanations.json --output countercase/evaluation/results/human_eval_template.csv
```

### Score filled human evaluation CSV

```powershell
python -m countercase.evaluation.explanation_eval score-human --csv countercase/evaluation/results/human_eval_filled.csv --output countercase/evaluation/results/human_eval_report.json
```

### Create test set from citation extraction

```powershell
python -m countercase.evaluation.create_test_set citations --case-texts countercase/data/case_texts.json --citation-index countercase/data/citation_index.json --output countercase/evaluation/data/citation_test_set.json
```

### Merge multiple test sets

```powershell
python -m countercase.evaluation.create_test_set merge countercase/evaluation/data/test_set.json countercase/evaluation/data/citation_test_set.json --output countercase/evaluation/data/merged.json
```

### Split test set by year (train/test)

```powershell
python -m countercase.evaluation.create_test_set split countercase/evaluation/data/merged.json --cutoff 2020
```

### Aggregate all evaluation results

```powershell
python -m countercase.evaluation.aggregate_results --results-dir countercase/evaluation/results
```

Output: `countercase/evaluation/results/unified_summary.json`, `all_tables.tex`, `evaluation_summary.png`