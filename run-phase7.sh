#!/usr/bin/env bash
# COUNTERCASE Phase 7 -- Evaluation, Ablation, and Research Writeup
#
# Usage:
#   ./run-phase7.sh                                    # run all steps
#   ./run-phase7.sh --steps retrieval,counterfactual   # run specific steps
#   ./run-phase7.sh --test-set path/to/test.json       # custom test set
#
# Valid steps: retrieval, counterfactual, faithfulness, human-template,
#              create-testset, merge-testset, split-testset, aggregate

set -euo pipefail

# --- Defaults ---
STEPS="all"
TEST_SET="countercase/evaluation/data/test_set.json"
OUTPUT_DIR="countercase/evaluation/results"
CASE_TEXTS="countercase/data/case_texts.json"
CITATION_INDEX="countercase/data/citation_index.json"
CUTOFF=2020

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)        STEPS="$2"; shift 2 ;;
        --test-set)     TEST_SET="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --case-texts)   CASE_TEXTS="$2"; shift 2 ;;
        --citation-index) CITATION_INDEX="$2"; shift 2 ;;
        --cutoff)       CUTOFF="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--steps STEPS] [--test-set PATH] [--output-dir DIR]"
            echo "       [--case-texts PATH] [--citation-index PATH] [--cutoff YEAR]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Resolve steps ---
ALL_STEPS=(retrieval counterfactual faithfulness human-template create-testset merge-testset split-testset aggregate)
if [[ "$STEPS" == "all" ]]; then
    RUN_STEPS=("${ALL_STEPS[@]}")
else
    IFS=',' read -ra RUN_STEPS <<< "$STEPS"
fi

echo "========================================"
echo "  COUNTERCASE Phase 7 Runner"
echo "========================================"
echo "Steps    : ${RUN_STEPS[*]}"
echo "Test Set : $TEST_SET"
echo "Output   : $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

PASSED=0
FAILED=0
START_TIME=$(date +%s)

run_step() {
    local name="$1"
    shift
    echo "--- [$name] ---"
    local step_start=$(date +%s)
    if "$@"; then
        local step_end=$(date +%s)
        echo "  OK ($((step_end - step_start))s)"
        PASSED=$((PASSED + 1))
    else
        local step_end=$(date +%s)
        echo "  FAILED ($((step_end - step_start))s)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

contains() {
    local needle="$1"
    shift
    for item in "$@"; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

# --- 1. Retrieval evaluation ---
if contains "retrieval" "${RUN_STEPS[@]}"; then
    run_step "Retrieval Evaluation" python -c "
from countercase.evaluation.eval_harness import EvalHarness
h = EvalHarness()
ts = h.load_test_set('${TEST_SET}')
h.run_full_evaluation(ts, output_dir='${OUTPUT_DIR}')
"
fi

# --- 2. Counterfactual evaluation ---
if contains "counterfactual" "${RUN_STEPS[@]}"; then
    run_step "Counterfactual Evaluation" python -m countercase.evaluation.counterfactual_eval \
        --eval-set countercase/evaluation/data/cf_eval_set.json \
        --output "$OUTPUT_DIR/counterfactual_eval.json"
fi

# --- 3. Explanation faithfulness ---
if contains "faithfulness" "${RUN_STEPS[@]}"; then
    run_step "Explanation Faithfulness" python -m countercase.evaluation.explanation_eval faithfulness \
        --input countercase/evaluation/data/explanations.json \
        --output "$OUTPUT_DIR/faithfulness.json"
fi

# --- 4. Human evaluation template ---
if contains "human-template" "${RUN_STEPS[@]}"; then
    run_step "Human Eval Template" python -m countercase.evaluation.explanation_eval gen-template \
        --input countercase/evaluation/data/explanations.json \
        --output "$OUTPUT_DIR/human_eval_template.csv"
fi

# --- 5. Create test set from citations ---
if contains "create-testset" "${RUN_STEPS[@]}"; then
    CREATE_ARGS=(-m countercase.evaluation.create_test_set citations
        --case-texts "$CASE_TEXTS"
        --output countercase/evaluation/data/citation_test_set.json)
    if [[ -f "$CITATION_INDEX" ]]; then
        CREATE_ARGS+=(--citation-index "$CITATION_INDEX")
    fi
    run_step "Create Citation Test Set" python "${CREATE_ARGS[@]}"
fi

# --- 6. Merge test sets ---
if contains "merge-testset" "${RUN_STEPS[@]}"; then
    run_step "Merge Test Sets" python -m countercase.evaluation.create_test_set merge \
        countercase/evaluation/data/test_set.json \
        countercase/evaluation/data/citation_test_set.json \
        --output countercase/evaluation/data/merged.json
fi

# --- 7. Split test set by year ---
if contains "split-testset" "${RUN_STEPS[@]}"; then
    run_step "Split Test Set (cutoff=$CUTOFF)" python -m countercase.evaluation.create_test_set split \
        countercase/evaluation/data/merged.json \
        --cutoff "$CUTOFF"
fi

# --- 8. Aggregate all results ---
if contains "aggregate" "${RUN_STEPS[@]}"; then
    run_step "Aggregate Results" python -m countercase.evaluation.aggregate_results \
        --results-dir "$OUTPUT_DIR"
fi

# --- Summary ---
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "========================================"
echo "  Phase 7 Complete"
echo "  Passed: $PASSED  Failed: $FAILED"
echo "  Total time: ${ELAPSED}s"
echo "========================================"

echo ""
echo "Output files:"
ls -lhS "$OUTPUT_DIR"/ 2>/dev/null || echo "  (none)"

[[ $FAILED -gt 0 ]] && exit 1
exit 0
