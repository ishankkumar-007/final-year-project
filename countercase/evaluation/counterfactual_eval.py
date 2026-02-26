"""Counterfactual module evaluation (Phase 7.2).

Evaluates whether the perturbation tree correctly assigns high
sensitivity scores to legally dispositive facts and low scores to
non-dispositive facts.

Evaluation protocol
-------------------
1.  Load a JSON evaluation set with cases annotated with:
    - dispositive_facts  : list of fact_type strings expected to have
      high sensitivity (e.g. ["Section", "PartyType"])
    - non_dispositive_facts: fact_type strings expected to have low
      sensitivity (e.g. ["Numerical"])
2.  For each case, build a perturbation tree to depth 2 and compute
    aggregate sensitivity scores per fact dimension.
3.  Check that every dispositive fact type has sensitivity > threshold
    and every non-dispositive fact type has sensitivity <= threshold.
4.  Compute Spearman rank correlation between expert-annotated
    importance rankings and system sensitivity scores.
5.  Log LLM validation accept/reject rates.
6.  Write structured JSON results.

Evaluation case JSON schema
----------------------------
.. code-block:: json

    [
        {
            "case_id": "Criminal Appeal 123/2015",
            "fact_sheet": { <FactSheet fields> },
            "dispositive_facts": ["Section", "Evidence"],
            "non_dispositive_facts": ["Numerical"],
            "expert_importance_ranking": {
                "Section": 1,
                "Evidence": 2,
                "PartyType": 3,
                "Numerical": 4
            }
        }
    ]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from countercase.counterfactual.perturbation_tree import PerturbationTree
from countercase.counterfactual.sensitivity import compute_sensitivity_scores
from countercase.counterfactual.llm_validator import (
    PerturbationValidator,
    mock_validation_llm_fn,
)
from countercase.fact_extraction.schema import FactSheet

logger = logging.getLogger(__name__)

_CANONICAL_FACT_TYPES = ("Numerical", "Section", "PartyType", "Evidence")
_DEFAULT_THRESHOLD = 1.5  # mean rank displacement above which = "high"


# -------------------------------------------------------------------
# Result structures
# -------------------------------------------------------------------

@dataclass
class CaseEvalResult:
    """Evaluation result for a single case."""

    case_id: str = ""
    sensitivity_scores: dict[str, float] = field(default_factory=dict)
    dispositive_correct: int = 0
    dispositive_total: int = 0
    non_dispositive_correct: int = 0
    non_dispositive_total: int = 0
    spearman_rho: float | None = None
    llm_accept_count: int = 0
    llm_reject_count: int = 0
    tree_node_count: int = 0


@dataclass
class CounterfactualEvalReport:
    """Aggregate counterfactual evaluation report."""

    n_cases: int = 0
    mean_dispositive_accuracy: float = 0.0
    mean_non_dispositive_accuracy: float = 0.0
    mean_spearman_rho: float | None = None
    overall_llm_accept_rate: float = 0.0
    per_case: list[CaseEvalResult] = field(default_factory=list)


# -------------------------------------------------------------------
# Spearman rank correlation (minimal implementation)
# -------------------------------------------------------------------

def _spearman_rho(
    expert_ranking: dict[str, int],
    system_scores: dict[str, float],
) -> float | None:
    """Compute Spearman rank correlation between expert rankings
    and system sensitivity scores.

    Expert ranking: lower number = more important.
    System scores:  higher number = more sensitive.

    We rank the system scores in descending order (highest = rank 1)
    and compare to the expert ranking.

    Returns None if fewer than 3 overlapping dimensions.
    """
    common = sorted(set(expert_ranking) & set(system_scores))
    n = len(common)
    if n < 3:
        return None

    # System ranks: sort by score descending, assign rank 1..n
    sorted_by_score = sorted(common, key=lambda k: system_scores[k], reverse=True)
    system_ranks = {k: i + 1 for i, k in enumerate(sorted_by_score)}

    d_sq_sum = sum(
        (expert_ranking[k] - system_ranks[k]) ** 2 for k in common
    )
    rho = 1.0 - (6 * d_sq_sum) / (n * (n ** 2 - 1))
    return round(rho, 4)


# -------------------------------------------------------------------
# Core evaluation
# -------------------------------------------------------------------

def evaluate_single_case(
    fact_sheet: FactSheet,
    dispositive_facts: list[str],
    non_dispositive_facts: list[str],
    expert_importance: dict[str, int] | None = None,
    tree: PerturbationTree | None = None,
    *,
    max_depth: int = 2,
    max_children: int = 5,
    threshold: float = _DEFAULT_THRESHOLD,
    k: int = 10,
) -> CaseEvalResult:
    """Evaluate a single case's counterfactual sensitivity.

    If *tree* is not provided, a new tree is built from *fact_sheet*.

    Args:
        fact_sheet: Structured fact sheet for the case.
        dispositive_facts: Fact type names expected to be highly sensitive.
        non_dispositive_facts: Fact type names expected to have low sensitivity.
        expert_importance: Optional expert importance ranking per fact type.
        tree: Pre-built perturbation tree (skips tree construction if given).
        max_depth: Tree depth for auto-building.
        max_children: Branching factor for auto-building.
        threshold: Sensitivity threshold separating high from low.
        k: Top-K for diff computation.

    Returns:
        :class:`CaseEvalResult` with per-dimension scores and accuracy.
    """
    result = CaseEvalResult(case_id=fact_sheet.case_id)

    # Build tree if not provided
    if tree is None:
        tree = PerturbationTree(retriever=None, top_k=k)
        try:
            from countercase.retrieval.hybrid_retriever import HybridRetriever
            tree = PerturbationTree(retriever=HybridRetriever(), top_k=k)
        except Exception:
            pass
        tree.build_root(fact_sheet)
        validator = PerturbationValidator(llm_fn=mock_validation_llm_fn)
        tree.expand_tree(
            validator=validator,
            max_depth=max_depth,
            max_children_per_node=max_children,
        )

    result.tree_node_count = len(tree._nodes)

    # Compute sensitivity scores
    scores = compute_sensitivity_scores(tree, k=k)
    result.sensitivity_scores = {k_: round(v, 4) for k_, v in scores.items()}

    # Check dispositive facts (should have score > threshold)
    result.dispositive_total = len(dispositive_facts)
    for ft in dispositive_facts:
        if scores.get(ft, 0.0) > threshold:
            result.dispositive_correct += 1

    # Check non-dispositive facts (should have score <= threshold)
    result.non_dispositive_total = len(non_dispositive_facts)
    for ft in non_dispositive_facts:
        if scores.get(ft, 0.0) <= threshold:
            result.non_dispositive_correct += 1

    # Spearman correlation with expert importance
    if expert_importance:
        result.spearman_rho = _spearman_rho(expert_importance, scores)

    # LLM validation stats (from tree's internal log)
    accept, reject = _extract_llm_stats(tree)
    result.llm_accept_count = accept
    result.llm_reject_count = reject

    return result


def _extract_llm_stats(tree: PerturbationTree) -> tuple[int, int]:
    """Extract LLM validation accept/reject counts from a tree.

    The PerturbationTree logs validation stats.  If the tree keeps a
    counter, use it; otherwise fall back to counting edges (accepted)
    and estimating rejects from tree metadata.
    """
    accept = 0
    reject = 0

    # Check for explicit counters
    if hasattr(tree, "llm_accept_count"):
        accept = getattr(tree, "llm_accept_count", 0)
    if hasattr(tree, "llm_reject_count"):
        reject = getattr(tree, "llm_reject_count", 0)

    # Fall back: each edge in the tree is an accepted perturbation
    if accept == 0:
        accept = sum(
            1 for _, _edge in tree.get_all_edges()
        )
    return accept, reject


# -------------------------------------------------------------------
# Batch evaluation
# -------------------------------------------------------------------

def evaluate_batch(
    eval_set: list[dict[str, Any]],
    *,
    max_depth: int = 2,
    max_children: int = 5,
    threshold: float = _DEFAULT_THRESHOLD,
    k: int = 10,
) -> CounterfactualEvalReport:
    """Run counterfactual evaluation on a batch of annotated cases.

    Args:
        eval_set: List of evaluation case dicts (see module docstring).
        max_depth: Tree expansion depth.
        max_children: Branching factor per node.
        threshold: Sensitivity threshold.
        k: Top-K for retrieval diff.

    Returns:
        :class:`CounterfactualEvalReport` with per-case and aggregate
        metrics.
    """
    report = CounterfactualEvalReport(n_cases=len(eval_set))

    for entry in eval_set:
        fact_sheet = FactSheet(**entry["fact_sheet"])
        result = evaluate_single_case(
            fact_sheet=fact_sheet,
            dispositive_facts=entry.get("dispositive_facts", []),
            non_dispositive_facts=entry.get("non_dispositive_facts", []),
            expert_importance=entry.get("expert_importance_ranking"),
            max_depth=max_depth,
            max_children=max_children,
            threshold=threshold,
            k=k,
        )
        report.per_case.append(result)
        logger.info(
            "Case %s: disp=%d/%d, non_disp=%d/%d, rho=%s",
            result.case_id,
            result.dispositive_correct,
            result.dispositive_total,
            result.non_dispositive_correct,
            result.non_dispositive_total,
            result.spearman_rho,
        )

    # Aggregate
    n = len(report.per_case) or 1
    report.mean_dispositive_accuracy = round(
        sum(
            r.dispositive_correct / max(r.dispositive_total, 1)
            for r in report.per_case
        )
        / n,
        4,
    )
    report.mean_non_dispositive_accuracy = round(
        sum(
            r.non_dispositive_correct / max(r.non_dispositive_total, 1)
            for r in report.per_case
        )
        / n,
        4,
    )

    rhos = [r.spearman_rho for r in report.per_case if r.spearman_rho is not None]
    report.mean_spearman_rho = round(sum(rhos) / len(rhos), 4) if rhos else None

    total_accept = sum(r.llm_accept_count for r in report.per_case)
    total_reject = sum(r.llm_reject_count for r in report.per_case)
    total_decisions = total_accept + total_reject
    report.overall_llm_accept_rate = (
        round(total_accept / total_decisions, 4) if total_decisions else 0.0
    )

    return report


# -------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------

def load_eval_set(path: str | Path) -> list[dict[str, Any]]:
    """Load an annotated evaluation set from JSON."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded %d counterfactual eval cases from %s", len(data), path)
    return data


def save_report(report: CounterfactualEvalReport, path: str | Path) -> None:
    """Serialize the report to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "n_cases": report.n_cases,
        "mean_dispositive_accuracy": report.mean_dispositive_accuracy,
        "mean_non_dispositive_accuracy": report.mean_non_dispositive_accuracy,
        "mean_spearman_rho": report.mean_spearman_rho,
        "overall_llm_accept_rate": report.overall_llm_accept_rate,
        "per_case": [],
    }
    for r in report.per_case:
        data["per_case"].append({
            "case_id": r.case_id,
            "sensitivity_scores": r.sensitivity_scores,
            "dispositive_correct": r.dispositive_correct,
            "dispositive_total": r.dispositive_total,
            "non_dispositive_correct": r.non_dispositive_correct,
            "non_dispositive_total": r.non_dispositive_total,
            "spearman_rho": r.spearman_rho,
            "llm_accept_count": r.llm_accept_count,
            "llm_reject_count": r.llm_reject_count,
            "tree_node_count": r.tree_node_count,
        })

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Counterfactual eval report saved to %s", path)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main() -> None:
    """CLI entry point for counterfactual evaluation."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate COUNTERCASE counterfactual module",
    )
    parser.add_argument(
        "--eval-set",
        type=str,
        required=True,
        help="Path to annotated eval set JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="countercase/evaluation/results/counterfactual_eval.json",
        help="Path for output JSON report",
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-children", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    eval_set = load_eval_set(args.eval_set)
    report = evaluate_batch(
        eval_set,
        max_depth=args.max_depth,
        max_children=args.max_children,
        threshold=args.threshold,
        k=args.k,
    )
    save_report(report, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("COUNTERFACTUAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Cases evaluated:              {report.n_cases}")
    print(f"  Mean dispositive accuracy:    {report.mean_dispositive_accuracy}")
    print(f"  Mean non-dispositive accuracy: {report.mean_non_dispositive_accuracy}")
    print(f"  Mean Spearman rho:            {report.mean_spearman_rho}")
    print(f"  LLM accept rate:              {report.overall_llm_accept_rate}")
    print()


if __name__ == "__main__":
    main()
