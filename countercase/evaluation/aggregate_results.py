"""Results aggregation and research output (Phase 7.5).

Combines outputs from:
    - Retrieval evaluation (eval_harness.py)
    - Counterfactual evaluation (counterfactual_eval.py)
    - Explanation evaluation (explanation_eval.py)

Produces:
    - Unified JSON summary
    - LaTeX tables for the research writeup
    - Comparison charts (PNG)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[0]
_DEFAULT_RESULTS_DIR = _ROOT / "results"


# -------------------------------------------------------------------
# Unified summary
# -------------------------------------------------------------------

@dataclass
class UnifiedSummary:
    """Aggregated evaluation summary across all modules."""

    retrieval: dict[str, Any] = field(default_factory=dict)
    counterfactual: dict[str, Any] = field(default_factory=dict)
    explanation_faithfulness: dict[str, Any] = field(default_factory=dict)
    explanation_human: dict[str, Any] = field(default_factory=dict)


def load_retrieval_report(path: str | Path) -> dict[str, Any]:
    """Load retrieval evaluation JSON."""
    path = Path(path)
    if not path.exists():
        logger.warning("Retrieval report not found: %s", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_counterfactual_report(path: str | Path) -> dict[str, Any]:
    """Load counterfactual evaluation JSON."""
    path = Path(path)
    if not path.exists():
        logger.warning("Counterfactual report not found: %s", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_faithfulness_report(path: str | Path) -> dict[str, Any]:
    """Load faithfulness evaluation JSON."""
    path = Path(path)
    if not path.exists():
        logger.warning("Faithfulness report not found: %s", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_human_eval_report(path: str | Path) -> dict[str, Any]:
    """Load human evaluation report JSON."""
    path = Path(path)
    if not path.exists():
        logger.warning("Human eval report not found: %s", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def aggregate(
    results_dir: str | Path = _DEFAULT_RESULTS_DIR,
) -> UnifiedSummary:
    """Load all evaluation reports and combine into one summary.

    Expects the following files in *results_dir*:
        - retrieval_eval.json
        - counterfactual_eval.json
        - faithfulness.json
        - human_eval_report.json

    Missing files are tolerated with a warning.
    """
    d = Path(results_dir)
    summary = UnifiedSummary(
        retrieval=load_retrieval_report(d / "retrieval_eval.json"),
        counterfactual=load_counterfactual_report(d / "counterfactual_eval.json"),
        explanation_faithfulness=load_faithfulness_report(d / "faithfulness.json"),
        explanation_human=load_human_eval_report(d / "human_eval_report.json"),
    )
    return summary


# -------------------------------------------------------------------
# LaTeX table generation
# -------------------------------------------------------------------

def _escape_latex(s: str) -> str:
    """Escape underscores for LaTeX."""
    return s.replace("_", r"\_")


def generate_retrieval_table(retrieval: dict[str, Any]) -> str:
    """Generate a LaTeX table summarising retrieval ablation results."""
    modes = retrieval.get("modes", {})
    if not modes:
        return "% No retrieval results available."

    metric_keys = [
        "mean_mrr_10", "mean_ndcg_10",
        "mean_recall_5", "mean_recall_10", "mean_recall_20",
    ]
    headers = ["MRR@10", "NDCG@10", "R@5", "R@10", "R@20"]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Retrieval evaluation across ablation modes.}",
        r"\label{tab:retrieval_ablation}",
        r"\begin{tabular}{l" + "c" * len(headers) + "}",
        r"\toprule",
        "Mode & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]

    for mode, metrics in modes.items():
        vals = [f"{metrics.get(m, 0.0):.4f}" for m in metric_keys]
        lines.append(_escape_latex(mode) + " & " + " & ".join(vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_counterfactual_table(cf: dict[str, Any]) -> str:
    """Generate a LaTeX table summarising counterfactual evaluation."""
    if not cf:
        return "% No counterfactual results available."

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Counterfactual module evaluation.}",
        r"\label{tab:counterfactual_eval}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
    ]

    metrics = [
        ("Cases evaluated", str(cf.get("n_cases", 0))),
        ("Dispositive fact accuracy", f"{cf.get('mean_dispositive_accuracy', 0.0):.4f}"),
        ("Non-dispositive accuracy", f"{cf.get('mean_non_dispositive_accuracy', 0.0):.4f}"),
        ("Mean Spearman $\\rho$", f"{cf.get('mean_spearman_rho', 0.0) or 0.0:.4f}"),
        ("LLM accept rate", f"{cf.get('overall_llm_accept_rate', 0.0):.4f}"),
    ]

    for name, val in metrics:
        lines.append(f"{name} & {val}" + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_explanation_table(
    faithfulness: dict[str, Any],
    human_eval: dict[str, Any],
) -> str:
    """Generate a LaTeX table for explanation quality."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Explanation quality evaluation.}",
        r"\label{tab:explanation_eval}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
    ]

    rows: list[tuple[str, str]] = []

    if faithfulness:
        rows.append(("Explanations evaluated", str(faithfulness.get("n_explanations", 0))))
        rows.append(("Faithfulness score", f"{faithfulness.get('mean_score', 0.0):.4f}"))

    if human_eval:
        rows.append(("Human: clarity (1-5)", f"{human_eval.get('mean_clarity', 0.0):.2f}"))
        rows.append(("Human: accuracy (1-5)", f"{human_eval.get('mean_accuracy', 0.0):.2f}"))
        rows.append(("Human: usefulness (1-5)", f"{human_eval.get('mean_usefulness', 0.0):.2f}"))

    if not rows:
        rows.append(("(no data)", "---"))

    for name, val in rows:
        lines.append(f"{name} & {val}" + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_all_tables(summary: UnifiedSummary) -> str:
    """Generate all LaTeX tables concatenated."""
    parts = [
        generate_retrieval_table(summary.retrieval),
        "",
        generate_counterfactual_table(summary.counterfactual),
        "",
        generate_explanation_table(
            summary.explanation_faithfulness,
            summary.explanation_human,
        ),
    ]
    return "\n".join(parts)


# -------------------------------------------------------------------
# Chart generation
# -------------------------------------------------------------------

def generate_summary_chart(
    summary: UnifiedSummary,
    output_path: str | Path,
) -> None:
    """Generate a summary bar chart covering key metrics.

    Shows retrieval MRR@10 per mode, counterfactual accuracy, and
    faithfulness score as a grouped bar chart.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib/numpy not available; skipping chart.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: Retrieval MRR@10 by mode ---
    modes = summary.retrieval.get("modes", {})
    if modes:
        mode_names = list(modes.keys())
        mrr_vals = [modes[m].get("mean_mrr_10", 0.0) for m in mode_names]
        short_names = [m.replace("_", "\n") for m in mode_names]
        bars = axes[0].bar(range(len(mode_names)), mrr_vals, color="#4C72B0")
        axes[0].set_xticks(range(len(mode_names)))
        axes[0].set_xticklabels(short_names, fontsize=8)
        axes[0].set_ylabel("MRR@10")
        axes[0].set_title("Retrieval: MRR@10 by Mode")
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(axis="y", alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center")
        axes[0].set_title("Retrieval: MRR@10")

    # --- Panel 2: Counterfactual accuracy ---
    cf = summary.counterfactual
    if cf:
        labels = ["Dispositive\nAccuracy", "Non-Disp.\nAccuracy", "Spearman rho"]
        vals = [
            cf.get("mean_dispositive_accuracy", 0.0),
            cf.get("mean_non_dispositive_accuracy", 0.0),
            cf.get("mean_spearman_rho", 0.0) or 0.0,
        ]
        colours = ["#55A868", "#C44E52", "#8172B2"]
        axes[1].bar(range(len(labels)), vals, color=colours)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, fontsize=9)
        axes[1].set_ylabel("Score")
        axes[1].set_title("Counterfactual Evaluation")
        axes[1].set_ylim(-0.2, 1.05)
        axes[1].grid(axis="y", alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center")
        axes[1].set_title("Counterfactual")

    # --- Panel 3: Explanation scores ---
    faith = summary.explanation_faithfulness
    human = summary.explanation_human
    exp_labels: list[str] = []
    exp_vals: list[float] = []
    exp_colors: list[str] = []

    if faith:
        exp_labels.append("Faithfulness")
        exp_vals.append(faith.get("mean_score", 0.0))
        exp_colors.append("#DD8452")

    if human:
        for key, label, color in [  # noqa: E501
            ("mean_clarity", "Clarity", "#4C72B0"),
            ("mean_accuracy", "Accuracy", "#55A868"),
            ("mean_usefulness", "Usefulness", "#C44E52"),
        ]:
            exp_labels.append(label)
            exp_vals.append(human.get(key, 0.0) / 5.0)  # normalise to 0-1
            exp_colors.append(color)

    if exp_labels:
        axes[2].bar(range(len(exp_labels)), exp_vals, color=exp_colors)
        axes[2].set_xticks(range(len(exp_labels)))
        axes[2].set_xticklabels(exp_labels, fontsize=9)
        axes[2].set_ylabel("Score (normalised)")
        axes[2].set_title("Explanation Quality")
        axes[2].set_ylim(0, 1.05)
        axes[2].grid(axis="y", alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No data", ha="center", va="center")
        axes[2].set_title("Explanation Quality")

    fig.suptitle("COUNTERCASE — Evaluation Summary", fontsize=14, y=1.02)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Summary chart saved to %s", output_path)


# -------------------------------------------------------------------
# Full aggregation pipeline
# -------------------------------------------------------------------

def run_aggregation(
    results_dir: str | Path = _DEFAULT_RESULTS_DIR,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full aggregation: load, combine, generate outputs.

    Args:
        results_dir: Directory containing individual eval reports.
        output_dir: Directory for aggregated outputs.  Defaults to
            *results_dir*.

    Returns:
        Dict with file paths of generated outputs.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    summary = aggregate(results_dir)

    # Unified JSON
    unified_path = output_dir / "unified_summary.json"
    unified_data = {
        "retrieval": summary.retrieval,
        "counterfactual": summary.counterfactual,
        "explanation_faithfulness": summary.explanation_faithfulness,
        "explanation_human": summary.explanation_human,
    }
    with unified_path.open("w", encoding="utf-8") as fh:
        json.dump(unified_data, fh, indent=2)
    logger.info("Unified summary saved to %s", unified_path)

    # LaTeX tables
    latex = generate_all_tables(summary)
    latex_path = output_dir / "all_tables.tex"
    with latex_path.open("w", encoding="utf-8") as fh:
        fh.write(latex)
    logger.info("LaTeX tables saved to %s", latex_path)

    # Chart
    chart_path = output_dir / "evaluation_summary.png"
    generate_summary_chart(summary, chart_path)

    return {
        "unified_json": str(unified_path),
        "latex_tables": str(latex_path),
        "chart": str(chart_path),
    }


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main() -> None:
    """CLI entry point for results aggregation."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Aggregate COUNTERCASE evaluation results",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(_DEFAULT_RESULTS_DIR),
        help="Directory with evaluation JSON reports",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to --results-dir)",
    )
    args = parser.parse_args()

    paths = run_aggregation(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("AGGREGATION COMPLETE")
    print("=" * 60)
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print()


if __name__ == "__main__":
    main()
