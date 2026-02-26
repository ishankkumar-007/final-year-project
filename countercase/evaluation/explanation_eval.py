"""Explanation quality evaluation (Phase 7.3).

Provides two evaluation axes for explanations:

1. **Faithfulness** — every sentence in a per-result explanation must
   be traceable to a span in the retrieved chunk text or its metadata.
   Target: > 90 % faithful sentences.

2. **Human evaluation template** — generates a CSV template that legal
   experts fill in with Likert-scale ratings for clarity, legal
   accuracy, and usefulness.

Faithfulness is computed automatically; human eval is exported as a
CSV for offline annotation that is then loaded back for scoring.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Faithfulness evaluation
# -------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple regex-based)."""
    sentences = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _sentence_is_grounded(
    sentence: str,
    sources: list[str],
    overlap_threshold: float = 0.4,
) -> bool:
    """Check if a sentence is grounded in at least one source text.

    A sentence is considered grounded if it has ≥ *overlap_threshold*
    fraction of its content words present in at least one source.
    This is a lightweight proxy for entailment checking without
    requiring an NLI model.

    Args:
        sentence: The explanation sentence.
        sources: List of source texts (chunk text or metadata strings).
        overlap_threshold: Minimum word overlap fraction.

    Returns:
        ``True`` if the sentence is grounded.
    """
    # Tokenise explanation sentence
    sent_tokens = set(re.findall(r"[a-z][a-z0-9]+", sentence.lower()))
    if not sent_tokens:
        return True  # empty / non-informative → vacuously faithful

    for src in sources:
        src_tokens = set(re.findall(r"[a-z][a-z0-9]+", src.lower()))
        overlap = len(sent_tokens & src_tokens)
        if overlap / len(sent_tokens) >= overlap_threshold:
            return True
    return False


def compute_faithfulness(
    explanation: str,
    source_texts: list[str],
    overlap_threshold: float = 0.4,
) -> dict[str, Any]:
    """Compute faithfulness score for a single explanation.

    Args:
        explanation: The full explanation text.
        source_texts: Chunk texts and metadata strings the explanation
            should be grounded in.
        overlap_threshold: Word-overlap fraction for grounding.

    Returns:
        Dict with ``n_sentences``, ``n_faithful``, ``score``, and
        ``unfaithful_sentences`` (list of failing sentences).
    """
    sentences = _split_sentences(explanation)
    n = len(sentences)
    faithful_count = 0
    unfaithful: list[str] = []

    for sent in sentences:
        if _sentence_is_grounded(sent, source_texts, overlap_threshold):
            faithful_count += 1
        else:
            unfaithful.append(sent)

    score = faithful_count / n if n else 1.0
    return {
        "n_sentences": n,
        "n_faithful": faithful_count,
        "score": round(score, 4),
        "unfaithful_sentences": unfaithful,
    }


def evaluate_faithfulness_batch(
    explanations: list[dict[str, Any]],
    overlap_threshold: float = 0.4,
) -> dict[str, Any]:
    """Evaluate faithfulness over a batch of explanations.

    Each item in *explanations* must have:

    - ``explanation``: the explanation text.
    - ``source_texts``: list of source strings.
    - ``id`` (optional): identifier for the explanation.

    Returns:
        Dict with ``n_explanations``, ``mean_score``, ``per_explanation``
        list.
    """
    results: list[dict[str, Any]] = []
    for item in explanations:
        eid = item.get("id", f"exp_{len(results)}")
        r = compute_faithfulness(
            explanation=item["explanation"],
            source_texts=item["source_texts"],
            overlap_threshold=overlap_threshold,
        )
        r["id"] = eid
        results.append(r)

    scores = [r["score"] for r in results]
    mean = sum(scores) / len(scores) if scores else 0.0

    return {
        "n_explanations": len(results),
        "mean_score": round(mean, 4),
        "per_explanation": results,
    }


# -------------------------------------------------------------------
# Human evaluation CSV template generation
# -------------------------------------------------------------------

_CSV_HEADER = [
    "case_id",
    "chunk_id",
    "explanation_text",
    "source_chunk_text",
    "clarity_1_to_5",
    "legal_accuracy_1_to_5",
    "usefulness_1_to_5",
    "annotator_id",
    "notes",
]


def generate_human_eval_template(
    explanations: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write a CSV template for human evaluation.

    Each row contains the explanation and its source text.  The
    annotator fills in the Likert-scale columns.

    Args:
        explanations: List of dicts with ``case_id``, ``chunk_id``,
            ``explanation``, and ``source_chunk_text``.
        output_path: File path for the CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        for item in explanations:
            writer.writerow([
                item.get("case_id", ""),
                item.get("chunk_id", ""),
                item.get("explanation", ""),
                item.get("source_chunk_text", ""),
                "",  # clarity
                "",  # accuracy
                "",  # usefulness
                "",  # annotator
                "",  # notes
            ])

    logger.info(
        "Human eval template with %d rows written to %s",
        len(explanations), output_path,
    )


# -------------------------------------------------------------------
# Human evaluation scoring (load filled CSV)
# -------------------------------------------------------------------

@dataclass
class HumanEvalReport:
    """Aggregated human evaluation results."""

    n_rows: int = 0
    mean_clarity: float = 0.0
    mean_accuracy: float = 0.0
    mean_usefulness: float = 0.0
    per_annotator: dict[str, dict[str, float]] = field(default_factory=dict)


def score_human_eval(csv_path: str | Path) -> HumanEvalReport:
    """Load a filled human evaluation CSV and compute aggregate scores.

    Empty or non-numeric rating cells are skipped.

    Args:
        csv_path: Path to the filled CSV file.

    Returns:
        :class:`HumanEvalReport` with mean Likert scores.
    """
    csv_path = Path(csv_path)
    report = HumanEvalReport()

    clarity_vals: list[float] = []
    accuracy_vals: list[float] = []
    usefulness_vals: list[float] = []
    annotator_scores: dict[str, dict[str, list[float]]] = {}

    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            report.n_rows += 1
            annotator = row.get("annotator_id", "unknown").strip()

            for col, store in (
                ("clarity_1_to_5", clarity_vals),
                ("legal_accuracy_1_to_5", accuracy_vals),
                ("usefulness_1_to_5", usefulness_vals),
            ):
                raw = row.get(col, "").strip()
                if raw:
                    try:
                        val = float(raw)
                        store.append(val)
                        # Per-annotator tracking
                        annotator_scores.setdefault(
                            annotator, {"clarity": [], "accuracy": [], "usefulness": []}
                        )
                        short = col.split("_")[0]
                        if short == "clarity":
                            annotator_scores[annotator]["clarity"].append(val)
                        elif short == "legal":
                            annotator_scores[annotator]["accuracy"].append(val)
                        elif short == "usefulness":
                            annotator_scores[annotator]["usefulness"].append(val)
                    except ValueError:
                        pass

    report.mean_clarity = round(sum(clarity_vals) / len(clarity_vals), 2) if clarity_vals else 0.0
    report.mean_accuracy = round(sum(accuracy_vals) / len(accuracy_vals), 2) if accuracy_vals else 0.0
    report.mean_usefulness = round(sum(usefulness_vals) / len(usefulness_vals), 2) if usefulness_vals else 0.0

    for ann, scores_dict in annotator_scores.items():
        report.per_annotator[ann] = {}
        for dim_name, dim_vals in scores_dict.items():
            if dim_vals:
                report.per_annotator[ann][f"mean_{dim_name}"] = round(
                    sum(dim_vals) / len(dim_vals), 2
                )

    logger.info(
        "Human eval: %d rows, clarity=%.2f, accuracy=%.2f, usefulness=%.2f",
        report.n_rows, report.mean_clarity, report.mean_accuracy, report.mean_usefulness,
    )
    return report


# -------------------------------------------------------------------
# I/O
# -------------------------------------------------------------------

def save_faithfulness_report(
    report: dict[str, Any],
    path: str | Path,
) -> None:
    """Save faithfulness evaluation report to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Faithfulness report saved to %s", path)


def save_human_eval_report(
    report: HumanEvalReport,
    path: str | Path,
) -> None:
    """Save scored human evaluation report to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "n_rows": report.n_rows,
        "mean_clarity": report.mean_clarity,
        "mean_accuracy": report.mean_accuracy,
        "mean_usefulness": report.mean_usefulness,
        "per_annotator": report.per_annotator,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Human eval report saved to %s", path)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main() -> None:
    """CLI entry point for explanation evaluation."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate COUNTERCASE explanations",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Faithfulness sub-command
    faith_parser = sub.add_parser("faithfulness", help="Run faithfulness evaluation")
    faith_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="JSON file with explanations and source texts",
    )
    faith_parser.add_argument(
        "--output",
        type=str,
        default="countercase/evaluation/results/faithfulness.json",
    )
    faith_parser.add_argument("--threshold", type=float, default=0.4)

    # Human eval template generation
    tmpl_parser = sub.add_parser("gen-template", help="Generate human eval CSV template")
    tmpl_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="JSON with explanation items",
    )
    tmpl_parser.add_argument(
        "--output",
        type=str,
        default="countercase/evaluation/results/human_eval_template.csv",
    )

    # Human eval scoring
    score_parser = sub.add_parser("score-human", help="Score filled human eval CSV")
    score_parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Filled human evaluation CSV",
    )
    score_parser.add_argument(
        "--output",
        type=str,
        default="countercase/evaluation/results/human_eval_report.json",
    )

    args = parser.parse_args()

    if args.command == "faithfulness":
        with open(args.input, encoding="utf-8") as fh:
            data = json.load(fh)
        report = evaluate_faithfulness_batch(data, overlap_threshold=args.threshold)
        save_faithfulness_report(report, args.output)
        print(f"\nFaithfulness: {report['mean_score']:.4f} "
              f"({report['n_explanations']} explanations)")

    elif args.command == "gen-template":
        with open(args.input, encoding="utf-8") as fh:
            data = json.load(fh)
        generate_human_eval_template(data, args.output)
        print(f"Template written to {args.output}")

    elif args.command == "score-human":
        report = score_human_eval(args.csv)
        save_human_eval_report(report, args.output)
        print(f"\nHuman eval: clarity={report.mean_clarity}, "
              f"accuracy={report.mean_accuracy}, "
              f"usefulness={report.mean_usefulness}")


if __name__ == "__main__":
    main()
