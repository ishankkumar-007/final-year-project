"""Evaluation harness for retrieval quality measurement.

Runs queries from a JSON test set through the HybridRetriever (and
ablation variants), computes MRR@K, NDCG@K, Recall@K, and writes a
structured JSON report.

Test set format
---------------
.. code-block:: json

    [
        {
            "query_case_id": "case_001",
            "query_text": "criminal appeal IPC 302 murder ...",
            "relevant_case_ids": ["case_042", "case_057", "case_123"]
        },
        ...
    ]
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from countercase.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from countercase.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
# Ablation modes
# -----------------------------------------------------------------

ABLATION_MODES = (
    "full_system",
    "dpr_only",
    "chroma_only",
    "hybrid_no_mmr",
    "hybrid_no_reranker",
)


# -----------------------------------------------------------------
# Result structures
# -----------------------------------------------------------------

@dataclass
class QueryResult:
    """Per-query evaluation result."""

    query_case_id: str = ""
    mode: str = ""
    mrr_10: float = 0.0
    ndcg_10: float = 0.0
    recall_5: float = 0.0
    recall_10: float = 0.0
    recall_20: float = 0.0
    retrieved_ids: list[str] = field(default_factory=list)
    latency_s: float = 0.0


@dataclass
class EvalReport:
    """Full evaluation report."""

    n_queries: int = 0
    modes: dict[str, dict[str, float]] = field(default_factory=dict)
    per_query: list[QueryResult] = field(default_factory=list)


# -----------------------------------------------------------------
# Harness
# -----------------------------------------------------------------

class EvalHarness:
    """Evaluation harness orchestrating ablation runs.

    Args:
        retriever: A pre-configured :class:`HybridRetriever`.
        top_k: Number of results per query.
        ks: Recall cutoff values.
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        top_k: int = 10,
        ks: tuple[int, ...] = (5, 10, 20),
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.top_k = top_k
        self.ks = ks

    # -----------------------------------------------------------------
    # Test set loading
    # -----------------------------------------------------------------

    @staticmethod
    def load_test_set(path: str | Path) -> list[dict[str, Any]]:
        """Load a JSON test set file.

        Returns a list of dicts with keys ``query_case_id``,
        ``query_text``, and ``relevant_case_ids``.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Loaded %d test queries from %s", len(data), path)
        return data

    # -----------------------------------------------------------------
    # Single query evaluation
    # -----------------------------------------------------------------

    def _eval_query(
        self,
        query_text: str,
        query_case_id: str,
        relevant_ids: set[str],
        mode: str,
    ) -> QueryResult:
        """Run a single query in the given mode and compute metrics."""
        t0 = time.perf_counter()

        results = self._run_mode(query_text, mode)
        latency = time.perf_counter() - t0

        # Extract case_ids from results for metric computation.
        # Use case_id from metadata; fall back to chunk_id if empty.
        raw_case_ids = [
            r.case_id if r.case_id else r.chunk_id for r in results
        ]

        # Deduplicate: keep only the first occurrence of each case_id.
        # Standard IR eval counts each document once at its highest rank.
        seen: set[str] = set()
        ranked_case_ids: list[str] = []
        for cid in raw_case_ids:
            if cid not in seen:
                seen.add(cid)
                ranked_case_ids.append(cid)

        qr = QueryResult(
            query_case_id=query_case_id,
            mode=mode,
            mrr_10=mrr_at_k(ranked_case_ids, relevant_ids, 10),
            ndcg_10=ndcg_at_k(ranked_case_ids, relevant_ids, 10),
            recall_5=recall_at_k(ranked_case_ids, relevant_ids, 5),
            recall_10=recall_at_k(ranked_case_ids, relevant_ids, 10),
            recall_20=recall_at_k(ranked_case_ids, relevant_ids, 20),
            retrieved_ids=ranked_case_ids[:self.top_k],
            latency_s=round(latency, 4),
        )
        return qr

    def _run_mode(
        self, query_text: str, mode: str
    ) -> list[RetrievalResult]:
        """Dispatch to the correct retrieval method."""
        k = self.top_k
        if mode == "full_system":
            return self.retriever.retrieve(query_text, top_k=k)
        elif mode == "dpr_only":
            return self.retriever.retrieve_dpr_only(query_text, top_k=k)
        elif mode == "chroma_only":
            return self.retriever.retrieve_chroma_only(query_text, top_k=k)
        elif mode == "hybrid_no_mmr":
            return self.retriever.retrieve_hybrid_no_mmr(query_text, top_k=k)
        elif mode == "hybrid_no_reranker":
            return self.retriever.retrieve_hybrid_no_reranker(query_text, top_k=k)
        else:
            raise ValueError(f"Unknown ablation mode: {mode}")

    # -----------------------------------------------------------------
    # Full evaluation
    # -----------------------------------------------------------------

    def run(
        self,
        test_set: list[dict[str, Any]],
        modes: tuple[str, ...] | None = None,
    ) -> EvalReport:
        """Run evaluation across all queries and modes.

        Args:
            test_set: List of test-set entries (see module docstring).
            modes: Ablation modes to evaluate.  Defaults to all.

        Returns:
            :class:`EvalReport` with aggregated and per-query metrics.
        """
        modes = modes or ABLATION_MODES
        report = EvalReport(n_queries=len(test_set))

        per_query_results: list[QueryResult] = []

        for mode in modes:
            mode_results: list[QueryResult] = []
            logger.info("=== Evaluating mode: %s (%d queries) ===", mode, len(test_set))

            for entry in test_set:
                qr = self._eval_query(
                    query_text=entry["query_text"],
                    query_case_id=entry["query_case_id"],
                    relevant_ids=set(entry["relevant_case_ids"]),
                    mode=mode,
                )
                mode_results.append(qr)
                per_query_results.append(qr)

            # Aggregate metrics for this mode
            n = len(mode_results) or 1
            report.modes[mode] = {
                "mean_mrr_10": round(sum(q.mrr_10 for q in mode_results) / n, 4),
                "mean_ndcg_10": round(sum(q.ndcg_10 for q in mode_results) / n, 4),
                "mean_recall_5": round(sum(q.recall_5 for q in mode_results) / n, 4),
                "mean_recall_10": round(sum(q.recall_10 for q in mode_results) / n, 4),
                "mean_recall_20": round(sum(q.recall_20 for q in mode_results) / n, 4),
                "mean_latency_s": round(
                    sum(q.latency_s for q in mode_results) / n, 4
                ),
            }
            logger.info("Mode %s: %s", mode, report.modes[mode])

        report.per_query = per_query_results
        return report

    # -----------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------

    @staticmethod
    def save_report(report: EvalReport, path: str | Path) -> None:
        """Serialise the evaluation report to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "n_queries": report.n_queries,
            "modes": report.modes,
            "per_query": [asdict(q) for q in report.per_query],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("Report saved to %s", path)


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------

def main() -> None:
    """CLI entry point: evaluate from a JSON test set file."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="COUNTERCASE evaluation harness")
    parser.add_argument(
        "--test-set",
        type=str,
        required=True,
        help="Path to the JSON test set file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="countercase/data/eval_report.json",
        help="Path for output JSON report",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results per query",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=ABLATION_MODES,
        default=list(ABLATION_MODES),
        help="Ablation modes to evaluate",
    )
    args = parser.parse_args()

    harness = EvalHarness(top_k=args.top_k)
    test_set = EvalHarness.load_test_set(args.test_set)
    report = harness.run(test_set, modes=tuple(args.modes))
    EvalHarness.save_report(report, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for mode, metrics in report.modes.items():
        print(f"\n  {mode}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value}")
    print()


if __name__ == "__main__":
    main()
