"""Information retrieval evaluation metrics.

Provides MRR@K, NDCG@K, and Recall@K for evaluating ranked result lists
against a set of relevant document IDs.
"""

from __future__ import annotations

import math


def mrr_at_k(
    ranked_results: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Mean Reciprocal Rank at K.

    Returns the reciprocal of the rank of the first relevant result
    within the top K.  Returns 0.0 if no relevant result appears.

    Args:
        ranked_results: Ordered list of chunk/case IDs (best first).
        relevant_ids: Set of IDs considered relevant.
        k: Cutoff rank.

    Returns:
        MRR score in [0, 1].
    """
    for i, rid in enumerate(ranked_results[:k]):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    ranked_results: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Uses binary relevance: a result is either relevant (gain=1) or
    not (gain=0).  The ideal ranking places all relevant documents
    at the top positions.

    Args:
        ranked_results: Ordered list of chunk/case IDs (best first).
        relevant_ids: Set of IDs considered relevant.
        k: Cutoff rank.

    Returns:
        NDCG score in [0, 1].  Returns 0.0 if ``relevant_ids`` is empty.
    """
    if not relevant_ids:
        return 0.0

    # DCG of the actual ranking
    dcg = 0.0
    for i, rid in enumerate(ranked_results[:k]):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank is 1-indexed

    # Ideal DCG: all relevant docs at the top
    n_relevant_in_topk = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(j + 2) for j in range(n_relevant_in_topk))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def recall_at_k(
    ranked_results: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Recall at K.

    Fraction of relevant documents that appear in the top K results.

    Args:
        ranked_results: Ordered list of chunk/case IDs (best first).
        relevant_ids: Set of IDs considered relevant.
        k: Cutoff rank.

    Returns:
        Recall score in [0, 1].  Returns 0.0 if ``relevant_ids`` is
        empty.
    """
    if not relevant_ids:
        return 0.0

    retrieved_relevant = sum(
        1 for rid in ranked_results[:k] if rid in relevant_ids
    )
    return retrieved_relevant / len(relevant_ids)
