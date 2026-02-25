"""Reciprocal Rank Fusion for combining ranked result lists.

Implements the standard RRF formula:
    score(chunk) = sum over lists of (1 / (k + rank))
where rank is 1-indexed.
"""

from __future__ import annotations

from collections import defaultdict


def rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    For each chunk appearing in any of the input lists, compute an RRF
    score as the sum of ``1 / (k + rank)`` across all lists where the
    chunk appears.  ``rank`` is 1-indexed.

    Args:
        ranked_lists: A list of ranked result lists.  Each inner list
            contains ``(chunk_id, original_score)`` tuples in rank order
            (best first).
        k: The RRF smoothing constant.  Higher values reduce the
            influence of high-ranked items.

    Returns:
        A single merged list of ``(chunk_id, rrf_score)`` tuples sorted
        by RRF score descending.
    """
    scores: dict[str, float] = defaultdict(float)

    for result_list in ranked_lists:
        for rank_zero, (chunk_id, _original_score) in enumerate(result_list):
            rank = rank_zero + 1  # 1-indexed
            scores[chunk_id] += 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return fused
