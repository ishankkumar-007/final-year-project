"""Maximal Marginal Relevance selection for diverse retrieval.

Implements the MMR objective:
    selected = argmax_{c in remaining} (
        lambda_mult * relevance(c)
        - (1 - lambda_mult) * max_sim(c, selected)
    )

Designed to be applied after RRF fusion to diversify the top-K results.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Returns 0.0 if either vector has zero norm.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def mmr_select(
    candidates: list[tuple[str, float]],
    embeddings: dict[str, list[float] | np.ndarray],
    top_k: int,
    lambda_mult: float = 0.6,
) -> list[tuple[str, float]]:
    """Select diverse results using Maximal Marginal Relevance.

    At each step, the candidate that maximizes the MMR objective is
    added to the selected set.  The MMR score balances relevance
    (from the input score, typically RRF) and diversity (minimum
    cosine distance from already-selected embeddings).

    Args:
        candidates: ``(chunk_id, relevance_score)`` pairs sorted by
            relevance descending.  Typically the output of
            :func:`countercase.retrieval.rrf.rrf_fuse`.
        embeddings: Mapping from ``chunk_id`` to embedding vector.
            Vectors may be lists or numpy arrays.
        top_k: Number of results to return.
        lambda_mult: Balance between relevance (1.0) and diversity
            (0.0).  Default ``0.6`` is slightly relevance-heavy,
            appropriate for legal retrieval where thematic overlap
            is often legitimate.

    Returns:
        A list of ``(chunk_id, mmr_score)`` tuples in selection order.
    """
    if not candidates:
        return []

    # Normalise relevance scores to [0, 1] for fair combination.
    max_rel = max(score for _, score in candidates)
    min_rel = min(score for _, score in candidates)
    rel_range = max_rel - min_rel if max_rel != min_rel else 1.0

    # Pre-convert embeddings to numpy arrays and filter candidates
    # whose embeddings are unavailable.
    valid_candidates: list[tuple[str, float]] = []
    emb_cache: dict[str, np.ndarray] = {}

    for chunk_id, score in candidates:
        if chunk_id in embeddings:
            emb = embeddings[chunk_id]
            emb_cache[chunk_id] = (
                np.asarray(emb, dtype=np.float32)
                if not isinstance(emb, np.ndarray)
                else emb.astype(np.float32)
            )
            valid_candidates.append((chunk_id, score))
        else:
            warnings.warn(
                f"MMR: embedding missing for chunk '{chunk_id}'; skipping",
                stacklevel=2,
            )

    if not valid_candidates:
        logger.warning("MMR: no valid candidates with embeddings; returning empty list")
        return []

    top_k = min(top_k, len(valid_candidates))

    selected: list[tuple[str, float]] = []
    selected_embs: list[np.ndarray] = []
    remaining = dict(valid_candidates)  # chunk_id -> relevance_score

    for _ in range(top_k):
        best_id: str | None = None
        best_mmr: float = -float("inf")

        for chunk_id, rel_score in remaining.items():
            norm_rel = (rel_score - min_rel) / rel_range

            # Max similarity to already-selected embeddings
            if selected_embs:
                max_sim = max(
                    _cosine_similarity(emb_cache[chunk_id], sel_emb)
                    for sel_emb in selected_embs
                )
            else:
                max_sim = 0.0

            mmr_score = lambda_mult * norm_rel - (1.0 - lambda_mult) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_id = chunk_id

        if best_id is None:
            break

        selected.append((best_id, best_mmr))
        selected_embs.append(emb_cache[best_id])
        del remaining[best_id]

    return selected
