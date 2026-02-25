"""Cross-encoder re-ranker for precision re-ranking of retrieval candidates.

Wraps a ``sentence_transformers.CrossEncoder`` model that scores
``(query, passage)`` pairs directly.  The model is lazy-loaded on
first call to avoid slow startup when re-ranking is disabled.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Cross-encoder re-ranker with swappable model and pass-through mode.

    Args:
        model_name: HuggingFace model identifier for the cross-encoder.
            Pass ``None`` or ``"none"`` to create a no-op pass-through
            instance that returns candidates in their original order.
    """

    def __init__(self, model_name: str | None = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._passthrough = (
            model_name is None or str(model_name).strip().lower() == "none"
        )

    # -----------------------------------------------------------------
    # Lazy loading
    # -----------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the cross-encoder on first use."""
        if self._model is not None or self._passthrough:
            return

        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder: %s", self._model_name)
        self._model = CrossEncoder(self._model_name)
        logger.info("Cross-encoder loaded")

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Re-rank candidates by cross-encoder relevance score.

        Args:
            query: The search query text.
            candidates: ``(chunk_id, chunk_text)`` pairs to re-rank.
            top_k: Number of top results to return.  ``None`` returns
                all candidates.

        Returns:
            ``(chunk_id, cross_encoder_score)`` tuples sorted by score
            descending.  In pass-through mode the score is the inverse
            of the original rank position (``1 / (i + 1)``).
        """
        if not candidates:
            return []

        if self._passthrough:
            results = [
                (cid, 1.0 / (i + 1))
                for i, (cid, _text) in enumerate(candidates)
            ]
            return results[:top_k] if top_k else results

        self._ensure_model()

        pairs = [(query, text) for (_cid, text) in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        scored = [
            (cid, float(score))
            for (cid, _text), score in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k] if top_k else scored

    @property
    def is_passthrough(self) -> bool:
        """``True`` if the instance is a no-op pass-through."""
        return self._passthrough
