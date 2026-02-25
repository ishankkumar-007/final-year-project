"""Hybrid retriever composing DPR + ChromaDB, RRF, MMR, and cross-encoder.

This is the main retrieval interface for the COUNTERCASE system.
It executes a six-stage pipeline:
    1. Metadata pre-filter construction
    2. DPR + ChromaDB ANN search  (top_k=50 each)
    3. Reciprocal Rank Fusion
    4. MMR diversity selection
    5. Cross-encoder re-ranking  (optional)
    6. Source-attribution metadata attachment
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from countercase.config.settings import settings
from countercase.indexing.dual_index import DualIndex
from countercase.retrieval.mmr import mmr_select
from countercase.retrieval.reranker import CrossEncoderReranker
from countercase.retrieval.rrf import rrf_fuse

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# Result dataclass
# -----------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single retrieval result with full metadata and stage scores."""

    chunk_id: str
    text: str = ""
    source_pdf: str = ""
    page_number: int = 0
    section_type: str = "Unknown"
    case_id: str = ""
    year: int = 0
    bench_type: str = "Unknown"
    act_sections: str = ""
    outcome_label: str = "Unknown"

    # Per-stage scores
    rrf_score: float = 0.0
    mmr_score: float = 0.0
    reranker_score: float = 0.0
    final_rank: int = 0


# -----------------------------------------------------------------
# Main retriever
# -----------------------------------------------------------------

class HybridRetriever:
    """Six-stage hybrid retriever with timing instrumentation.

    Args:
        dual_index: Pre-configured :class:`DualIndex`.  A default
            instance is created if ``None``.
        reranker: Pre-configured cross-encoder re-ranker.  Defaults to
            the MS-MARCO MiniLM cross-encoder.
    """

    def __init__(
        self,
        dual_index: DualIndex | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.dual_index = dual_index or DualIndex()
        self.reranker = reranker or CrossEncoderReranker()

    # -----------------------------------------------------------------
    # Embedding fetch helper
    # -----------------------------------------------------------------

    def _fetch_embeddings(
        self, chunk_ids: list[str]
    ) -> dict[str, np.ndarray]:
        """Retrieve ChromaDB embeddings for the given chunk IDs.

        Returns a dict mapping chunk_id → numpy embedding vector.
        """
        chroma = self.dual_index.chroma
        chroma._ensure_client()

        embs: dict[str, np.ndarray] = {}
        # Query chromaDB in a single batch for efficiency
        if not chunk_ids:
            return embs

        try:
            result = chroma._collection.get(
                ids=chunk_ids, include=["embeddings"]
            )
            ids_out = result.get("ids") if result else []
            emb_raw = result.get("embeddings") if result else None
            if ids_out and emb_raw is not None:
                for cid, emb in zip(ids_out, emb_raw):
                    if emb is not None:
                        embs[cid] = np.asarray(emb, dtype=np.float32)
        except Exception:
            logger.exception("Failed to fetch embeddings from ChromaDB")
        return embs

    # -----------------------------------------------------------------
    # Main retrieval method
    # -----------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
        lambda_mult: float = 0.6,
        use_reranker: bool = True,
        candidate_pool: int | None = None,
    ) -> list[RetrievalResult]:
        """Execute the full six-stage retrieval pipeline.

        Args:
            query: The search query text.
            top_k: Number of final results to return.
            metadata_filters: Optional ChromaDB ``where`` filter dict
                applied as a pre-filter (Stage 1).
            lambda_mult: MMR diversity parameter (Stage 4).
            use_reranker: Whether to apply cross-encoder re-ranking
                (Stage 5).  Set to ``False`` for speed or ablation.
            candidate_pool: Number of candidates per index in Stage 2.
                Defaults to ``settings.TOP_K`` (50).

        Returns:
            Ordered list of :class:`RetrievalResult` with final_rank
            set starting from 1.
        """
        pool_k = candidate_pool or settings.TOP_K
        timings: dict[str, float] = {}

        # ---- Stage 1: Metadata pre-filter ----
        t0 = time.perf_counter()
        where_clause = metadata_filters  # Pass through to ChromaDB
        timings["stage1_prefilter"] = time.perf_counter() - t0

        # ---- Stage 2: DPR + ChromaDB ANN search ----
        t0 = time.perf_counter()
        dpr_results, chroma_results = self.dual_index.query(
            query, top_k=pool_k, metadata_filters=where_clause
        )
        timings["stage2_dual_search"] = time.perf_counter() - t0

        # ---- Stage 3: RRF Fusion ----
        t0 = time.perf_counter()
        fused = rrf_fuse([dpr_results, chroma_results], k=settings.RRF_K)
        timings["stage3_rrf_fusion"] = time.perf_counter() - t0

        # Build RRF score lookup
        rrf_scores: dict[str, float] = dict(fused)

        # ---- Stage 4: MMR selection ----
        t0 = time.perf_counter()
        # Fetch embeddings for all fused candidates
        fused_ids = [cid for cid, _ in fused]
        embeddings = self._fetch_embeddings(fused_ids)

        # MMR selects at most top_k from fused list
        mmr_results = mmr_select(
            fused, embeddings, top_k=top_k, lambda_mult=lambda_mult
        )
        timings["stage4_mmr"] = time.perf_counter() - t0

        # Build MMR score lookup
        mmr_scores: dict[str, float] = dict(mmr_results)

        # ---- Stage 5: Cross-encoder re-ranking (optional) ----
        t0 = time.perf_counter()
        if use_reranker and not self.reranker.is_passthrough:
            # Fetch texts for MMR results
            mmr_ids = [cid for cid, _ in mmr_results]
            texts = self._fetch_texts(mmr_ids)
            ce_candidates = [(cid, texts.get(cid, "")) for cid in mmr_ids]
            reranked = self.reranker.rerank(query, ce_candidates, top_k=top_k)
            reranker_scores: dict[str, float] = dict(reranked)
            final_order = [cid for cid, _ in reranked]
        else:
            reranker_scores = {}
            final_order = [cid for cid, _ in mmr_results]
        timings["stage5_reranker"] = time.perf_counter() - t0

        # ---- Stage 6: Source-attribution metadata ----
        t0 = time.perf_counter()
        results: list[RetrievalResult] = []
        for rank_idx, chunk_id in enumerate(final_order):
            meta = self.dual_index.chroma.get_metadata(chunk_id)
            text = self.dual_index.chroma.get_document(chunk_id)

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    source_pdf=str(meta.get("source_pdf", "")),
                    page_number=int(meta.get("page_number", 0)),
                    section_type=str(meta.get("section_type", "Unknown")),
                    case_id=str(meta.get("case_id", "")),
                    year=int(meta.get("year", 0)),
                    bench_type=str(meta.get("bench_type", "Unknown")),
                    act_sections=str(meta.get("act_sections", "")),
                    outcome_label=str(meta.get("outcome_label", "Unknown")),
                    rrf_score=rrf_scores.get(chunk_id, 0.0),
                    mmr_score=mmr_scores.get(chunk_id, 0.0),
                    reranker_score=reranker_scores.get(chunk_id, 0.0),
                    final_rank=rank_idx + 1,
                )
            )
        timings["stage6_attribution"] = time.perf_counter() - t0

        # Log timings
        total = sum(timings.values())
        timing_parts = ", ".join(
            f"{k}={v:.3f}s" for k, v in timings.items()
        )
        logger.info(
            "Retrieval complete in %.3fs | %s | %d results",
            total,
            timing_parts,
            len(results),
        )

        return results

    # -----------------------------------------------------------------
    # Ablation helpers
    # -----------------------------------------------------------------

    def retrieve_dpr_only(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """DPR-only retrieval (no ChromaDB, no RRF, no MMR, no reranker)."""
        dpr_results = self.dual_index.dpr.query(query, top_k=top_k)
        return self._results_from_id_scores(dpr_results, top_k=top_k)

    def retrieve_chroma_only(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """ChromaDB-only retrieval (no DPR, no RRF, no MMR, no reranker)."""
        chroma_results = self.dual_index.chroma.query(
            query, top_k=top_k, where=metadata_filters
        )
        return self._results_from_id_scores(chroma_results, top_k=top_k)

    def retrieve_hybrid_no_mmr(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Hybrid DPR+ChromaDB with RRF but no MMR and no reranker."""
        dpr_results, chroma_results = self.dual_index.query(
            query, top_k=settings.TOP_K, metadata_filters=metadata_filters
        )
        fused = rrf_fuse([dpr_results, chroma_results], k=settings.RRF_K)
        return self._results_from_id_scores(fused[:top_k], top_k=top_k)

    def retrieve_hybrid_no_reranker(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
        lambda_mult: float = 0.6,
    ) -> list[RetrievalResult]:
        """Full hybrid with MMR but no cross-encoder re-ranking."""
        return self.retrieve(
            query,
            top_k=top_k,
            metadata_filters=metadata_filters,
            lambda_mult=lambda_mult,
            use_reranker=False,
        )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _fetch_texts(self, chunk_ids: list[str]) -> dict[str, str]:
        """Retrieve document text for a list of chunk IDs."""
        texts: dict[str, str] = {}
        for cid in chunk_ids:
            texts[cid] = self.dual_index.chroma.get_document(cid)
        return texts

    def _results_from_id_scores(
        self,
        id_scores: list[tuple[str, float]],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Convert a list of (chunk_id, score) to RetrievalResult objects."""
        results: list[RetrievalResult] = []
        for rank_idx, (chunk_id, score) in enumerate(id_scores[:top_k]):
            meta = self.dual_index.chroma.get_metadata(chunk_id)
            text = self.dual_index.chroma.get_document(chunk_id)
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    source_pdf=str(meta.get("source_pdf", "")),
                    page_number=int(meta.get("page_number", 0)),
                    section_type=str(meta.get("section_type", "Unknown")),
                    case_id=str(meta.get("case_id", "")),
                    year=int(meta.get("year", 0)),
                    bench_type=str(meta.get("bench_type", "Unknown")),
                    act_sections=str(meta.get("act_sections", "")),
                    outcome_label=str(meta.get("outcome_label", "Unknown")),
                    rrf_score=score,
                    mmr_score=0.0,
                    reranker_score=0.0,
                    final_rank=rank_idx + 1,
                )
            )
        return results
