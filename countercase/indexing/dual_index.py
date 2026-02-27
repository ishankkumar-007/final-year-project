"""Dual index combining DPR (FAISS) and ChromaDB for hybrid retrieval.

Queries both indexes independently and returns their result lists
separately so downstream fusion can combine them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from countercase.config.settings import settings
from countercase.indexing.chroma_index import ChromaIndexWrapper
from countercase.indexing.dpr_index import DPRIndexWrapper

logger = logging.getLogger(__name__)


class DualIndex:
    """Facade over DPR and ChromaDB indexes.

    Args:
        dpr_index: Pre-configured ``DPRIndexWrapper``.  If ``None``, a
            default instance is created.
        chroma_index: Pre-configured ``ChromaIndexWrapper``.  If ``None``,
            a default instance is created.
    """

    def __init__(
        self,
        dpr_index: DPRIndexWrapper | None = None,
        chroma_index: ChromaIndexWrapper | None = None,
    ) -> None:
        self.dpr = dpr_index or DPRIndexWrapper()
        self.chroma = chroma_index or ChromaIndexWrapper()

        # Auto-load persisted DPR index if files exist on disk.
        # ChromaDB auto-loads via PersistentClient; DPR/FAISS does not.
        dpr_faiss = self.dpr._index_dir / "dpr.faiss"
        if self.dpr._index is None and dpr_faiss.exists():
            logger.info("Auto-loading DPR index from %s", self.dpr._index_dir)
            self.dpr.load()

    # -----------------------------------------------------------------
    # Indexing
    # -----------------------------------------------------------------

    def index_chunks(
        self,
        chunk_ids: list[str],
        chunk_texts: list[str],
        metadatas: list[dict[str, Any]],
        dpr_batch_size: int = 16,
        chroma_batch_size: int = 100,
    ) -> None:
        """Index chunks into both DPR (FAISS) and ChromaDB.

        Args:
            chunk_ids: Unique identifiers for each chunk.
            chunk_texts: Text content of each chunk.
            metadatas: Per-chunk metadata dicts for ChromaDB.
            dpr_batch_size: Batch size for DPR encoding.
            chroma_batch_size: Batch size for ChromaDB upserts.
        """
        logger.info("Indexing %d chunks into dual index...", len(chunk_ids))
        self.dpr.index_chunks(chunk_ids, chunk_texts, batch_size=dpr_batch_size)
        self.chroma.add_chunks(
            chunk_ids, chunk_texts, metadatas, batch_size=chroma_batch_size
        )
        logger.info("Dual indexing complete")

    def add_chunks(
        self,
        chunk_ids: list[str],
        chunk_texts: list[str],
        metadatas: list[dict[str, Any]],
        dpr_batch_size: int = 16,
        chroma_batch_size: int = 100,
    ) -> None:
        """Incrementally add chunks to both indexes.

        Unlike :meth:`index_chunks`, this preserves existing data.
        ChromaDB upserts naturally; DPR skips duplicates internally.

        Args:
            chunk_ids: Unique identifiers for each chunk.
            chunk_texts: Text content of each chunk.
            metadatas: Per-chunk metadata dicts for ChromaDB.
            dpr_batch_size: Batch size for DPR encoding.
            chroma_batch_size: Batch size for ChromaDB upserts.
        """
        logger.info("Incrementally adding %d chunks to dual index...", len(chunk_ids))
        self.dpr.add_chunks(chunk_ids, chunk_texts, batch_size=dpr_batch_size)
        self.chroma.add_chunks(
            chunk_ids, chunk_texts, metadatas, batch_size=chroma_batch_size
        )
        logger.info("Incremental dual indexing complete")

    # -----------------------------------------------------------------
    # Querying
    # -----------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Query both indexes and return their results separately.

        Args:
            query_text: The search query.
            top_k: Number of results per index.  Defaults to
                ``settings.TOP_K``.
            metadata_filters: Optional ChromaDB ``where`` filter.  The
                DPR index does not support metadata filtering, so this
                is only applied to the ChromaDB query.

        Returns:
            A tuple ``(dpr_results, chroma_results)`` where each is a
            list of ``(chunk_id, score)`` tuples sorted by score
            descending.
        """
        top_k = top_k or settings.TOP_K
        dpr_results = self.dpr.query(query_text, top_k=top_k)
        chroma_results = self.chroma.query(
            query_text, top_k=top_k, where=metadata_filters
        )
        logger.info(
            "Dual query returned %d DPR + %d Chroma results",
            len(dpr_results),
            len(chroma_results),
        )
        return dpr_results, chroma_results

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(
        self,
        dpr_dir: Path | None = None,
    ) -> None:
        """Persist the DPR index to disk.

        ChromaDB persists automatically via its ``PersistentClient``.

        Args:
            dpr_dir: Directory for the DPR FAISS index.
        """
        self.dpr.save(index_dir=dpr_dir)

    def load(
        self,
        dpr_dir: Path | None = None,
    ) -> None:
        """Load a previously saved DPR index.

        ChromaDB is loaded automatically when its client is initialized.

        Args:
            dpr_dir: Directory containing the DPR FAISS index.
        """
        self.dpr.load(index_dir=dpr_dir)
