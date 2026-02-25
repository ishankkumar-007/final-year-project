"""ChromaDB index wrapper for COUNTERCASE.

Stores embeddings from the configured sentence-transformers model with
per-chunk metadata for filtered ANN search.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from countercase.config.settings import settings

logger = logging.getLogger(__name__)


class ChromaIndexWrapper:
    """ChromaDB collection wrapper with metadata-filtered query support.

    Args:
        persist_dir: Directory for ChromaDB persistence.  Defaults to
            ``settings.CHROMA_PERSIST_DIR``.
        collection_name: Name of the ChromaDB collection.
        embedding_model: Sentence-transformers model name.
    """

    def __init__(
        self,
        persist_dir: Path | None = None,
        collection_name: str = "countercase_chunks",
        embedding_model: str | None = None,
    ) -> None:
        self._persist_dir = str(persist_dir or settings.CHROMA_PERSIST_DIR)
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model or settings.EMBEDDING_MODEL

        self._client: chromadb.ClientAPI | None = None
        self._collection: Any = None
        self._ef: Any = None  # embedding function

    # -----------------------------------------------------------------
    # Initialization helpers
    # -----------------------------------------------------------------

    def _ensure_client(self) -> None:
        """Create the ChromaDB client and collection if needed."""
        if self._client is not None:
            return

        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Use sentence-transformers embedding function.
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=self._embedding_model_name,
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d items)",
            self._collection_name,
            self._collection.count(),
        )

    # -----------------------------------------------------------------
    # Indexing
    # -----------------------------------------------------------------

    def add_chunks(
        self,
        chunk_ids: list[str],
        chunk_texts: list[str],
        metadatas: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """Add chunks with metadata to the ChromaDB collection.

        Args:
            chunk_ids: Unique identifiers for each chunk.
            chunk_texts: Text content of each chunk.
            metadatas: Per-chunk metadata dicts.  Expected keys:
                ``year``, ``bench_type``, ``act_sections``,
                ``section_type``, ``outcome_label``, ``source_pdf``,
                ``page_number``, ``case_id``.
            batch_size: Number of chunks per upsert call.
        """
        self._ensure_client()

        # Sanitize metadata: ChromaDB requires str, int, float, or bool.
        safe_metas: list[dict[str, Any]] = []
        for meta in metadatas:
            safe: dict[str, Any] = {}
            for k, v in meta.items():
                if isinstance(v, list):
                    safe[k] = ", ".join(str(x) for x in v)
                elif v is None:
                    safe[k] = ""
                else:
                    safe[k] = v
            safe_metas.append(safe)

        total = len(chunk_ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self._collection.upsert(
                ids=chunk_ids[start:end],
                documents=chunk_texts[start:end],
                metadatas=safe_metas[start:end],
            )
        logger.info("Upserted %d chunks to ChromaDB", total)

    # -----------------------------------------------------------------
    # Querying
    # -----------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Query the ChromaDB collection with optional metadata filtering.

        Args:
            query_text: The search query text.
            top_k: Number of results to return.  Defaults to
                ``settings.TOP_K``.
            where: Optional ChromaDB ``where`` filter dict for metadata
                pre-filtering.

        Returns:
            A list of ``(chunk_id, distance)`` tuples.  Lower distance
            means higher similarity for cosine space.
        """
        self._ensure_client()
        top_k = top_k or settings.TOP_K

        n_results = min(top_k, self._collection.count())
        if n_results == 0:
            logger.warning("ChromaDB collection is empty; returning no results")
            return []

        query_kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": n_results,
        }
        if where:
            query_kwargs["where"] = where

        try:
            result = self._collection.query(**query_kwargs)
        except Exception:
            logger.exception("ChromaDB query failed")
            return []

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]

        # ChromaDB cosine distance: 0 = identical, 2 = opposite.
        # Convert to a similarity-like score: score = 1 - distance/2.
        pairs: list[tuple[str, float]] = []
        for cid, dist in zip(ids, distances):
            score = 1.0 - dist / 2.0
            pairs.append((cid, score))

        return pairs

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def count(self) -> int:
        """Return the number of items in the collection."""
        self._ensure_client()
        return self._collection.count()

    def get_metadata(self, chunk_id: str) -> dict[str, Any]:
        """Retrieve the metadata dict for a single chunk.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            Metadata dict, or empty dict if not found.
        """
        self._ensure_client()
        try:
            result = self._collection.get(ids=[chunk_id], include=["metadatas"])
            if result["metadatas"]:
                return result["metadatas"][0]
        except Exception:
            logger.exception("Failed to get metadata for %s", chunk_id)
        return {}

    def get_document(self, chunk_id: str) -> str:
        """Retrieve the text of a single chunk.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            Chunk text, or empty string if not found.
        """
        self._ensure_client()
        try:
            result = self._collection.get(ids=[chunk_id], include=["documents"])
            if result["documents"]:
                return result["documents"][0]
        except Exception:
            logger.exception("Failed to get document for %s", chunk_id)
        return ""
