"""DPR (Dense Passage Retrieval) index using FAISS.

Wraps the facebook/dpr-question_encoder and dpr-ctx_encoder models with
a FAISS inner-product index.  Models are lazy-loaded on first use.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from countercase.config.settings import settings

logger = logging.getLogger(__name__)


class DPRIndexWrapper:
    """Dense Passage Retrieval index backed by FAISS.

    Models are loaded lazily on first call to ``encode_query`` or
    ``index_chunks`` to avoid slow import-time initialization.

    Args:
        question_model: HuggingFace model name for query encoding.
        context_model: HuggingFace model name for passage encoding.
        index_dir: Directory to persist the FAISS index and mappings.
    """

    def __init__(
        self,
        question_model: str | None = None,
        context_model: str | None = None,
        index_dir: Path | None = None,
    ) -> None:
        self._question_model_name = question_model or settings.DPR_QUESTION_MODEL
        self._context_model_name = context_model or settings.DPR_CONTEXT_MODEL
        self._index_dir = index_dir or settings.DPR_INDEX_DIR

        # Lazy-loaded resources
        self._q_encoder: Any = None
        self._q_tokenizer: Any = None
        self._ctx_encoder: Any = None
        self._ctx_tokenizer: Any = None

        self._index: faiss.IndexFlatIP | None = None
        self._id_map: list[str] = []  # FAISS internal idx -> chunk_id

    # -----------------------------------------------------------------
    # Lazy model loading
    # -----------------------------------------------------------------

    def _load_question_encoder(self) -> None:
        """Load the DPR question encoder and tokenizer."""
        if self._q_encoder is not None:
            return
        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

        logger.info("Loading DPR question encoder: %s", self._question_model_name)
        self._q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            self._question_model_name
        )
        self._q_encoder = DPRQuestionEncoder.from_pretrained(
            self._question_model_name
        )
        self._q_encoder.eval()

    def _load_context_encoder(self) -> None:
        """Load the DPR context encoder and tokenizer."""
        if self._ctx_encoder is not None:
            return
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

        logger.info("Loading DPR context encoder: %s", self._context_model_name)
        self._ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            self._context_model_name
        )
        self._ctx_encoder = DPRContextEncoder.from_pretrained(
            self._context_model_name
        )
        self._ctx_encoder.eval()

    # -----------------------------------------------------------------
    # Encoding helpers
    # -----------------------------------------------------------------

    def _encode_query(self, text: str) -> np.ndarray:
        """Encode a query string into a DPR embedding.

        Args:
            text: The query text.

        Returns:
            A 1-D numpy array (float32) of the embedding.
        """
        import torch

        self._load_question_encoder()
        inputs = self._q_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self._q_encoder(**inputs)
        # pooler_output: (1, dim)
        embedding = outputs.pooler_output.squeeze(0).cpu().numpy().astype(np.float32)
        return embedding

    def _encode_context(self, text: str) -> np.ndarray:
        """Encode a passage/context string into a DPR embedding.

        Args:
            text: The passage text.

        Returns:
            A 1-D numpy array (float32) of the embedding.
        """
        import torch

        self._load_context_encoder()
        inputs = self._ctx_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self._ctx_encoder(**inputs)
        embedding = outputs.pooler_output.squeeze(0).cpu().numpy().astype(np.float32)
        return embedding

    def _encode_contexts_batch(
        self, texts: list[str], batch_size: int = 16
    ) -> np.ndarray:
        """Encode a batch of passages into DPR embeddings.

        Args:
            texts: List of passage texts.
            batch_size: Number of passages per forward pass.

        Returns:
            A 2-D numpy array of shape ``(len(texts), dim)``.
        """
        import torch

        self._load_context_encoder()
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self._ctx_tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = self._ctx_encoder(**inputs)
            embeddings = outputs.pooler_output.cpu().numpy().astype(np.float32)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    # -----------------------------------------------------------------
    # Indexing
    # -----------------------------------------------------------------

    def index_chunks(
        self,
        chunk_ids: list[str],
        chunk_texts: list[str],
        batch_size: int = 16,
    ) -> None:
        """Build the FAISS index from a set of chunks.

        Args:
            chunk_ids: Unique identifiers for each chunk.
            chunk_texts: Text content of each chunk.
            batch_size: Encoding batch size.
        """
        logger.info("Encoding %d chunks with DPR context encoder...", len(chunk_texts))
        embeddings = self._encode_contexts_batch(chunk_texts, batch_size=batch_size)
        dim = embeddings.shape[1]

        self._index = faiss.IndexFlatIP(dim)
        # Normalize for inner product to behave like cosine similarity.
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self._id_map = list(chunk_ids)
        logger.info("FAISS index built: %d vectors, dim=%d", self._index.ntotal, dim)

    # -----------------------------------------------------------------
    # Querying
    # -----------------------------------------------------------------

    def query(
        self, query_text: str, top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """Search the DPR index for the most relevant chunks.

        Args:
            query_text: The search query.
            top_k: Number of results to return.  Defaults to
                ``settings.TOP_K``.

        Returns:
            A list of ``(chunk_id, score)`` tuples sorted by score
            descending.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("DPR index is empty; returning no results")
            return []

        top_k = top_k or settings.TOP_K
        query_emb = self._encode_query(query_text).reshape(1, -1)
        faiss.normalize_L2(query_emb)

        scores, indices = self._index.search(query_emb, min(top_k, self._index.ntotal))
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, index_dir: Path | None = None) -> None:
        """Serialize the FAISS index and id mapping to disk.

        Args:
            index_dir: Directory to save into.  Defaults to the
                configured ``DPR_INDEX_DIR``.
        """
        index_dir = index_dir or self._index_dir
        index_dir.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            index_path = index_dir / "dpr.faiss"
            faiss.write_index(self._index, str(index_path))
            logger.info("FAISS index saved to %s", index_path)

        map_path = index_dir / "id_map.pkl"
        with open(map_path, "wb") as fh:
            pickle.dump(self._id_map, fh)
        logger.info("ID map saved to %s", map_path)

    def load(self, index_dir: Path | None = None) -> None:
        """Load a previously saved FAISS index and id mapping.

        Args:
            index_dir: Directory to load from.  Defaults to the
                configured ``DPR_INDEX_DIR``.
        """
        index_dir = index_dir or self._index_dir
        index_path = index_dir / "dpr.faiss"
        map_path = index_dir / "id_map.pkl"

        if not index_path.exists() or not map_path.exists():
            logger.error("DPR index files not found in %s", index_dir)
            return

        self._index = faiss.read_index(str(index_path))
        with open(map_path, "rb") as fh:
            self._id_map = pickle.load(fh)
        logger.info(
            "DPR index loaded: %d vectors, %d ids",
            self._index.ntotal,
            len(self._id_map),
        )
