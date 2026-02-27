"""Configuration settings for the COUNTERCASE project.

All paths are relative to the repository root. Parameters are tunable
via the Settings dataclass.  The ``.env`` file at the repo root is
loaded automatically so that ``HF_TOKEN`` and other secrets are
available via ``os.environ``.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root (or cwd) so HF_TOKEN is available globally.
load_dotenv(override=False)


def _repo_root() -> Path:
    """Return the repository root directory.

    Walks upward from this file until it finds the directory containing
    ``judgments-data/``.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "judgments-data").exists():
            return current
        current = current.parent
    # Fallback: two levels up from config/settings.py
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class Settings:
    """Central configuration for the COUNTERCASE system.

    Attributes:
        DATA_DIR: Path to extracted judgment tar directories.
        METADATA_DIR: Path to parquet metadata files.
        CHUNK_SIZE: Number of tokens per chunk.
        CHUNK_OVERLAP: Overlap tokens between consecutive chunks.
        EMBEDDING_MODEL: Sentence-transformers model for ChromaDB embeddings.
        DPR_QUESTION_MODEL: HuggingFace model for DPR question encoding.
        DPR_CONTEXT_MODEL: HuggingFace model for DPR context encoding.
        CHROMA_PERSIST_DIR: Directory for ChromaDB persistence.
        DPR_INDEX_DIR: Directory for serialized DPR FAISS index.
        TOP_K: Number of candidates retrieved per index.
        RRF_K: Constant k used in reciprocal rank fusion.
    """

    REPO_ROOT: Path = field(default_factory=_repo_root)

    # Data paths (relative to repo root, resolved at post_init)
    DATA_DIR: Path = field(default=None)  # type: ignore[assignment]
    METADATA_DIR: Path = field(default=None)  # type: ignore[assignment]

    # Chunking
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 128

    # Embedding models
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    DPR_QUESTION_MODEL: str = "facebook/dpr-question_encoder-single-nq-base"
    DPR_CONTEXT_MODEL: str = "facebook/dpr-ctx_encoder-single-nq-base"

    # Index persistence
    CHROMA_PERSIST_DIR: Path = field(default=None)  # type: ignore[assignment]
    DPR_INDEX_DIR: Path = field(default=None)  # type: ignore[assignment]

    # Retrieval
    TOP_K: int = 50
    RRF_K: int = 60

    # LLM configuration (Mistral API)
    LLM_API_URL: str = field(default=None)  # type: ignore[assignment]
    LLM_API_KEY: str = field(default=None)  # type: ignore[assignment]
    LLM_API_MODEL: str = field(default=None)  # type: ignore[assignment]
    LOCAL_LLM_MODEL: str = field(default=None)  # type: ignore[assignment]
    LLM_TIMEOUT: int = 60

    # Output / intermediary data
    DATA_OUTPUT_DIR: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.DATA_DIR is None:
            self.DATA_DIR = self.REPO_ROOT / "judgments-data" / "data" / "tar"
        if self.METADATA_DIR is None:
            self.METADATA_DIR = self.REPO_ROOT / "judgments-data" / "metadata" / "parquet"
        if self.CHROMA_PERSIST_DIR is None:
            self.CHROMA_PERSIST_DIR = self.REPO_ROOT / "countercase" / "data" / "chroma"
        if self.DPR_INDEX_DIR is None:
            self.DPR_INDEX_DIR = self.REPO_ROOT / "countercase" / "data" / "dpr_index"
        if self.DATA_OUTPUT_DIR is None:
            self.DATA_OUTPUT_DIR = self.REPO_ROOT / "countercase" / "data"
        if self.LLM_API_URL is None:
            self.LLM_API_URL = os.getenv("LLM_API_URL", "https://api.mistral.ai/v1")
        if self.LLM_API_KEY is None:
            self.LLM_API_KEY = os.getenv("LLM_API_KEY", "")
        if self.LLM_API_MODEL is None:
            self.LLM_API_MODEL = os.getenv("LLM_API_MODEL", "open-mistral-7b")
        if self.LOCAL_LLM_MODEL is None:
            self.LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")


# Module-level singleton for convenience.
settings = Settings()
