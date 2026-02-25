"""Token-aware chunking for legal judgment text.

Uses LangChain RecursiveCharacterTextSplitter with tiktoken-based
token counting and legal-aware separators.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from countercase.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk produced from a judgment.

    Attributes:
        chunk_id: Unique identifier in the form ``{case_id}_chunk_{index}``.
        text: The chunk text content.
        source_pdf: Path or name of the source PDF.
        page_number: 1-indexed page where this chunk originates.
        section_type: Structural section label (e.g. ``Facts``, ``Held``).
        char_start: Start character offset in the full document text.
        char_end: End character offset in the full document text.
    """

    chunk_id: str
    text: str
    source_pdf: str
    page_number: int
    section_type: str
    char_start: int
    char_end: int


def _build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Build a RecursiveCharacterTextSplitter with legal-aware separators.

    Token counting uses the ``cl100k_base`` encoding from tiktoken.

    Args:
        chunk_size: Target chunk size in tokens.  Defaults to
            ``settings.CHUNK_SIZE``.
        chunk_overlap: Overlap in tokens.  Defaults to
            ``settings.CHUNK_OVERLAP``.

    Returns:
        A configured ``RecursiveCharacterTextSplitter`` instance.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    enc = tiktoken.get_encoding("cl100k_base")

    separators = [
        "\n\n",
        "\n",
        # Legal-aware: numbered paragraphs, lettered sub-points, roman
        # numeral sub-points.
        "\n\\d+\\.\\s",
        "\n\\([a-z]\\)\\s",
        "\n\\([ivx]+\\)\\s",
        ". ",
        " ",
        "",
    ]

    return RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda t: len(enc.encode(t)),
        is_separator_regex=True,
    )


def chunk_text(
    full_text: str,
    case_id: str,
    source_pdf: str,
    page_number: int = 1,
    section_type: str = "Unknown",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split a document's text into token-counted chunks.

    Args:
        full_text: The complete text to chunk.
        case_id: Identifier for the source case (used in ``chunk_id``).
        source_pdf: Path or name of the source PDF.
        page_number: Default page number for chunks.
        section_type: Default section type for chunks.
        chunk_size: Override for chunk size in tokens.
        chunk_overlap: Override for overlap in tokens.

    Returns:
        A list of ``Chunk`` objects with sequential ids.
    """
    splitter = _build_splitter(chunk_size, chunk_overlap)
    text_pieces = splitter.split_text(full_text)

    chunks: list[Chunk] = []
    search_start = 0
    for idx, piece in enumerate(text_pieces):
        # Locate the chunk in the original text for char offsets.
        char_start = full_text.find(piece, search_start)
        if char_start == -1:
            # Fallback: estimate from prior chunk position.
            char_start = search_start
        char_end = char_start + len(piece)
        # Advance search start past overlap to avoid re-matching.
        search_start = max(search_start, char_start + 1)

        chunk_id = f"{case_id}_chunk_{idx:04d}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=piece,
            source_pdf=source_pdf,
            page_number=page_number,
            section_type=section_type,
            char_start=char_start,
            char_end=char_end,
        ))

    logger.info(
        "Chunked '%s' into %d chunks (size=%s, overlap=%s)",
        case_id,
        len(chunks),
        chunk_size or settings.CHUNK_SIZE,
        chunk_overlap or settings.CHUNK_OVERLAP,
    )
    return chunks
