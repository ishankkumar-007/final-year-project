"""Section tagging for text chunks.

Tags each chunk with the section type from the section detector by
computing which detected section has the greatest overlap with the
chunk's character range.
"""

from __future__ import annotations

import logging

from countercase.ingestion.section_detector import Section, SectionType
from countercase.preprocessing.chunker import Chunk

logger = logging.getLogger(__name__)


def _overlap(start1: int, end1: int, start2: int, end2: int) -> int:
    """Compute the character overlap between two ranges.

    Args:
        start1: Start of range 1.
        end1: End of range 1.
        start2: Start of range 2.
        end2: End of range 2.

    Returns:
        Number of overlapping characters (>= 0).
    """
    return max(0, min(end1, end2) - max(start1, start2))


def tag_chunks(
    chunks: list[Chunk],
    sections: list[Section],
) -> list[Chunk]:
    """Tag each chunk with the section type that has the most overlap.

    The function mutates chunks in-place (updates ``section_type``) and
    also returns them for convenience.

    Args:
        chunks: List of chunks with ``char_start`` and ``char_end`` set.
        sections: List of detected sections from the section detector.

    Returns:
        The same list of chunks with updated ``section_type`` fields.
    """
    if not sections:
        logger.warning("No sections provided; all chunks tagged as Unknown")
        return chunks

    for chunk in chunks:
        best_overlap = 0
        best_type = SectionType.Unknown

        for section in sections:
            ov = _overlap(
                chunk.char_start, chunk.char_end,
                section.start_char, section.end_char,
            )
            if ov > best_overlap:
                best_overlap = ov
                best_type = section.section_type

        chunk.section_type = best_type.value

    logger.info("Tagged %d chunks with section types", len(chunks))
    return chunks
