"""PDF text extraction using pdfplumber.

Handles multi-column layouts, header/footer removal, and batch
processing of judgment PDF directories.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    """Extracted text for a single PDF page.

    Attributes:
        pdf_path: Path to the source PDF file.
        page_number: 1-indexed page number.
        text: Extracted and cleaned text for this page.
    """

    pdf_path: str
    page_number: int
    text: str


def _reconstruct_reading_order(words: list[dict], page_height: float) -> list[dict]:
    """Sort word-level bounding boxes into reading order.

    Words are sorted primarily by their top coordinate (row), then by
    their left coordinate (column within a row).  This handles
    multi-column layouts by grouping words that share similar vertical
    positions.

    Args:
        words: List of word dicts from pdfplumber (must contain ``top``,
            ``x0``, ``bottom``, ``text``).
        page_height: Total height of the page in points.

    Returns:
        Sorted list of word dicts in reading order.
    """
    if not words:
        return []

    # Assign each word to a row bucket. Words within 3 points of each
    # other vertically are considered the same line.
    row_tolerance = 3.0
    sorted_by_top = sorted(words, key=lambda w: (w["top"], w["x0"]))

    rows: list[list[dict]] = []
    current_row: list[dict] = [sorted_by_top[0]]
    current_top = sorted_by_top[0]["top"]

    for word in sorted_by_top[1:]:
        if abs(word["top"] - current_top) <= row_tolerance:
            current_row.append(word)
        else:
            rows.append(sorted(current_row, key=lambda w: w["x0"]))
            current_row = [word]
            current_top = word["top"]
    rows.append(sorted(current_row, key=lambda w: w["x0"]))

    ordered: list[dict] = []
    for row in rows:
        ordered.extend(row)
    return ordered


def _remove_boilerplate(
    words: list[dict],
    page_height: float,
    header_fraction: float = 0.05,
    footer_fraction: float = 0.05,
) -> list[dict]:
    """Remove header and footer words from a page.

    Any word whose top is in the top ``header_fraction`` of the page or
    whose bottom is in the bottom ``footer_fraction`` is discarded.

    Args:
        words: List of word dicts from pdfplumber.
        page_height: Total height of the page in points.
        header_fraction: Fraction of page height for the header zone.
        footer_fraction: Fraction of page height for the footer zone.

    Returns:
        Filtered list of word dicts.
    """
    header_limit = page_height * header_fraction
    footer_limit = page_height * (1.0 - footer_fraction)
    return [
        w for w in words
        if w["top"] >= header_limit and w["bottom"] <= footer_limit
    ]


def extract_pdf(pdf_path: str) -> list[PageText]:
    """Extract text from a single PDF file.

    Handles multi-column layouts by reconstructing reading order from
    word bounding boxes.  Headers and footers (top/bottom 5% of the
    page) are removed.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of ``PageText`` objects, one per page with non-empty text.
    """
    pages: list[PageText] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                try:
                    words = page.extract_words() or []
                    if not words:
                        continue

                    page_height = float(page.height)
                    filtered = _remove_boilerplate(words, page_height)
                    if not filtered:
                        continue

                    ordered = _reconstruct_reading_order(filtered, page_height)
                    text = " ".join(w["text"] for w in ordered)
                    if text.strip():
                        pages.append(PageText(
                            pdf_path=str(pdf_path),
                            page_number=page_idx + 1,
                            text=text.strip(),
                        ))
                except Exception:
                    logger.exception(
                        "Error processing page %d of %s", page_idx + 1, pdf_path
                    )
    except Exception:
        logger.exception("Failed to open PDF: %s", pdf_path)

    return pages


def extract_directory(dir_path: str) -> dict[str, list[PageText]]:
    """Extract text from all PDFs in a directory.

    Args:
        dir_path: Path to a directory containing PDF files.

    Returns:
        A dict mapping PDF filename to a list of ``PageText`` objects.
    """
    results: dict[str, list[PageText]] = {}
    directory = Path(dir_path)
    if not directory.is_dir():
        logger.error("Directory does not exist: %s", dir_path)
        return results

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", dir_path)
        return results

    logger.info("Processing %d PDFs in %s", len(pdf_files), dir_path)
    for pdf_file in pdf_files:
        pages = extract_pdf(str(pdf_file))
        if pages:
            results[pdf_file.name] = pages
            logger.info("Extracted %d pages from %s", len(pages), pdf_file.name)
        else:
            logger.warning("No text extracted from %s", pdf_file.name)

    return results
