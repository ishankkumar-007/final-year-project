"""Phase 1 end-to-end pipeline for COUNTERCASE.

Ties together PDF extraction, text cleaning, section detection,
chunking, dual indexing (DPR + ChromaDB), and RRF fusion into a
single smoke-test script.

Usage:
    python -m countercase.pipeline_phase1 [--start-year 2024] [--end-year 2025]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from countercase.config.settings import settings
from countercase.indexing.chroma_index import ChromaIndexWrapper
from countercase.indexing.dual_index import DualIndex
from countercase.indexing.dpr_index import DPRIndexWrapper
from countercase.ingestion.metadata_extractor import (
    extract_metadata_from_text,
    inspect_metadata_schema,
    load_metadata,
)
from countercase.ingestion.noise_filter import clean_text
from countercase.ingestion.pdf_extractor import extract_directory
from countercase.ingestion.section_detector import detect_sections
from countercase.preprocessing.chunker import Chunk, chunk_text
from countercase.preprocessing.section_tagger import tag_chunks
from countercase.retrieval.rrf import rrf_fuse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _extract_pdfs_for_year(year: int) -> Path | None:
    """Return the path to extracted PDFs for a given year, or None.

    Looks for the extracted directory produced by
    ``extract-judgments.ps1``.
    """
    english_dir = (
        settings.DATA_DIR / f"year={year}" / "english" / "english"
    )
    if english_dir.is_dir():
        return english_dir
    # Fallback: maybe PDFs are directly inside the english/ folder
    alt = settings.DATA_DIR / f"year={year}" / "english"
    if alt.is_dir() and any(alt.glob("*.pdf")):
        return alt
    return None


def _build_case_id_from_filename(pdf_name: str, year: int) -> str:
    """Derive a case_id from the PDF filename and year.

    Args:
        pdf_name: Name of the PDF file.
        year: Year directory the PDF came from.

    Returns:
        A string suitable as a case identifier.
    """
    stem = Path(pdf_name).stem
    return f"{stem}_{year}"


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def run_pipeline(start_year: int, end_year: int) -> None:
    """Execute the Phase 1 pipeline over a year range.

    Steps:
        1. Inspect and log metadata schema.
        2. For each year, extract PDFs, clean text, detect sections,
           chunk, and collect chunks with metadata.
        3. Build the dual DPR + ChromaDB index.
        4. Run a sample query with RRF fusion and print results.

    Args:
        start_year: First year (inclusive).
        end_year: Last year (inclusive).
    """

    # ---- Step 1: Metadata inspection ----
    logger.info("=== Step 1: Metadata inspection ===")
    inspection = inspect_metadata_schema(start_year, end_year)
    if inspection:
        logger.info(
            "Metadata schema columns: %s", list(inspection.keys())
        )
    else:
        logger.warning("No metadata available for inspection")

    # Load metadata for later enrichment
    metadata_df = load_metadata(start_year, end_year)
    logger.info("Loaded metadata DataFrame: %d rows", len(metadata_df))

    # ---- Step 2: Extract, clean, chunk ----
    logger.info("=== Step 2: PDF extraction and chunking ===")
    all_chunk_ids: list[str] = []
    all_chunk_texts: list[str] = []
    all_metadatas: list[dict[str, Any]] = []
    all_chunks: list[Chunk] = []

    for year in range(start_year, end_year + 1):
        pdf_dir = _extract_pdfs_for_year(year)
        if pdf_dir is None:
            logger.warning("No extracted PDFs for year %d", year)
            continue

        logger.info("Processing PDFs for year %d from %s", year, pdf_dir)
        pdf_pages = extract_directory(str(pdf_dir))

        for pdf_name, pages in tqdm(
            pdf_pages.items(), desc=f"Year {year}", unit="pdf"
        ):
            if not pages:
                continue

            # Combine page texts into full document text.
            full_text = "\n\n".join(p.text for p in pages)
            full_text = clean_text(full_text)

            if not full_text.strip():
                logger.warning("Empty text after cleaning: %s", pdf_name)
                continue

            # Derive case_id from text or filename.
            case_id = _build_case_id_from_filename(pdf_name, year)

            # Extract text-based metadata.
            text_meta = extract_metadata_from_text(full_text)
            if text_meta.get("case_id"):
                case_id = (
                    text_meta["case_id"]
                    .replace("/", "_")
                    .replace(" ", "_")
                )

            # Detect sections.
            sections = detect_sections(full_text)

            # Chunk the full text.
            chunks = chunk_text(
                full_text=full_text,
                case_id=case_id,
                source_pdf=pdf_name,
                page_number=pages[0].page_number,
                section_type="Unknown",
            )

            # Tag chunks with section types.
            if sections:
                tag_chunks(chunks, sections)

            # Assign page numbers: best effort from character offset
            # mapping to page boundaries.
            page_boundaries: list[int] = []
            offset = 0
            for p in pages:
                page_boundaries.append(offset)
                offset += len(p.text) + 2  # +2 for "\n\n" join

            for chunk in chunks:
                # Find which page this chunk's start falls in.
                for pidx in range(len(page_boundaries) - 1, -1, -1):
                    if chunk.char_start >= page_boundaries[pidx]:
                        chunk.page_number = pages[pidx].page_number
                        break

            # Build metadata dicts for indexing.
            for chunk in chunks:
                meta: dict[str, Any] = {
                    "year": text_meta.get("year") or year,
                    "bench_type": text_meta.get("bench_type", "Unknown"),
                    "act_sections": ", ".join(
                        text_meta.get("act_sections", [])
                    ),
                    "section_type": chunk.section_type,
                    "outcome_label": text_meta.get("outcome_label", "Unknown"),
                    "source_pdf": chunk.source_pdf,
                    "page_number": chunk.page_number,
                    "case_id": case_id,
                }

                all_chunk_ids.append(chunk.chunk_id)
                all_chunk_texts.append(chunk.text)
                all_metadatas.append(meta)

            all_chunks.extend(chunks)

    logger.info("Total chunks collected: %d", len(all_chunk_ids))

    if not all_chunk_ids:
        logger.error(
            "No chunks produced. Ensure tar files are extracted for "
            "years %d-%d using extract-judgments.ps1",
            start_year,
            end_year,
        )
        return

    # ---- Step 3: Build dual index ----
    logger.info("=== Step 3: Building dual index (incremental) ===")
    dual = DualIndex()
    dual.add_chunks(all_chunk_ids, all_chunk_texts, all_metadatas)
    dual.save()
    logger.info("Dual index updated and saved")

    # ---- Step 4: Sample query with RRF ----
    logger.info("=== Step 4: Sample query with RRF fusion ===")
    sample_query = (
        "criminal appeal murder Section 302 IPC dying declaration"
    )
    dpr_results, chroma_results = dual.query(sample_query)
    fused = rrf_fuse([dpr_results, chroma_results], k=settings.RRF_K)

    logger.info("--- Top 10 RRF results for query: '%s' ---", sample_query)
    for rank, (chunk_id, rrf_score) in enumerate(fused[:10], start=1):
        # Retrieve chunk text and metadata from ChromaDB.
        text = dual.chroma.get_document(chunk_id)
        meta = dual.chroma.get_metadata(chunk_id)
        snippet = text[:200].replace("\n", " ") if text else "(no text)"
        logger.info(
            "Rank %2d | RRF=%.6f | chunk=%s | pdf=%s | page=%s | "
            "section=%s\n         %s",
            rank,
            rrf_score,
            chunk_id,
            meta.get("source_pdf", "?"),
            meta.get("page_number", "?"),
            meta.get("section_type", "?"),
            snippet,
        )

    logger.info("=== Phase 1 pipeline complete ===")


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="COUNTERCASE Phase 1 pipeline"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2024,
        help="First year to process (inclusive, default: 2024)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last year to process (inclusive, default: 2025)",
    )
    args = parser.parse_args()
    run_pipeline(args.start_year, args.end_year)


if __name__ == "__main__":
    main()
