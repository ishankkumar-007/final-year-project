"""Quick smoke test for the Phase 1 pipeline on a small subset of PDFs."""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from countercase.config.settings import settings
from countercase.indexing.dual_index import DualIndex
from countercase.ingestion.metadata_extractor import extract_metadata_from_text
from countercase.ingestion.noise_filter import clean_text
from countercase.ingestion.pdf_extractor import extract_pdf
from countercase.ingestion.section_detector import detect_sections
from countercase.preprocessing.chunker import chunk_text
from countercase.preprocessing.section_tagger import tag_chunks
from countercase.retrieval.rrf import rrf_fuse


def main() -> None:
    pdf_dir = settings.DATA_DIR / "year=2024" / "english" / "english"
    pdf_files = sorted(pdf_dir.glob("*.pdf"))[:10]
    logger.info("Processing %d PDFs...", len(pdf_files))

    all_ids: list[str] = []
    all_texts: list[str] = []
    all_metas: list[dict] = []

    for pdf_file in pdf_files:
        pages = extract_pdf(str(pdf_file))
        if not pages:
            continue
        full_text = "\n\n".join(p.text for p in pages)
        full_text = clean_text(full_text)
        if not full_text.strip():
            continue

        case_id = pdf_file.stem
        text_meta = extract_metadata_from_text(full_text)
        sections = detect_sections(full_text)
        chunks = chunk_text(full_text, case_id, pdf_file.name)
        if sections:
            tag_chunks(chunks, sections)

        for c in chunks:
            all_ids.append(c.chunk_id)
            all_texts.append(c.text)
            all_metas.append({
                "year": text_meta.get("year") or 2024,
                "bench_type": text_meta.get("bench_type", "Unknown"),
                "act_sections": ", ".join(text_meta.get("act_sections", [])),
                "section_type": c.section_type,
                "outcome_label": text_meta.get("outcome_label", "Unknown"),
                "source_pdf": c.source_pdf,
                "page_number": c.page_number,
                "case_id": case_id,
            })

    logger.info("Total chunks: %d", len(all_ids))

    # Build dual index
    logger.info("Building dual index...")
    dual = DualIndex()
    dual.index_chunks(all_ids, all_texts, all_metas, dpr_batch_size=8)
    dual.save()
    logger.info("Index built and saved")

    # Query
    query = "criminal appeal murder Section 302 IPC"
    dpr_r, chroma_r = dual.query(query, top_k=10)
    fused = rrf_fuse([dpr_r, chroma_r], k=60)

    logger.info("Top 5 RRF results for: '%s'", query)
    for rank, (cid, score) in enumerate(fused[:5], 1):
        meta = dual.chroma.get_metadata(cid)
        text = dual.chroma.get_document(cid)[:120].replace("\n", " ")
        logger.info(
            "  %d. score=%.6f | %s | section=%s | pdf=%s",
            rank, score, cid,
            meta.get("section_type", "?"),
            meta.get("source_pdf", "?"),
        )
        logger.info("     %s...", text)

    logger.info("Phase 1 smoke test PASSED")


if __name__ == "__main__":
    main()
