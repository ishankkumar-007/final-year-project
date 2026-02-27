"""Phase 3 -- Fact Sheet Extraction and NER pipeline.

Extracts PDFs for a small year range, detects sections, locates the
facts section, runs the LLM-based fact sheet extractor on sample cases,
runs the NER tagger, and prints results.  Stores successful fact sheets
in the fact store.

Usage:
    python -m countercase.pipeline_phase3 [--start-year 2024] [--end-year 2025] [--max-cases 10]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from countercase.config.settings import settings
from countercase.fact_extraction.fact_sheet_extractor import (
    FactSheetExtractor,
    mock_llm_fn,
)
from countercase.fact_extraction.fact_store import (
    list_fact_sheets,
    load_fact_sheet,
    save_fact_sheet,
)
from countercase.fact_extraction.ner_tagger import tag_perturbation_candidates
from countercase.fact_extraction.schema import FactSheet
from countercase.fact_extraction.section_locator import locate_facts_section
from countercase.ingestion.noise_filter import clean_text
from countercase.ingestion.pdf_extractor import extract_directory, extract_pdf
from countercase.ingestion.section_detector import detect_sections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("__main__")

SEPARATOR = "=" * 72
THIN_SEP = "-" * 72


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _extract_pdfs_for_year(year: int) -> Path | None:
    """Return path to extracted PDFs for a year, or None."""
    english_dir = settings.DATA_DIR / f"year={year}" / "english" / "english"
    if english_dir.is_dir():
        return english_dir
    alt = settings.DATA_DIR / f"year={year}" / "english"
    if alt.is_dir() and any(alt.glob("*.pdf")):
        return alt
    return None


def _build_case_id(pdf_name: str) -> str:
    """Derive a case_id from the PDF filename."""
    return Path(pdf_name).stem


def _select_llm_fn():
    """Select the best available LLM function.

    Priority: API (if key set) > local (if transformers available) > mock.
    """
    import os

    if os.getenv("LLM_API_KEY"):
        from countercase.fact_extraction.fact_sheet_extractor import api_llm_fn

        logger.info("Using API LLM backend (LLM_API_KEY is set)")
        return api_llm_fn

    try:
        import transformers  # noqa: F401

        if not os.getenv("LOCAL_LLM_MODEL"):
            logger.info(
                "No LLM_API_KEY and no LOCAL_LLM_MODEL set; "
                "falling back to mock LLM"
            )
            return mock_llm_fn

        from countercase.fact_extraction.fact_sheet_extractor import local_llm_fn

        logger.info("Using local LLM backend")
        return local_llm_fn
    except ImportError:
        pass

    logger.info("Using mock LLM backend (no real LLM available)")
    return mock_llm_fn


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def run_pipeline(
    start_year: int,
    end_year: int,
    max_cases: int = 10,
) -> None:
    """Execute the Phase 3 pipeline.

    Args:
        start_year: First year (inclusive).
        end_year: Last year (inclusive).
        max_cases: Maximum number of cases to process.
    """
    print(f"\n{SEPARATOR}")
    print("  COUNTERCASE Phase 3 -- Fact Sheet Extraction and NER")
    print(SEPARATOR)

    # -- Step 1: Collect PDFs and their text ----------------------------
    print("\n  Step 1: Extracting PDF texts...")
    t0 = time.perf_counter()

    # Pre-load existing fact sheets so we can skip already-processed cases
    existing_ids = set(list_fact_sheets())
    skipped = 0

    case_texts: list[tuple[str, str, list]] = []  # (case_id, full_text, sections)

    for year in range(start_year, end_year + 1):
        pdf_dir = _extract_pdfs_for_year(year)
        if pdf_dir is None:
            logger.warning("No extracted PDFs for year %d", year)
            continue

        logger.info("Processing PDFs for year %d from %s", year, pdf_dir)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))

        for pdf_file in tqdm(
            pdf_files, desc=f"Year {year}", unit="pdf"
        ):
            if len(case_texts) >= max_cases:
                break

            case_id = _build_case_id(pdf_file.name)

            # Skip cases that already have a fact sheet in the store
            if case_id in existing_ids:
                skipped += 1
                continue

            try:
                pages = extract_pdf(str(pdf_file))
            except Exception as exc:
                logger.warning("Failed to extract %s: %s", pdf_file.name, exc)
                continue

            if not pages:
                continue

            full_text = "\n\n".join(p.text for p in pages)
            full_text = clean_text(full_text)

            if not full_text.strip():
                continue

            sections = detect_sections(full_text)
            case_texts.append((case_id, full_text, sections))

        if len(case_texts) >= max_cases:
            break

    t_extract = time.perf_counter() - t0
    print(f"  Collected {len(case_texts)} new cases in {t_extract:.1f}s")
    print(f"  Skipped {skipped} cases already in fact store\n")

    if not case_texts:
        print("  No cases found. Exiting.")
        return

    # -- Step 2: Section locator ----------------------------------------
    print(f"\n{THIN_SEP}")
    print("  Step 2: Locating facts sections")
    print(THIN_SEP)

    facts_found = 0
    case_facts: list[tuple[str, str]] = []  # (case_id, facts_text)

    for case_id, full_text, sections in case_texts:
        facts_text = locate_facts_section(full_text, sections)
        has_dedicated = any(
            s.section_type.value == "Facts" for s in sections
        )
        if facts_text:
            facts_found += 1
            case_facts.append((case_id, facts_text))
            print(
                f"  [{case_id}]  facts={len(facts_text)} chars  "
                f"detected_heading={'Yes' if has_dedicated else 'Fallback'}"
            )
        else:
            print(f"  [{case_id}]  facts=NONE")

    rate = facts_found / len(case_texts) * 100 if case_texts else 0
    print(f"\n  Facts section located: {facts_found}/{len(case_texts)} ({rate:.0f}%)")

    # -- Step 3: LLM-based fact sheet extraction ------------------------
    print(f"\n{THIN_SEP}")
    print("  Step 3: LLM-based fact sheet extraction")
    print(THIN_SEP)

    llm_fn = _select_llm_fn()
    extractor = FactSheetExtractor(llm_fn=llm_fn, timeout=60)

    successes = 0
    failures = 0
    extracted_sheets: list[tuple[str, FactSheet, str]] = []  # (case_id, sheet, facts)

    for case_id, facts_text in case_facts:
        t_start = time.perf_counter()
        sheet = extractor.extract(facts_text, case_id=case_id)
        t_elapsed = time.perf_counter() - t_start

        if sheet is not None:
            successes += 1
            extracted_sheets.append((case_id, sheet, facts_text))
            save_fact_sheet(case_id, sheet)
            print(f"\n  [{case_id}] EXTRACTED in {t_elapsed:.1f}s")
            print(f"    parties: {sheet.parties.petitioner_type} v {sheet.parties.respondent_type}")
            print(f"    sections: {sheet.sections_cited[:5]}")
            print(f"    evidence: {len(sheet.evidence_items)} items")
            print(f"    amounts: {len(sheet.numerical_facts.amounts)}, "
                  f"ages: {len(sheet.numerical_facts.ages)}, "
                  f"durations: {len(sheet.numerical_facts.durations)}")
            print(f"    outcome: {sheet.outcome}")
        else:
            failures += 1
            print(f"\n  [{case_id}] FAILED in {t_elapsed:.1f}s")

    print(f"\n  Extraction results: {successes} success, {failures} failures")

    # -- Step 4: NER tagging --------------------------------------------
    print(f"\n{THIN_SEP}")
    print("  Step 4: NER tagging for perturbation candidates")
    print(THIN_SEP)

    for case_id, sheet, facts_text in extracted_sheets[:5]:  # Show 5
        spans = tag_perturbation_candidates(facts_text)
        print(f"\n  [{case_id}] {len(spans)} tagged spans:")

        # Group by entity type for cleaner output
        by_type: dict[str, list[str]] = {}
        for span in spans:
            key = span.entity_type.value
            by_type.setdefault(key, []).append(
                f'"{span.text[:60]}" @{span.start}-{span.end}'
            )

        for etype, examples in sorted(by_type.items()):
            print(f"    {etype}: {len(examples)} spans")
            for ex in examples[:3]:
                print(f"      - {ex}")
            if len(examples) > 3:
                print(f"      ... and {len(examples) - 3} more")

    # -- Step 5: Fact store verification --------------------------------
    print(f"\n{THIN_SEP}")
    print("  Step 5: Fact store verification")
    print(THIN_SEP)

    stored_ids = list_fact_sheets()
    print(f"  Fact store contains {len(stored_ids)} fact sheets")

    if extracted_sheets:
        test_id = extracted_sheets[0][0]
        loaded = load_fact_sheet(test_id)
        if loaded is not None:
            print(f"  Round-trip test: loaded '{test_id}' successfully")
            assert loaded.case_id == extracted_sheets[0][1].case_id
            print(f"  Case ID match: {loaded.case_id}")
        else:
            print(f"  Round-trip test: FAILED to load '{test_id}'")

    # -- Summary --------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("  PHASE 3 SUMMARY")
    print(SEPARATOR)
    print(f"  Cases processed:        {len(case_texts)}")
    print(f"  Facts sections found:   {facts_found}/{len(case_texts)} ({rate:.0f}%)")
    print(f"  Fact sheets extracted:  {successes}")
    print(f"  Extraction failures:    {failures}")
    print(f"  Fact sheets in store:   {len(stored_ids)}")
    print()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 3 -- Fact Sheet Extraction and NER pipeline",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2024,
        help="First year to process (default: 2024)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last year to process (default: 2025)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=10,
        help="Maximum cases to process (default: 10)",
    )
    args = parser.parse_args()
    run_pipeline(args.start_year, args.end_year, args.max_cases)


if __name__ == "__main__":
    main()
