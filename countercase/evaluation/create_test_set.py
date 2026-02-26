"""Test set creation tool (Phase 7.4).

Creates evaluation test sets for the retrieval and counterfactual
evaluation harnesses.  Supports two modes:

1. **Interactive annotation** – CLI wizard that lets a user search
   for cases, display their fact sheets, and annotate relevant
   precedents by case_id.

2. **Citation-based auto-ground-truth** – automatically builds a
   test set by extracting citation strings from judgment text and
   treating cited cases as relevant.  This produces a large but
   noisy ground truth that can be refined later.

Output format
-------------
JSON list conforming to the eval_harness test set schema:

.. code-block:: json

    [
        {
            "query_case_id": "Criminal Appeal 123/2015",
            "query_text": "<facts section text>",
            "relevant_case_ids": ["case_042", "case_057"]
        }
    ]
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Citation extraction patterns
# -------------------------------------------------------------------

# Matches common Indian law report citation strings:
#   (2015) 3 SCC 421
#   2015 (4) SCALE 312
#   AIR 2003 SC 1234
#   [1998] 1 SCR 150
_CITATION_YEAR_RE = re.compile(
    r"""
    (?:                          # Group 1: year patterns
        \((\d{4})\)              # (2015)
        |
        (\d{4})\s*\(             # 2015 (
        |
        \[(\d{4})\]              # [2015]
        |
        AIR\s+(\d{4})            # AIR 2003
    )
    """,
    re.VERBOSE,
)

# Broad citation string extractor – captures "X vs Y" or reporter refs
_CITE_PATTERN = re.compile(
    r"""
    (?:                                      # reporter style
        (?:\(?\d{4}\)?)\s*               # year
        [\(\[\s]*\d+[\)\]\s]*            # volume
        \s*(?:SCC|SCR|SCALE|AIR\s*SC|Bom\s*CR|All\s*ER)  # reporter
        \s*\d+                           # page
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Simple pattern: "Criminal Appeal No. 1234 of 2015"
_CASE_NUMBER_RE = re.compile(
    r"(?:Criminal|Civil|Writ|SLP|Transfer)\s+(?:Appeal|Petition|Case)"
    r"\s+(?:No\.?\s*)?(\d+)\s+(?:of|/)\s+(\d{4})",
    re.IGNORECASE,
)


def extract_citations(text: str) -> list[str]:
    """Extract case citation strings from judgment text.

    Returns a deduplicated list of raw citation strings found in *text*.
    These are not normalised to case_ids — downstream matching is needed.
    """
    citations: list[str] = []
    seen: set[str] = set()

    # Reporter citations
    for m in _CITE_PATTERN.finditer(text):
        raw = m.group(0).strip()
        if raw not in seen:
            citations.append(raw)
            seen.add(raw)

    # Case-number citations
    for m in _CASE_NUMBER_RE.finditer(text):
        raw = m.group(0).strip()
        if raw not in seen:
            citations.append(raw)
            seen.add(raw)

    return citations


def _citation_to_case_id(
    citation: str,
    index: dict[str, str] | None = None,
) -> str | None:
    """Attempt to map a raw citation string to a corpus case_id.

    If an *index* mapping ``citation -> case_id`` is provided, use it.
    Otherwise, try to construct a case_id from the case-number pattern.
    """
    if index and citation in index:
        return index[citation]

    m = _CASE_NUMBER_RE.search(citation)
    if m:
        num, year = m.group(1), m.group(2)
        category = citation.split()[0]  # Criminal / Civil / etc.
        return f"{category} Appeal {num}/{year}"

    return None


# -------------------------------------------------------------------
# Citation-based auto ground truth
# -------------------------------------------------------------------

def build_citation_test_set(
    case_texts: dict[str, str],
    fact_texts: dict[str, str] | None = None,
    citation_index: dict[str, str] | None = None,
    max_cases: int | None = None,
    min_relevant: int = 2,
) -> list[dict[str, Any]]:
    """Build a test set automatically from citations in judgment text.

    For each case, extract citations from its full text and resolve them
    to corpus case_ids.  Cases with fewer than *min_relevant* resolved
    citations are skipped.

    Args:
        case_texts: Mapping ``case_id -> full judgment text``.
        fact_texts: Mapping ``case_id -> facts section text`` (query text).
            If not provided, the first 2000 chars of the judgment are used.
        citation_index: Optional mapping ``raw citation -> case_id``.
        max_cases: Limit the number of test cases (None = all).
        min_relevant: Skip cases with fewer resolved relevant IDs.

    Returns:
        List of test-set entries.
    """
    corpus_ids = set(case_texts.keys())
    test_set: list[dict[str, Any]] = []

    for case_id, text in case_texts.items():
        if max_cases and len(test_set) >= max_cases:
            break

        raw_cites = extract_citations(text)
        resolved: list[str] = []
        for cite in raw_cites:
            rid = _citation_to_case_id(cite, citation_index)
            if rid and rid in corpus_ids and rid != case_id:
                resolved.append(rid)

        # Deduplicate
        resolved = list(dict.fromkeys(resolved))

        if len(resolved) < min_relevant:
            continue

        # Query text: facts section or first 2000 chars
        query_text = ""
        if fact_texts and case_id in fact_texts:
            query_text = fact_texts[case_id]
        else:
            query_text = text[:2000]

        test_set.append({
            "query_case_id": case_id,
            "query_text": query_text,
            "relevant_case_ids": resolved,
        })

    logger.info(
        "Citation-based test set: %d cases (from %d total)",
        len(test_set), len(case_texts),
    )
    return test_set


# -------------------------------------------------------------------
# Interactive annotation CLI
# -------------------------------------------------------------------

def interactive_annotation(
    case_ids: list[str],
    fact_texts: dict[str, str],
    output_path: str | Path,
) -> None:
    """Run an interactive CLI session for test-set annotation.

    Presents each case's facts to the user and prompts them to
    enter relevant case_ids (comma-separated).

    Args:
        case_ids: Ordered list of case_ids to annotate.
        fact_texts: Mapping ``case_id -> facts section text``.
        output_path: Path where the test set JSON will be saved.
    """
    output_path = Path(output_path)
    test_set: list[dict[str, Any]] = []

    # Load existing partial progress if file exists
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as fh:
            test_set = json.load(fh)
        done_ids = {e["query_case_id"] for e in test_set}
        case_ids = [cid for cid in case_ids if cid not in done_ids]
        print(f"Resuming: {len(test_set)} cases done, {len(case_ids)} remaining.\n")

    for idx, case_id in enumerate(case_ids, 1):
        print(f"\n{'=' * 60}")
        print(f"[{idx}/{len(case_ids)}] Case: {case_id}")
        print("=" * 60)

        facts = fact_texts.get(case_id, "(no fact text available)")
        # Show first 800 chars
        preview = facts[:800]
        if len(facts) > 800:
            preview += "\n... (truncated)"
        print(preview)
        print()

        response = input(
            "Enter relevant case_ids (comma-separated), "
            "'s' to skip, 'q' to quit: "
        ).strip()

        if response.lower() == "q":
            break
        if response.lower() == "s":
            continue

        relevant_ids = [
            cid.strip() for cid in response.split(",") if cid.strip()
        ]
        if relevant_ids:
            test_set.append({
                "query_case_id": case_id,
                "query_text": facts,
                "relevant_case_ids": relevant_ids,
            })
            # Save after each annotation
            _save_test_set(test_set, output_path)
            print(f"  -> Saved ({len(relevant_ids)} relevant IDs)")

    print(f"\nDone. {len(test_set)} annotated cases saved to {output_path}")


# -------------------------------------------------------------------
# Merge / split helpers
# -------------------------------------------------------------------

def merge_test_sets(*paths: str | Path) -> list[dict[str, Any]]:
    """Merge multiple test set files, deduplicating by query_case_id."""
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data:
            qid = entry["query_case_id"]
            if qid in merged:
                # Merge relevant_case_ids
                existing = set(merged[qid]["relevant_case_ids"])
                existing.update(entry["relevant_case_ids"])
                merged[qid]["relevant_case_ids"] = list(existing)
            else:
                merged[qid] = entry
    return list(merged.values())


def split_by_year(
    test_set: list[dict[str, Any]],
    cutoff_year: int = 2020,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a test set into train/test by year extracted from case_id.

    Cases with year >= cutoff_year go into the test split.
    """
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    year_re = re.compile(r"(\d{4})")
    for entry in test_set:
        m = year_re.search(entry["query_case_id"])
        if m and int(m.group(1)) >= cutoff_year:
            test.append(entry)
        else:
            train.append(entry)

    return train, test


# -------------------------------------------------------------------
# I/O
# -------------------------------------------------------------------

def _save_test_set(
    test_set: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write test set to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(test_set, fh, indent=2, ensure_ascii=False)


def save_test_set(
    test_set: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Write test set to JSON (public API)."""
    _save_test_set(test_set, Path(path))
    logger.info("Test set (%d entries) saved to %s", len(test_set), path)


def load_test_set(path: str | Path) -> list[dict[str, Any]]:
    """Load a test set from JSON."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded %d test entries from %s", len(data), path)
    return data


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main() -> None:
    """CLI entry point for test set creation."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Create evaluation test sets for COUNTERCASE",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Citation-based auto ground truth
    cite_parser = sub.add_parser(
        "citations",
        help="Build test set from extracted citations",
    )
    cite_parser.add_argument(
        "--case-texts",
        type=str,
        required=True,
        help="JSON mapping case_id -> judgment text",
    )
    cite_parser.add_argument(
        "--fact-texts",
        type=str,
        default=None,
        help="Optional JSON mapping case_id -> facts section text",
    )
    cite_parser.add_argument(
        "--output",
        type=str,
        default="countercase/evaluation/data/citation_test_set.json",
    )
    cite_parser.add_argument("--max-cases", type=int, default=None)
    cite_parser.add_argument("--min-relevant", type=int, default=2)

    # Merge sub-command
    merge_parser = sub.add_parser("merge", help="Merge multiple test sets")
    merge_parser.add_argument("files", nargs="+", help="Test set JSON files to merge")
    merge_parser.add_argument(
        "--output",
        type=str,
        default="countercase/evaluation/data/merged_test_set.json",
    )

    # Split sub-command
    split_parser = sub.add_parser("split", help="Split test set by year")
    split_parser.add_argument("file", help="Input test set JSON")
    split_parser.add_argument("--cutoff", type=int, default=2020)
    split_parser.add_argument(
        "--output-dir",
        type=str,
        default="countercase/evaluation/data",
    )

    args = parser.parse_args()

    if args.command == "citations":
        with open(args.case_texts, encoding="utf-8") as fh:
            case_texts = json.load(fh)
        fact_texts = None
        if args.fact_texts:
            with open(args.fact_texts, encoding="utf-8") as fh:
                fact_texts = json.load(fh)
        ts = build_citation_test_set(
            case_texts,
            fact_texts=fact_texts,
            max_cases=args.max_cases,
            min_relevant=args.min_relevant,
        )
        save_test_set(ts, args.output)
        print(f"Created test set with {len(ts)} cases -> {args.output}")

    elif args.command == "merge":
        merged = merge_test_sets(*args.files)
        save_test_set(merged, args.output)
        print(f"Merged {len(args.files)} files -> {len(merged)} cases")

    elif args.command == "split":
        ts = load_test_set(args.file)
        train, test = split_by_year(ts, cutoff_year=args.cutoff)
        out_dir = Path(args.output_dir)
        save_test_set(train, out_dir / "train_set.json")
        save_test_set(test, out_dir / "test_set.json")
        print(f"Split: {len(train)} train, {len(test)} test (cutoff={args.cutoff})")


if __name__ == "__main__":
    main()
