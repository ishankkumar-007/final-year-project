"""Metadata extraction from parquet files and judgment text.

Provides functions to load per-year metadata from the ADX dataset parquet
files, inspect the schema, and extract structured metadata from raw
judgment text using regex heuristics.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from countercase.config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parquet-based metadata loading
# ---------------------------------------------------------------------------

def load_metadata(start_year: int, end_year: int) -> pd.DataFrame:
    """Load and combine parquet metadata for a range of years.

    Args:
        start_year: First year (inclusive).
        end_year: Last year (inclusive).

    Returns:
        A combined DataFrame of metadata across the requested years.
    """
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        parquet_path = settings.METADATA_DIR / f"year={year}" / "metadata.parquet"
        if not parquet_path.exists():
            logger.warning("Parquet file not found for year %d: %s", year, parquet_path)
            continue
        try:
            df = pd.read_parquet(parquet_path)
            df["_year"] = year
            frames.append(df)
            logger.info("Loaded %d rows for year %d", len(df), year)
        except Exception:
            logger.exception("Failed to read parquet for year %d", year)

    if not frames:
        logger.warning("No metadata loaded for years %d-%d", start_year, end_year)
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined metadata: %d rows, %d columns", len(combined), len(combined.columns))
    return combined


def inspect_metadata_schema(
    start_year: int,
    end_year: int,
    save_path: Path | None = None,
) -> dict[str, Any]:
    """Inspect parquet metadata schema and save results.

    Args:
        start_year: First year (inclusive).
        end_year: Last year (inclusive).
        save_path: Optional path to save the inspection JSON.  Defaults to
            ``countercase/data/metadata_inspection.json``.

    Returns:
        A dict mapping column names to dtype, null count, and sample values.
    """
    df = load_metadata(start_year, end_year)
    if df.empty:
        return {}

    inspection: dict[str, Any] = {}
    for col in df.columns:
        sample_values = df[col].dropna().head(3).tolist()
        # Convert non-serializable types
        safe_samples: list[Any] = []
        for v in sample_values:
            try:
                json.dumps(v)
                safe_samples.append(v)
            except (TypeError, ValueError):
                safe_samples.append(str(v))

        inspection[col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "non_null_count": int(df[col].notna().sum()),
            "sample_values": safe_samples,
        }

    if save_path is None:
        save_path = settings.DATA_OUTPUT_DIR / "metadata_inspection.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(inspection, fh, indent=2, ensure_ascii=False, default=str)
    logger.info("Metadata inspection saved to %s", save_path)
    return inspection


# ---------------------------------------------------------------------------
# Regex-based metadata extraction from judgment text
# ---------------------------------------------------------------------------

_CASE_ID_PATTERN = re.compile(
    r"(?:Criminal|Civil|Special Leave|Writ|Transfer|Review|Contempt|Original)"
    r"\s+(?:Appeal|Petition|Case|Application|Suit)"
    r"(?:\s*\(?\w*\)?)?\s*(?:No\.?\s*)?"
    r"(\d[\d\-/]*\d)\s*/\s*(\d{4})",
    re.IGNORECASE,
)

_YEAR_PATTERN = re.compile(
    r"\b(?:dated|decided\s+on|date\s+of\s+(?:judgment|order))\b[:\s]*"
    r"(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})",
    re.IGNORECASE,
)

_YEAR_FALLBACK = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")

_SECTION_PATTERN = re.compile(
    r"(?:Section|Sec\.|S\.)\s*(\d+[A-Za-z]?)"
    r"(?:\s+(?:of\s+(?:the\s+)?)?(\w[\w\s]*?)(?=\s*[,;.\n]))?|"
    r"Article\s+(\d+[A-Za-z]?)",
    re.IGNORECASE,
)

_JUDGE_PATTERN = re.compile(
    r"(?:Hon['\u2019]?ble|Justice|J\.)\s+"
    r"([A-Z][a-zA-Z.\s\-]+?)(?=\s*(?:,|and\b|&|\n|J\.|$))",
    re.IGNORECASE,
)

_OUTCOME_KEYWORDS: dict[str, str] = {
    "appeal allowed": "Allowed",
    "appeal is allowed": "Allowed",
    "appeals allowed": "Allowed",
    "appeals are allowed": "Allowed",
    "appeal dismissed": "Dismissed",
    "appeal is dismissed": "Dismissed",
    "appeals dismissed": "Dismissed",
    "appeals are dismissed": "Dismissed",
    "petition allowed": "Allowed",
    "petition is allowed": "Allowed",
    "petition dismissed": "Dismissed",
    "petition is dismissed": "Dismissed",
    "petition disposed": "Disposed",
    "petition is disposed": "Disposed",
    "writ petition is allowed": "Allowed",
    "writ petition is dismissed": "Dismissed",
    "disposed of": "Disposed",
    "partly allowed": "Partly Allowed",
}


def _classify_bench(judge_count: int) -> str:
    """Return bench type string from judge count.

    Args:
        judge_count: Number of judges on the bench.

    Returns:
        A string describing the bench composition.
    """
    if judge_count <= 0:
        return "Unknown"
    if judge_count == 1:
        return "Single"
    if judge_count == 2:
        return "Division"
    if judge_count == 3:
        return "Three-Judge"
    return "Constitution"


def extract_metadata_from_text(full_text: str) -> dict[str, Any]:
    """Extract structured metadata from raw judgment text using regex.

    Fields extracted: case_id, year, act_sections, judge_names,
    bench_type, outcome_label.

    Args:
        full_text: The full text of a judgment.

    Returns:
        A dict with the extracted metadata fields.  Missing fields are
        set to ``None`` or empty lists.
    """
    result: dict[str, Any] = {
        "case_id": None,
        "year": None,
        "act_sections": [],
        "judge_names": [],
        "bench_type": "Unknown",
        "outcome_label": "Unknown",
    }

    # -- case_id --
    m = _CASE_ID_PATTERN.search(full_text[:3000])
    if m:
        number_part = m.group(1)
        year_part = m.group(2)
        prefix = full_text[m.start(): m.end()]
        # Rebuild a cleaner case_id
        prefix_clean = prefix.split(number_part)[0].strip()
        result["case_id"] = f"{prefix_clean} {number_part}/{year_part}".strip()

    # -- year --
    ym = _YEAR_PATTERN.search(full_text[:5000])
    if ym:
        result["year"] = int(ym.group(3))
    else:
        # Fallback: first 4-digit year in [1950, 2026]
        for fm in _YEAR_FALLBACK.finditer(full_text[:5000]):
            candidate = int(fm.group(1))
            if 1950 <= candidate <= 2026:
                result["year"] = candidate
                break

    # -- act_sections --
    sections: list[str] = []
    for sm in _SECTION_PATTERN.finditer(full_text):
        if sm.group(3):
            sections.append(f"Article {sm.group(3)}")
        else:
            sec_num = sm.group(1)
            act_name = (sm.group(2) or "").strip()
            if act_name:
                sections.append(f"Section {sec_num} {act_name}")
            else:
                sections.append(f"Section {sec_num}")
    result["act_sections"] = sorted(set(sections))

    # -- judge_names --
    judges: list[str] = []
    header_text = full_text[:3000]
    for jm in _JUDGE_PATTERN.finditer(header_text):
        name = jm.group(1).strip().rstrip(".")
        if len(name) > 2 and name not in judges:
            judges.append(name)
    result["judge_names"] = judges
    result["bench_type"] = _classify_bench(len(judges))

    # -- outcome_label --
    tail_start = max(0, int(len(full_text) * 0.8))
    tail_text = full_text[tail_start:].lower()
    for keyword, label in _OUTCOME_KEYWORDS.items():
        if keyword in tail_text:
            result["outcome_label"] = label
            break

    return result
