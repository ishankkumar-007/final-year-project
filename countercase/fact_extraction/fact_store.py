"""JSON-file-based fact sheet storage.

Stores each :class:`~countercase.fact_extraction.schema.FactSheet` as
a single JSON file keyed by ``case_id``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from countercase.config.settings import settings
from countercase.fact_extraction.schema import FactSheet

logger = logging.getLogger(__name__)

FACT_STORE_DIR: Path = settings.DATA_OUTPUT_DIR / "fact_store"


def _ensure_dir() -> None:
    FACT_STORE_DIR.mkdir(parents=True, exist_ok=True)


def _path_for(case_id: str) -> Path:
    # Replace characters that are invalid in Windows filenames.
    safe_id = case_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return FACT_STORE_DIR / f"{safe_id}.json"


def save_fact_sheet(case_id: str, fact_sheet: FactSheet) -> Path:
    """Persist a fact sheet to disk.

    Args:
        case_id: Unique case identifier (used as filename stem).
        fact_sheet: The fact sheet to save.

    Returns:
        Path to the written JSON file.
    """
    _ensure_dir()
    path = _path_for(case_id)
    path.write_text(fact_sheet.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved fact sheet for %s -> %s", case_id, path)
    return path


def load_fact_sheet(case_id: str) -> FactSheet | None:
    """Load a previously saved fact sheet.

    Args:
        case_id: Case identifier used when saving.

    Returns:
        The loaded ``FactSheet``, or ``None`` if not found.
    """
    path = _path_for(case_id)
    if not path.exists():
        logger.debug("No fact sheet found for %s at %s", case_id, path)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return FactSheet.model_validate(data)
    except Exception:
        logger.exception("Failed to load fact sheet from %s", path)
        return None


def list_fact_sheets() -> list[str]:
    """Return a list of case_ids that have saved fact sheets."""
    _ensure_dir()
    return [
        p.stem for p in FACT_STORE_DIR.glob("*.json")
    ]
