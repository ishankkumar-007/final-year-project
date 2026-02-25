"""Locate the Facts section within a judgment text.

Uses the Phase-1 section detector output when available, falling back
to heading-pattern heuristics and ultimately to a first-20% fallback.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from countercase.ingestion.section_detector import Section

logger = logging.getLogger(__name__)

# Heading patterns that mark the start of the facts section.
_FACTS_HEADINGS: re.Pattern[str] = re.compile(
    r"(?:^|\n)\s*(?:[IVXLC]+\.?\s+)?"
    r"(?:FACTS?|FACTUAL\s+BACKGROUND|STATEMENT\s+OF\s+(?:THE\s+)?CASE|"
    r"BRIEF\s+FACTS?|BACKGROUND\s+FACTS?|FACTUAL\s+MATRIX)"
    r"\s*[:.\-]?\s*(?:\n|$)",
    re.IGNORECASE,
)

# Heading patterns that typically follow the facts section.
_POST_FACTS_HEADINGS: re.Pattern[str] = re.compile(
    r"(?:^|\n)\s*(?:[IVXLC]+\.?\s+)?"
    r"(?:ISSUES?|SUBMISSIONS?|ARGUMENTS?|CONTENTIONS?|ANALYSIS|"
    r"DISCUSSION|REASONING|HELD|ORDER|CONCLUSION|"
    r"QUESTIONS?\s+(?:OF\s+LAW|FOR\s+CONSIDERATION))"
    r"\s*[:.\-]?\s*(?:\n|$)",
    re.IGNORECASE,
)


def locate_facts_section(
    full_text: str,
    sections: list[Section] | None = None,
) -> str:
    """Extract the facts section from a judgment.

    Strategy (in order of preference):
        1. If *sections* are provided and one has type ``Facts``,
           return its text directly.
        2. Search for facts-like heading patterns, then extract text
           between the heading and the next non-facts heading.
        3. Fallback: return the first 20 % of the judgment text.

    Args:
        full_text: Complete judgment text.
        sections: Pre-detected sections from
            :func:`countercase.ingestion.section_detector.detect_sections`.

    Returns:
        Raw text of the facts section.
    """
    if not full_text or not full_text.strip():
        return ""

    # -- Strategy 1: Use pre-detected sections --------------------------
    if sections:
        from countercase.ingestion.section_detector import SectionType

        for sec in sections:
            if sec.section_type == SectionType.Facts:
                logger.debug(
                    "Facts section found via detector (chars %d-%d)",
                    sec.start_char,
                    sec.end_char,
                )
                return sec.text

    # -- Strategy 2: Heading-pattern heuristic --------------------------
    facts_match = _FACTS_HEADINGS.search(full_text)
    if facts_match:
        start = facts_match.end()
        # Find the next non-facts heading after the facts heading.
        post_match = _POST_FACTS_HEADINGS.search(full_text, pos=start)
        end = post_match.start() if post_match else len(full_text)
        text = full_text[start:end].strip()
        if text:
            logger.debug(
                "Facts section found via heading heuristic (chars %d-%d)",
                start,
                end,
            )
            return text

    # -- Strategy 3: First 20 % fallback --------------------------------
    split_point = max(1, int(len(full_text) * 0.2))
    logger.debug(
        "No facts heading found; using first 20%% of text (%d chars)",
        split_point,
    )
    return full_text[:split_point].strip()
