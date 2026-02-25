"""Section detection for Indian Supreme Court judgments.

Detects structural sections (Facts, Issues, Submissions, Analysis,
Held, Ratio, Obiter) using regex patterns on headings.  Falls back to
heuristic splitting when no headings are found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class SectionType(Enum):
    """Enumeration of recognized judgment section types."""

    Facts = "Facts"
    Issues = "Issues"
    Submissions = "Submissions"
    Analysis = "Analysis"
    Held = "Held"
    Ratio = "Ratio"
    Obiter = "Obiter"
    Unknown = "Unknown"


@dataclass
class Section:
    """A detected section within a judgment.

    Attributes:
        section_type: The classified type of this section.
        start_char: Start character offset in the full text.
        end_char: End character offset in the full text.
        text: The text content of this section.
    """

    section_type: SectionType
    start_char: int
    end_char: int
    text: str


# Mapping from heading keywords to section types.  Order matters:
# first match wins for headings that could belong to multiple types.
_HEADING_MAP: list[tuple[re.Pattern[str], SectionType]] = [
    # Facts
    (re.compile(
        r"(?:^|\n)\s*(?:[IVXLC]+\.?\s+)?"
        r"(?:FACTS?|FACTUAL\s+BACKGROUND|STATEMENT\s+OF\s+(?:THE\s+)?CASE|"
        r"BRIEF\s+FACTS?|BACKGROUND\s+FACTS?|FACTUAL\s+MATRIX)"
        r"\s*[:.\-]?\s*(?:\n|$)",
        re.IGNORECASE,
    ), SectionType.Facts),
    # Issues
    (re.compile(
        r"(?:^|\n)\s*(?:[IVXLC]+\.?\s+)?"
        r"(?:ISSUES?(?:\s+FOR\s+CONSIDERATION)?|QUESTIONS?\s+(?:OF\s+LAW|FOR\s+CONSIDERATION))"
        r"\s*[:.\-]?\s*(?:\n|$)",
        re.IGNORECASE,
    ), SectionType.Issues),
    # Submissions
    (re.compile(
        r"(?:^|\n)\s*(?:[IVXLC]+\.?\s+)?"
        r"(?:SUBMISSIONS?|ARGUMENTS?|CONTENTIONS?|"
        r"SUBMISSIONS?\s+(?:OF|ON\s+BEHALF\s+OF)\s+(?:THE\s+)?"
        r"(?:APPELLANT|RESPONDENT|PETITIONER|PARTIES))"
        r"\s*[:.\-]?\s*(?:\n|$)",
        re.IGNORECASE,
    ), SectionType.Submissions),
    # Analysis / Discussion / Reasoning
    (re.compile(
        r"(?:^|\n)\s*(?:[IVXLC]+\.?\s+)?"
        r"(?:ANALYSIS|DISCUSSION|REASONING|CONSIDERATION|OUR\s+VIEW|"
        r"ANALYSIS\s+AND\s+DISCUSSION)"
        r"\s*[:.\-]?\s*(?:\n|$)",
        re.IGNORECASE,
    ), SectionType.Analysis),
    # Held / Order / Conclusion / Result
    (re.compile(
        r"(?:^|\n)\s*(?:[IVXLC]+\.?\s+)?"
        r"(?:HELD|ORDER|CONCLUSION|RESULT|JUDGMENT|DECISION|"
        r"(?:THE\s+)?(?:COURT\s+)?(?:HELD|ORDERED|CONCLUDES?)|"
        r"(?:FINAL\s+)?ORDER)"
        r"\s*[:.\-]?\s*(?:\n|$)",
        re.IGNORECASE,
    ), SectionType.Held),
    # Ratio Decidendi
    (re.compile(
        r"(?:^|\n)\s*(?:[IVXLC]+\.?\s+)?"
        r"(?:RATIO\s+DECIDENDI|RATIO)"
        r"\s*[:.\-]?\s*(?:\n|$)",
        re.IGNORECASE,
    ), SectionType.Ratio),
    # Obiter Dicta
    (re.compile(
        r"(?:^|\n)\s*(?:[IVXLC]+\.?\s+)?"
        r"(?:OBITER\s+DICTA|OBITER)"
        r"\s*[:.\-]?\s*(?:\n|$)",
        re.IGNORECASE,
    ), SectionType.Obiter),
]


def _find_headings(full_text: str) -> list[tuple[int, int, SectionType]]:
    """Find all section headings in the text.

    Returns a list of (start, end, section_type) tuples sorted by start
    position.
    """
    headings: list[tuple[int, int, SectionType]] = []
    for pattern, stype in _HEADING_MAP:
        for m in pattern.finditer(full_text):
            headings.append((m.start(), m.end(), stype))

    # Sort by position so sections are in document order.
    headings.sort(key=lambda h: h[0])

    # Deduplicate overlapping headings (keep the first match).
    deduped: list[tuple[int, int, SectionType]] = []
    last_end = -1
    for start, end, stype in headings:
        if start >= last_end:
            deduped.append((start, end, stype))
            last_end = end
    return deduped


def detect_sections(full_text: str) -> list[Section]:
    """Detect structural sections in a judgment text.

    Uses regex patterns to locate section headings.  When headings are
    found, each section spans from that heading to the start of the
    next heading (or the end of the document).

    If no headings are found, the first 20% of the text is classified
    as Facts and the remainder as Unknown.

    Args:
        full_text: The full judgment text after noise removal.

    Returns:
        A list of ``Section`` objects in document order.
    """
    if not full_text or not full_text.strip():
        return []

    headings = _find_headings(full_text)

    if not headings:
        # Fallback: first 20% is Facts, rest is Unknown.
        split_point = int(len(full_text) * 0.2)
        return [
            Section(
                section_type=SectionType.Facts,
                start_char=0,
                end_char=split_point,
                text=full_text[:split_point],
            ),
            Section(
                section_type=SectionType.Unknown,
                start_char=split_point,
                end_char=len(full_text),
                text=full_text[split_point:],
            ),
        ]

    sections: list[Section] = []

    # If the first heading is not at position 0, there is text before the
    # first detected heading.  Classify it as Unknown.
    first_heading_start = headings[0][0]
    if first_heading_start > 0:
        leading_text = full_text[:first_heading_start].strip()
        if leading_text:
            sections.append(Section(
                section_type=SectionType.Unknown,
                start_char=0,
                end_char=first_heading_start,
                text=leading_text,
            ))

    for idx, (h_start, h_end, stype) in enumerate(headings):
        if idx + 1 < len(headings):
            section_end = headings[idx + 1][0]
        else:
            section_end = len(full_text)
        section_text = full_text[h_end:section_end].strip()
        if section_text:
            sections.append(Section(
                section_type=stype,
                start_char=h_start,
                end_char=section_end,
                text=section_text,
            ))

    return sections
