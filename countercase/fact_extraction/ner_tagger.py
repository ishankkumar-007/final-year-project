"""Rule-based NER tagger for perturbation candidate identification.

Tags legally operative spans in case text: monetary amounts, ages,
durations, legal sections, party roles, and evidence types.  Designed
as a swappable component -- the interface is identical whether using
regex rules or a fine-tuned BERT NER model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Legal entity types for perturbation candidate tagging."""

    MONETARY_AMOUNT = "MONETARY_AMOUNT"
    AGE = "AGE"
    DURATION = "DURATION"
    LEGAL_SECTION = "LEGAL_SECTION"
    PARTY_ROLE = "PARTY_ROLE"
    EVIDENCE_TYPE = "EVIDENCE_TYPE"


@dataclass
class TaggedSpan:
    """A tagged span in the source text.

    Attributes:
        text: The matched text.
        start: Start character offset (inclusive).
        end: End character offset (exclusive).
        entity_type: The classified entity type.
    """

    text: str
    start: int
    end: int
    entity_type: EntityType


# -----------------------------------------------------------------------
# Regex patterns for each entity type
# -----------------------------------------------------------------------

_MONETARY_PATTERNS: list[re.Pattern[str]] = [
    # Rs. / Rs / INR followed by number (with comma separators) and
    # optional lakh/crore/thousand multiplier
    re.compile(
        r"(?:Rs\.?|INR|rupees?)\s*"
        r"[\d,]+(?:\.\d+)?"
        r"(?:\s*(?:lakh|lakhs|crore|crores|thousand|million|billion))?",
        re.IGNORECASE,
    ),
    # Standalone "X lakh/crore" (with number before the unit)
    re.compile(
        r"\b\d[\d,]*(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores)\b",
        re.IGNORECASE,
    ),
]

_AGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\baged?\s+(?:about\s+)?\d+\s*(?:years?)?\b", re.IGNORECASE),
    re.compile(r"\b\d+\s+years?\s+old\b", re.IGNORECASE),
    re.compile(r"\bminor\b", re.IGNORECASE),
    re.compile(r"\bmajor\b(?!\s+(?:general|part|portion|role))", re.IGNORECASE),
    re.compile(r"\bage\s+of\s+\d+\b", re.IGNORECASE),
]

_DURATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b\d+\s*(?:years?|months?|days?|weeks?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bperiod\s+of\s+\d+\s*(?:years?|months?|days?|weeks?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+\s*(?:years?|months?|days?)\s+(?:and|&)\s+"
        r"\d+\s*(?:years?|months?|days?)\b",
        re.IGNORECASE,
    ),
]

_LEGAL_SECTION_PATTERNS: list[re.Pattern[str]] = [
    # "Section 302 of IPC", "Section 498A", "S. 302"
    re.compile(
        r"\b(?:Section|Sec\.?|S\.)\s*\d+[A-Za-z]?"
        r"(?:\s*(?:of\s+(?:the\s+)?)?(?:IPC|CPC|CrPC|Cr\.P\.C|"
        r"Indian\s+Penal\s+Code|Evidence\s+Act|"
        r"Code\s+of\s+Criminal\s+Procedure|"
        r"Code\s+of\s+Civil\s+Procedure|"
        r"Motor\s+Vehicles?\s+Act|"
        r"Limitation\s+Act|"
        r"POCSO|NDPS|Prevention\s+of\s+Corruption\s+Act|"
        r"Companies\s+Act|Income[\s-]*Tax\s+Act|"
        r"Negotiable\s+Instruments?\s+Act|NI\s+Act|"
        r"Specific\s+Relief\s+Act|Transfer\s+of\s+Property\s+Act|"
        r"Hindu\s+Marriage\s+Act|Hindu\s+Succession\s+Act|"
        r"Domestic\s+Violence\s+Act|SC/ST\s+Act))?",
        re.IGNORECASE,
    ),
    # "IPC 302", "CPC 151", "CrPC 482"
    re.compile(
        r"\b(?:IPC|CPC|CrPC|Cr\.P\.C|POCSO|NDPS)\s*(?:Section\s*)?\d+[A-Za-z]?\b",
        re.IGNORECASE,
    ),
    # "Article 14", "Art. 21", "Article 226"
    re.compile(
        r"\b(?:Article|Art\.?)\s*\d+[A-Za-z]?"
        r"(?:\s*(?:of\s+(?:the\s+)?)?(?:Constitution|Constitution\s+of\s+India))?\b",
        re.IGNORECASE,
    ),
    # "Order VII Rule 11", "Order 39 Rule 1"
    re.compile(
        r"\bOrder\s+(?:[IVXLC]+|\d+)\s*(?:Rule\s+\d+[A-Za-z]?)?\b",
        re.IGNORECASE,
    ),
]

_PARTY_ROLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:petitioner|respondent|appellant|complainant|accused|"
        r"plaintiff|defendant|applicant|opposite\s+party|"
        r"prosecution|defence|informant)\b",
        re.IGNORECASE,
    ),
    # Closed party-type vocabulary matches
    re.compile(
        r"\b(?:corporation|partnership|state\s+government|"
        r"union\s+of\s+india|foreign\s+national|"
        r"public\s+sector|private\s+sector|"
        r"employee|contractor|tenant|licensee)\b",
        re.IGNORECASE,
    ),
]

_EVIDENCE_TYPE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:dying\s+declaration|confessional?\s+statement|confession|"
        r"FIR|first\s+information\s+report|"
        r"post[\s-]*mortem(?:\s+report)?|autopsy\s+report|"
        r"medical\s+report|medical\s+evidence|medico[\s-]*legal|"
        r"eye[\s-]*witness|witness\s+statement|"
        r"panchnama|panchanama|spot\s+inspection|"
        r"documentary\s+evidence|"
        r"electronic\s+evidence|digital\s+evidence|"
        r"circumstantial\s+evidence|"
        r"recovery\s+memo|seizure\s+memo|"
        r"expert\s+opinion|forensic\s+report)\b",
        re.IGNORECASE,
    ),
]

# Mapping entity type -> list of compiled patterns
_ENTITY_PATTERNS: dict[EntityType, list[re.Pattern[str]]] = {
    EntityType.MONETARY_AMOUNT: _MONETARY_PATTERNS,
    EntityType.AGE: _AGE_PATTERNS,
    EntityType.DURATION: _DURATION_PATTERNS,
    EntityType.LEGAL_SECTION: _LEGAL_SECTION_PATTERNS,
    EntityType.PARTY_ROLE: _PARTY_ROLE_PATTERNS,
    EntityType.EVIDENCE_TYPE: _EVIDENCE_TYPE_PATTERNS,
}


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

# Type alias for any tagger function (regex or model-based).
TaggerFn = Callable[[str], list[TaggedSpan]]


def tag_perturbation_candidates(fact_text: str) -> list[TaggedSpan]:
    """Tag perturbation candidate spans using rule-based regex NER.

    Args:
        fact_text: Raw text of the facts section.

    Returns:
        List of tagged spans sorted by start offset.
    """
    if not fact_text:
        return []

    spans: list[TaggedSpan] = []
    seen_offsets: set[tuple[int, int]] = set()

    for entity_type, patterns in _ENTITY_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(fact_text):
                key = (match.start(), match.end())
                if key in seen_offsets:
                    continue
                seen_offsets.add(key)
                spans.append(
                    TaggedSpan(
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        entity_type=entity_type,
                    )
                )

    spans.sort(key=lambda s: s.start)
    return spans
