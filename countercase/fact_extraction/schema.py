"""Pydantic models for the structured fact sheet schema.

Defines the canonical data structure for extracted case facts.  All
perturbation, retrieval, and explanation modules operate on these
models.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, field_validator

# -- Closed vocabularies ------------------------------------------------

PARTY_TYPES: tuple[str, ...] = (
    "Individual",
    "Corporation",
    "State",
    "UnionOfIndia",
    "Minor",
    "Partnership",
    "ForeignNational",
    "PublicSector",
    "PrivateSector",
    "Employee",
    "Contractor",
    "Tenant",
    "Licensee",
    "Unknown",
)

EVIDENCE_TYPES: tuple[str, ...] = (
    "DyingDeclaration",
    "Confession",
    "MedicalReport",
    "Document",
    "Witness",
    "FIR",
    "PostMortem",
    "ExpertOpinion",
    "ElectronicRecord",
    "RecoveryMemo",
    "SiteInspection",
    "Circumstantial",
)

# Regex for normalized section citations.
# Accepts patterns like "IPC-302", "Constitution-Article-21",
# "CPC-Order-7-Rule-11", "Evidence-Section-32".
SECTION_PATTERN: re.Pattern[str] = re.compile(
    r"^[A-Za-z][\w]*-(?:Section-|Article-|Order-|Rule-|S-|Art-)?\d[\w\-]*$"
)


# -- Models -------------------------------------------------------------


class PartyInfo(BaseModel):
    """Petitioner and respondent type classification."""

    petitioner_type: str | None = None
    respondent_type: str | None = None

    @field_validator("petitioner_type", "respondent_type", mode="before")
    @classmethod
    def _validate_party_type(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if v not in PARTY_TYPES:
            return "Unknown"
        return v


class EvidenceItem(BaseModel):
    """A single piece of evidence referenced in the facts."""

    evidence_type: str
    description: str

    @field_validator("evidence_type", mode="before")
    @classmethod
    def _validate_evidence_type(cls, v: str) -> str:
        if v not in EVIDENCE_TYPES:
            # Accept it anyway but normalise to closest match later
            return v
        return v


class NumericalFacts(BaseModel):
    """Quantitative facts extracted from a case."""

    amounts: list[dict[str, Any]] = []
    ages: list[dict[str, Any]] = []
    durations: list[dict[str, Any]] = []


class FactSheet(BaseModel):
    """Complete structured fact sheet for a single case.

    Attributes:
        case_id: Unique case identifier.
        parties: Petitioner and respondent type info.
        evidence_items: List of evidence pieces mentioned in the facts.
        sections_cited: Normalised legal section strings.
        numerical_facts: Amounts, ages, and durations.
        outcome: Disposition label or None.
    """

    case_id: str
    parties: PartyInfo = PartyInfo()
    evidence_items: list[EvidenceItem] = []
    sections_cited: list[str] = []
    numerical_facts: NumericalFacts = NumericalFacts()
    outcome: str | None = None

    @field_validator("sections_cited", mode="before")
    @classmethod
    def _validate_sections(cls, v: list[str] | None) -> list[str]:
        if v is None:
            return []
        validated: list[str] = []
        for s in v:
            if SECTION_PATTERN.match(s):
                validated.append(s)
            else:
                # Try basic normalisation: strip whitespace
                cleaned = s.strip().replace(" ", "-")
                if SECTION_PATTERN.match(cleaned):
                    validated.append(cleaned)
                else:
                    # Accept raw but log-worthy -- keep it for downstream
                    validated.append(s)
        return validated
