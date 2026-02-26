"""Rule-based perturbation functions for each fact type.

Pure functions that take a fact sheet (and optionally a tagged span or
adjacency map) and return a list of perturbed fact sheet copies, each
with exactly one fact changed.  Every returned pair includes a
:class:`PerturbationEdge` describing the change.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from countercase.fact_extraction.ner_tagger import EntityType, TaggedSpan
from countercase.fact_extraction.schema import (
    EVIDENCE_TYPES,
    PARTY_TYPES,
    EvidenceItem,
    FactSheet,
    NumericalFacts,
    PartyInfo,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Edge metadata
# -------------------------------------------------------------------

class FactType(Enum):
    """Broad fact dimension being perturbed."""

    Numerical = "Numerical"
    Section = "Section"
    PartyType = "PartyType"
    Evidence = "Evidence"


@dataclass
class PerturbationEdge:
    """Metadata describing a single fact change between two tree nodes.

    Attributes:
        fact_type: Broad category of the changed fact.
        original_value: String representation of the original value.
        perturbed_value: String representation of the new value.
        description: Human-readable summary of the change.
    """

    fact_type: FactType
    original_value: str
    perturbed_value: str
    description: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to a JSON-safe dict."""
        return {
            "fact_type": self.fact_type.value,
            "original_value": self.original_value,
            "perturbed_value": self.perturbed_value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> PerturbationEdge:
        """Deserialize from a dict."""
        return cls(
            fact_type=FactType(d["fact_type"]),
            original_value=d["original_value"],
            perturbed_value=d["perturbed_value"],
            description=d["description"],
        )


# -------------------------------------------------------------------
# Age / duration / amount thresholds
# -------------------------------------------------------------------

_AGE_BOUNDARIES: list[int] = [7, 12, 18, 21, 60]

_LIMITATION_PERIODS: list[float] = [1.0, 3.0, 6.0, 12.0]

_MONETARY_THRESHOLDS: list[float] = [
    25_000.0,       # Small Causes Court limit
    100_000.0,      # Motor Vehicles Act compensation threshold
    500_000.0,      # CPC summary procedure threshold
    1_000_000.0,    # 10 lakh
    10_000_000.0,   # 1 crore
    100_000_000.0,  # 10 crore
]


# -------------------------------------------------------------------
# Party type swap axes
# -------------------------------------------------------------------

_PARTY_SWAP_AXES: list[tuple[str, str]] = [
    ("Individual", "Minor"),
    ("Employee", "Contractor"),
    ("Tenant", "Licensee"),
    ("Individual", "Corporation"),
    ("State", "UnionOfIndia"),
]


# -------------------------------------------------------------------
# Common evidence types for addition perturbations
# -------------------------------------------------------------------

_COMMON_EVIDENCE_TYPES: list[str] = [
    "DyingDeclaration",
    "Confession",
    "MedicalReport",
]


# ===================================================================
# Perturbation functions
# ===================================================================


def perturb_numerical(
    span: TaggedSpan,
    fact_sheet: FactSheet,
) -> list[tuple[FactSheet, PerturbationEdge]]:
    """Perturb a numerical fact by shifting across legal thresholds.

    For MONETARY_AMOUNT spans, shift above and below statutory
    thresholds.  For AGE spans, shift across age-of-majority and
    similar boundaries.  For DURATION spans, double/halve or cross
    limitation periods.

    Args:
        span: A tagged span of type MONETARY_AMOUNT, AGE, or DURATION.
        fact_sheet: The source fact sheet to perturb.

    Returns:
        List of (perturbed_fact_sheet, edge) tuples.
    """
    results: list[tuple[FactSheet, PerturbationEdge]] = []

    if span.entity_type == EntityType.MONETARY_AMOUNT:
        results.extend(_perturb_amounts(span, fact_sheet))
    elif span.entity_type == EntityType.AGE:
        results.extend(_perturb_ages(span, fact_sheet))
    elif span.entity_type == EntityType.DURATION:
        results.extend(_perturb_durations(span, fact_sheet))
    else:
        logger.warning(
            "perturb_numerical called with non-numerical span type: %s",
            span.entity_type,
        )

    return results


def perturb_section(
    span: TaggedSpan,
    fact_sheet: FactSheet,
    adjacency_map: dict[str, list[str]],
) -> list[tuple[FactSheet, PerturbationEdge]]:
    """Replace a cited section with each of its adjacency-map neighbours.

    Args:
        span: A tagged span of type LEGAL_SECTION.
        fact_sheet: The source fact sheet.
        adjacency_map: Mapping from normalised section string to
            list of neighbouring sections.

    Returns:
        List of (perturbed_fact_sheet, edge) tuples.
    """
    results: list[tuple[FactSheet, PerturbationEdge]] = []

    matched_section = _find_matching_section(span.text, fact_sheet.sections_cited)
    if matched_section is None:
        logger.debug(
            "Section span '%s' not found in fact sheet sections_cited", span.text,
        )
        return results

    neighbours = adjacency_map.get(matched_section, [])
    if not neighbours:
        # Try normalised lookup
        normalised = _normalise_section_key(matched_section)
        neighbours = adjacency_map.get(normalised, [])

    for neighbour in neighbours:
        new_sections = [
            neighbour if s == matched_section else s
            for s in fact_sheet.sections_cited
        ]
        new_fs = _copy_fact_sheet(fact_sheet, sections_cited=new_sections)
        edge = PerturbationEdge(
            fact_type=FactType.Section,
            original_value=matched_section,
            perturbed_value=neighbour,
            description=(
                f"Section changed from {matched_section} to {neighbour}"
            ),
        )
        results.append((new_fs, edge))

    return results


def perturb_party_type(
    fact_sheet: FactSheet,
) -> list[tuple[FactSheet, PerturbationEdge]]:
    """Swap party types along predefined legal axes.

    For each party (petitioner, respondent), if its type appears on a
    swap axis, generate a version with the swapped type.

    Args:
        fact_sheet: The source fact sheet.

    Returns:
        List of (perturbed_fact_sheet, edge) tuples.
    """
    results: list[tuple[FactSheet, PerturbationEdge]] = []

    for role, current in [
        ("petitioner", fact_sheet.parties.petitioner_type),
        ("respondent", fact_sheet.parties.respondent_type),
    ]:
        if current is None or current == "Unknown":
            continue

        for a, b in _PARTY_SWAP_AXES:
            target: str | None = None
            if current == a:
                target = b
            elif current == b:
                target = a

            if target is None:
                continue

            if role == "petitioner":
                new_parties = PartyInfo(
                    petitioner_type=target,
                    respondent_type=fact_sheet.parties.respondent_type,
                )
            else:
                new_parties = PartyInfo(
                    petitioner_type=fact_sheet.parties.petitioner_type,
                    respondent_type=target,
                )

            new_fs = _copy_fact_sheet(fact_sheet, parties=new_parties)
            edge = PerturbationEdge(
                fact_type=FactType.PartyType,
                original_value=f"{role}={current}",
                perturbed_value=f"{role}={target}",
                description=(
                    f"{role.capitalize()} type changed from "
                    f"{current} to {target}"
                ),
            )
            results.append((new_fs, edge))

    return results


def perturb_evidence(
    fact_sheet: FactSheet,
) -> list[tuple[FactSheet, PerturbationEdge]]:
    """Toggle evidence items: remove existing, add plausible missing ones.

    For each evidence item present, create a version with it removed.
    For each common evidence type not present, create a version with
    it added.

    Args:
        fact_sheet: The source fact sheet.

    Returns:
        List of (perturbed_fact_sheet, edge) tuples.
    """
    results: list[tuple[FactSheet, PerturbationEdge]] = []

    # Removal perturbations
    for i, item in enumerate(fact_sheet.evidence_items):
        new_evidence = [e for j, e in enumerate(fact_sheet.evidence_items) if j != i]
        new_fs = _copy_fact_sheet(fact_sheet, evidence_items=new_evidence)
        edge = PerturbationEdge(
            fact_type=FactType.Evidence,
            original_value=f"present:{item.evidence_type}",
            perturbed_value=f"removed:{item.evidence_type}",
            description=f"Evidence removed: {item.evidence_type} ({item.description[:60]})",
        )
        results.append((new_fs, edge))

    # Addition perturbations
    present_types = {e.evidence_type for e in fact_sheet.evidence_items}
    for etype in _COMMON_EVIDENCE_TYPES:
        if etype in present_types:
            continue
        new_item = EvidenceItem(
            evidence_type=etype,
            description=f"Hypothetical {etype} added for perturbation analysis",
        )
        new_evidence = list(fact_sheet.evidence_items) + [new_item]
        new_fs = _copy_fact_sheet(fact_sheet, evidence_items=new_evidence)
        edge = PerturbationEdge(
            fact_type=FactType.Evidence,
            original_value=f"absent:{etype}",
            perturbed_value=f"added:{etype}",
            description=f"Evidence added: {etype}",
        )
        results.append((new_fs, edge))

    return results


# ===================================================================
# Internal helpers
# ===================================================================


def _copy_fact_sheet(
    fs: FactSheet,
    **overrides: Any,
) -> FactSheet:
    """Deep-copy a fact sheet with optional field overrides."""
    data = fs.model_dump()
    data.update(overrides)
    # Convert nested models back to dicts for Pydantic
    if "parties" in overrides and isinstance(overrides["parties"], PartyInfo):
        data["parties"] = overrides["parties"].model_dump()
    if "evidence_items" in overrides:
        data["evidence_items"] = [
            e.model_dump() if isinstance(e, EvidenceItem) else e
            for e in overrides["evidence_items"]
        ]
    if "numerical_facts" in overrides and isinstance(
        overrides["numerical_facts"], NumericalFacts
    ):
        data["numerical_facts"] = overrides["numerical_facts"].model_dump()
    return FactSheet.model_validate(data)


def _extract_number_from_span(text: str) -> float | None:
    """Try to parse a numeric value from a tagged span text."""
    import re

    # Remove currency symbols and commas
    cleaned = re.sub(r"[Rs.,INR\s]", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return float(cleaned)
    except ValueError:
        pass

    # Extract first number-like substring
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))

    return None


def _find_matching_section(
    span_text: str,
    sections_cited: list[str],
) -> str | None:
    """Find the sections_cited entry that best matches a span's text.

    Tries exact match first, then normalised key match, then substring.
    """
    import re

    # Extract section number from span text
    m = re.search(r"(\d+[A-Za-z]?)", span_text)
    if not m:
        return None
    num = m.group(1)

    for section in sections_cited:
        if num in section:
            return section

    return None


def _normalise_section_key(section: str) -> str:
    """Normalise a section string for adjacency-map lookup."""
    return section.strip().upper().replace(" ", "-")


# -------------------------------------------------------------------
# Numerical sub-perturbations
# -------------------------------------------------------------------


def _perturb_amounts(
    span: TaggedSpan,
    fact_sheet: FactSheet,
) -> list[tuple[FactSheet, PerturbationEdge]]:
    """Shift monetary amounts across statutory thresholds."""
    results: list[tuple[FactSheet, PerturbationEdge]] = []
    value = _extract_number_from_span(span.text)
    if value is None or value <= 0:
        return results

    target_values: set[float] = set()
    for threshold in _MONETARY_THRESHOLDS:
        if value < threshold:
            target_values.add(threshold + 1)
        elif value >= threshold:
            target_values.add(max(1.0, threshold - 1))

    # Also add double and half
    target_values.add(value * 2)
    if value > 2:
        target_values.add(value / 2)

    # Remove the original
    target_values.discard(value)

    for new_val in sorted(target_values):
        new_amounts = _replace_amount_in_facts(
            fact_sheet.numerical_facts.amounts, value, new_val,
        )
        new_nf = NumericalFacts(
            amounts=new_amounts,
            ages=copy.deepcopy(fact_sheet.numerical_facts.ages),
            durations=copy.deepcopy(fact_sheet.numerical_facts.durations),
        )
        new_fs = _copy_fact_sheet(fact_sheet, numerical_facts=new_nf)
        edge = PerturbationEdge(
            fact_type=FactType.Numerical,
            original_value=f"amount={value}",
            perturbed_value=f"amount={new_val}",
            description=(
                f"Monetary amount changed from {value:,.0f} to {new_val:,.0f}"
            ),
        )
        results.append((new_fs, edge))

    return results


def _perturb_ages(
    span: TaggedSpan,
    fact_sheet: FactSheet,
) -> list[tuple[FactSheet, PerturbationEdge]]:
    """Shift ages across legally significant boundaries."""
    results: list[tuple[FactSheet, PerturbationEdge]] = []
    value = _extract_number_from_span(span.text)
    if value is None or value <= 0:
        return results

    age = int(value)
    target_ages: set[int] = set()

    for boundary in _AGE_BOUNDARIES:
        if age >= boundary:
            target_ages.add(boundary - 1)
        else:
            target_ages.add(boundary)

    target_ages.discard(age)

    for new_age in sorted(target_ages):
        new_ages = _replace_age_in_facts(
            fact_sheet.numerical_facts.ages, age, new_age,
        )
        new_nf = NumericalFacts(
            amounts=copy.deepcopy(fact_sheet.numerical_facts.amounts),
            ages=new_ages,
            durations=copy.deepcopy(fact_sheet.numerical_facts.durations),
        )
        new_fs = _copy_fact_sheet(fact_sheet, numerical_facts=new_nf)

        # Determine which boundary was crossed
        crossed = [
            b for b in _AGE_BOUNDARIES
            if (age >= b) != (new_age >= b)
        ]
        boundary_desc = ""
        if crossed:
            boundary_desc = f" (crossed boundary at {crossed[0]})"

        edge = PerturbationEdge(
            fact_type=FactType.Numerical,
            original_value=f"age={age}",
            perturbed_value=f"age={new_age}",
            description=f"Age changed from {age} to {new_age}{boundary_desc}",
        )
        results.append((new_fs, edge))

    return results


def _perturb_durations(
    span: TaggedSpan,
    fact_sheet: FactSheet,
) -> list[tuple[FactSheet, PerturbationEdge]]:
    """Double/halve durations or shift across limitation periods."""
    results: list[tuple[FactSheet, PerturbationEdge]] = []
    value = _extract_number_from_span(span.text)
    if value is None or value <= 0:
        return results

    target_durations: set[float] = set()

    # Double and halve
    target_durations.add(value * 2)
    if value > 0.5:
        target_durations.add(value / 2)

    # Cross limitation periods
    for period in _LIMITATION_PERIODS:
        if value < period:
            target_durations.add(period + 0.5)
        elif value >= period:
            target_durations.add(max(0.5, period - 0.5))

    target_durations.discard(value)

    for new_dur in sorted(target_durations):
        new_durs = _replace_duration_in_facts(
            fact_sheet.numerical_facts.durations, value, new_dur,
        )
        new_nf = NumericalFacts(
            amounts=copy.deepcopy(fact_sheet.numerical_facts.amounts),
            ages=copy.deepcopy(fact_sheet.numerical_facts.ages),
            durations=new_durs,
        )
        new_fs = _copy_fact_sheet(fact_sheet, numerical_facts=new_nf)

        crossed = [
            p for p in _LIMITATION_PERIODS
            if (value >= p) != (new_dur >= p)
        ]
        boundary_desc = ""
        if crossed:
            boundary_desc = f" (crossed limitation period at {crossed[0]} years)"

        edge = PerturbationEdge(
            fact_type=FactType.Numerical,
            original_value=f"duration={value}",
            perturbed_value=f"duration={new_dur}",
            description=(
                f"Duration changed from {value} to {new_dur}{boundary_desc}"
            ),
        )
        results.append((new_fs, edge))

    return results


# -------------------------------------------------------------------
# Helpers for replacing values in numerical_facts lists
# -------------------------------------------------------------------


def _replace_amount_in_facts(
    amounts: list[dict[str, Any]],
    old_val: float,
    new_val: float,
) -> list[dict[str, Any]]:
    """Replace the first matching amount value."""
    result = copy.deepcopy(amounts)
    for item in result:
        if abs(item.get("value", 0) - old_val) < 0.01:
            item["value"] = new_val
            return result
    # If no exact match, append a new entry
    result.append({"value": new_val, "unit": "rupees", "context": "perturbed"})
    return result


def _replace_age_in_facts(
    ages: list[dict[str, Any]],
    old_val: int,
    new_val: int,
) -> list[dict[str, Any]]:
    """Replace the first matching age value."""
    result = copy.deepcopy(ages)
    for item in result:
        if item.get("value") == old_val:
            item["value"] = new_val
            return result
    result.append({"value": new_val, "descriptor": "perturbed"})
    return result


def _replace_duration_in_facts(
    durations: list[dict[str, Any]],
    old_val: float,
    new_val: float,
) -> list[dict[str, Any]]:
    """Replace the first matching duration value."""
    result = copy.deepcopy(durations)
    for item in result:
        if abs(item.get("value", 0) - old_val) < 0.01:
            item["value"] = new_val
            return result
    result.append({"value": new_val, "unit": "years", "context": "perturbed"})
    return result
