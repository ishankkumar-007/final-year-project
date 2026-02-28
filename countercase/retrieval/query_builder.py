"""Build natural-language retrieval queries from FactSheet objects.

The DPR question encoder was trained on natural English questions, so
keyword-style queries (e.g. ``IPC 302 murder Individual``) perform
poorly.  This module converts structured fact sheets into fluent query
paragraphs that better activate the encoder's learned representations.
"""

from __future__ import annotations

from countercase.fact_extraction.schema import FactSheet


def fact_sheet_to_query(fs: FactSheet) -> str:
    """Convert a *FactSheet* into a natural-language retrieval query.

    Produces a 2-4 sentence paragraph that reads like the facts summary
    of a legal judgment, which aligns with how the corpus chunks are
    written and how DPR expects queries to sound.
    """
    parts: list[str] = []

    # Opening sentence: party context
    pet = fs.parties.petitioner_type or "petitioner"
    resp = fs.parties.respondent_type or "respondent"
    parts.append(f"This case involves a {pet} against the {resp}.")

    # Legal provisions
    if fs.sections_cited:
        sections = ", ".join(fs.sections_cited[:8])
        parts.append(f"The relevant legal provisions are {sections}.")

    # Evidence summary
    if fs.evidence_items:
        ev_types = sorted({e.evidence_type for e in fs.evidence_items[:6]})
        parts.append(f"Evidence includes {', '.join(ev_types)}.")

    # Numerical facts
    nf = fs.numerical_facts
    num_parts: list[str] = []
    for a in nf.amounts[:2]:
        num_parts.append(
            f"{a.get('value', 0)} {a.get('unit', 'rupees')} ({a.get('context', '')})"
        )
    for a in nf.ages[:2]:
        num_parts.append(f"age {a.get('value', 0)} ({a.get('descriptor', '')})")
    for d in nf.durations[:2]:
        num_parts.append(
            f"{d.get('value', 0)} {d.get('unit', 'years')} ({d.get('context', '')})"
        )
    if num_parts:
        parts.append(f"Key facts include {', '.join(num_parts)}.")

    # Outcome
    if fs.outcome:
        parts.append(f"The outcome was: {fs.outcome}.")

    return " ".join(parts)
