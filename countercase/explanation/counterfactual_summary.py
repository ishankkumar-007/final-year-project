"""Counterfactual explanation generator for perturbation edges.

Generates a one-paragraph summary for each parent-child edge in
the perturbation tree explaining what fact changed, which cases
dropped/appeared, and why.  Deterministic template filling -- no LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from countercase.counterfactual.sensitivity import DiffResult

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Result field accessor
# -------------------------------------------------------------------

def _field(result: Any, name: str, default: Any = "") -> Any:
    """Get a field from a result object or dict."""
    if hasattr(result, name):
        return getattr(result, name)
    if isinstance(result, dict):
        return result.get(name, default)
    return default


def _result_sections(result: Any) -> set[str]:
    """Extract act_sections as a set from a retrieval result."""
    raw = _field(result, "act_sections", "")
    if isinstance(raw, list):
        return set(raw)
    return {s.strip() for s in str(raw).split(",") if s.strip()}


def _build_result_map(results: list[Any], k: int) -> dict[str, Any]:
    """Map case_id -> result for top-k results."""
    seen: set[str] = set()
    mapping: dict[str, Any] = {}
    for r in results[:k]:
        cid = str(_field(r, "case_id", ""))
        if cid and cid not in seen:
            mapping[cid] = r
            seen.add(cid)
    return mapping


# -------------------------------------------------------------------
# Main explanation function
# -------------------------------------------------------------------

def explain_edge(
    parent_node: Any,
    child_node: Any,
    diff: DiffResult,
) -> str:
    """Generate a counterfactual explanation for an edge.

    The explanation follows the template from plan.md section 8:
      1. State the fact change.
      2. State dropped cases and why they dropped.
      3. State new cases and why they appeared.
      4. If no change, state that the fact is not operative.

    Args:
        parent_node: Parent TreeNode (or any object with ``edge``,
            ``fact_sheet``, and ``retrieval_results``).
        child_node: Child TreeNode.
        diff: Pre-computed DiffResult between parent and child.

    Returns:
        One-paragraph explanation string.
    """
    edge = child_node.edge if hasattr(child_node, "edge") else None
    if edge is None:
        return ""

    parts: list[str] = []

    # --- 1. State the fact change ---
    parts.append(f"When {edge.description},")

    n_dropped = len(diff.dropped_cases)
    n_new = len(diff.new_cases)

    # --- 2. Dropped cases ---
    if n_dropped > 0:
        case_list = ", ".join(diff.dropped_cases[:5])
        if n_dropped > 5:
            case_list += f" (and {n_dropped - 5} more)"
        parts.append(
            f"{n_dropped} precedent{'s' if n_dropped != 1 else ''} "
            f"dropped out of the top results, including {case_list}."
        )

        # Try to explain why they dropped
        parent_map = _build_result_map(
            getattr(parent_node, "retrieval_results", None) or [], k=20,
        )
        _explain_dropped(
            parts, diff.dropped_cases, parent_map, edge,
        )
    else:
        parts.append("no precedents dropped out of the top results.")

    # --- 3. New cases ---
    if n_new > 0:
        case_list = ", ".join(diff.new_cases[:5])
        if n_new > 5:
            case_list += f" (and {n_new - 5} more)"
        parts.append(
            f"{n_new} new precedent{'s' if n_new != 1 else ''} "
            f"became applicable, including {case_list}."
        )

        child_map = _build_result_map(
            getattr(child_node, "retrieval_results", None) or [], k=20,
        )
        _explain_new(
            parts, diff.new_cases, child_map, edge,
        )
    elif n_dropped > 0:
        parts.append("No new precedents appeared.")

    # --- 4. No change case ---
    if n_dropped == 0 and n_new == 0:
        parts.clear()
        parts.append(
            f"When {edge.description}, "
            f"this fact change did not significantly alter the retrieval "
            f"results, suggesting it is not a legally operative fact for "
            f"precedent applicability."
        )

    return " ".join(parts)


# -------------------------------------------------------------------
# Helpers for explaining dropped / new cases
# -------------------------------------------------------------------

def _explain_dropped(
    parts: list[str],
    dropped_ids: list[str],
    parent_map: dict[str, Any],
    edge: Any,
) -> None:
    """Append an explanation of why dropped cases left the results."""
    fact_type = edge.fact_type.value if hasattr(edge.fact_type, "value") else str(edge.fact_type)
    original_val = getattr(edge, "original_value", "")

    if fact_type == "Section" and original_val:
        # Check if dropped cases cited the original section
        citing_original: list[str] = []
        for cid in dropped_ids[:5]:
            result = parent_map.get(cid)
            if result is None:
                continue
            sections = _result_sections(result)
            if original_val in sections:
                citing_original.append(cid)

        if citing_original:
            parts.append(
                f"These cases cited {original_val}, which is no longer "
                f"relevant after the change."
            )
            return

    if fact_type == "PartyType" and original_val:
        parts.append(
            f"These cases involved a {original_val}, but the perturbed "
            f"case involves a different party type, which falls under "
            f"different legal principles."
        )
        return

    if fact_type == "Evidence" and original_val:
        parts.append(
            f"These cases relied on {original_val} evidence, "
            f"which was altered in this perturbation."
        )
        return

    if fact_type == "Numerical" and original_val:
        parts.append(
            f"These cases were applicable under the original "
            f"numerical value ({original_val}), which was changed."
        )
        return


def _explain_new(
    parts: list[str],
    new_ids: list[str],
    child_map: dict[str, Any],
    edge: Any,
) -> None:
    """Append an explanation of why new cases appeared."""
    fact_type = edge.fact_type.value if hasattr(edge.fact_type, "value") else str(edge.fact_type)
    perturbed_val = getattr(edge, "perturbed_value", "")

    if fact_type == "Section" and perturbed_val:
        citing_perturbed: list[str] = []
        for cid in new_ids[:5]:
            result = child_map.get(cid)
            if result is None:
                continue
            sections = _result_sections(result)
            if perturbed_val in sections:
                citing_perturbed.append(cid)

        if citing_perturbed:
            parts.append(
                f"These cases involve {perturbed_val}, which matches "
                f"the altered facts."
            )
            return

    if fact_type == "PartyType" and perturbed_val:
        parts.append(
            f"These cases involve a {perturbed_val}, which matches "
            f"the altered party type."
        )
        return

    if fact_type == "Evidence" and perturbed_val:
        parts.append(
            f"These cases involve {perturbed_val} evidence, which "
            f"matches the altered evidence profile."
        )
        return

    if fact_type == "Numerical" and perturbed_val:
        parts.append(
            f"These cases are applicable under the perturbed numerical "
            f"value ({perturbed_val})."
        )
        return
