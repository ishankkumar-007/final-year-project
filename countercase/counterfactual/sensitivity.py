"""Sensitivity scoring and diff computation for perturbation nodes.

Compares retrieval result sets between parent and child nodes to
quantify the impact of each fact change.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from countercase.counterfactual.perturbation_tree import PerturbationTree

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Diff result
# -------------------------------------------------------------------

@dataclass
class DiffResult:
    """Comparison between a parent and child node's result sets.

    Attributes:
        dropped_cases: Case IDs in parent top-K but not child top-K.
        new_cases: Case IDs in child top-K but not parent top-K.
        stable_cases: Case IDs in both top-K sets.
        rank_displacements: Mapping of case_id to absolute rank displacement.
        mean_displacement: Average displacement across all cases.
    """

    dropped_cases: list[str] = field(default_factory=list)
    new_cases: list[str] = field(default_factory=list)
    stable_cases: list[str] = field(default_factory=list)
    rank_displacements: dict[str, float] = field(default_factory=dict)
    mean_displacement: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "dropped_cases": self.dropped_cases,
            "new_cases": self.new_cases,
            "stable_cases": self.stable_cases,
            "rank_displacements": self.rank_displacements,
            "mean_displacement": self.mean_displacement,
        }


# -------------------------------------------------------------------
# Diff computation
# -------------------------------------------------------------------

def _extract_case_ids(
    results: list[Any],
    k: int,
) -> list[str]:
    """Extract case_id list from retrieval results, limited to top k.

    Handles both RetrievalResult objects (with .case_id attribute)
    and plain dicts.
    """
    case_ids: list[str] = []
    seen: set[str] = set()

    for r in results[:k]:
        if hasattr(r, "case_id"):
            cid = r.case_id
        elif isinstance(r, dict):
            cid = r.get("case_id", "")
        else:
            continue

        if cid and cid not in seen:
            case_ids.append(cid)
            seen.add(cid)

    return case_ids


def compute_diff(
    parent_results: list[Any],
    child_results: list[Any],
    k: int = 10,
) -> DiffResult:
    """Compute the diff between parent and child retrieval result sets.

    For cases in both sets, displacement = ``|rank_parent - rank_child|``.
    For cases in only one set, displacement = ``|rank - (K + 1)|``.

    Args:
        parent_results: Retrieval results from the parent node.
        child_results: Retrieval results from the child node.
        k: Number of top results to compare.

    Returns:
        A :class:`DiffResult` with dropped, new, stable cases and
        rank displacements.
    """
    parent_ids = _extract_case_ids(parent_results, k)
    child_ids = _extract_case_ids(child_results, k)

    parent_set = set(parent_ids)
    child_set = set(child_ids)

    # Build rank maps (1-indexed)
    parent_ranks: dict[str, int] = {
        cid: rank + 1 for rank, cid in enumerate(parent_ids)
    }
    child_ranks: dict[str, int] = {
        cid: rank + 1 for rank, cid in enumerate(child_ids)
    }

    dropped = [cid for cid in parent_ids if cid not in child_set]
    new = [cid for cid in child_ids if cid not in parent_set]
    stable = [cid for cid in parent_ids if cid in child_set]

    # Compute rank displacements
    displacements: dict[str, float] = {}
    all_case_ids = parent_set | child_set

    for cid in all_case_ids:
        p_rank = parent_ranks.get(cid, k + 1)
        c_rank = child_ranks.get(cid, k + 1)
        displacements[cid] = abs(p_rank - c_rank)

    mean_disp = 0.0
    if displacements:
        mean_disp = sum(displacements.values()) / len(displacements)

    return DiffResult(
        dropped_cases=dropped,
        new_cases=new,
        stable_cases=stable,
        rank_displacements=displacements,
        mean_displacement=mean_disp,
    )


# -------------------------------------------------------------------
# Aggregate sensitivity scoring
# -------------------------------------------------------------------

def compute_sensitivity_scores(
    tree: PerturbationTree,
    k: int = 10,
) -> dict[str, float]:
    """Compute aggregate sensitivity per fact dimension.

    For each fact type (Numerical, Section, PartyType, Evidence),
    collect all edges in the tree where that fact type was perturbed,
    compute the mean rank displacement across those edges, and return
    a dict mapping fact type name to its aggregate sensitivity score.

    Args:
        tree: A populated perturbation tree.
        k: Top-K size for diff computation.

    Returns:
        Dict mapping fact type name (str) to mean displacement (float).
    """
    displacements_by_type: dict[str, list[float]] = defaultdict(list)

    for node_id, edge in tree.get_all_edges():
        node = tree.get_node(node_id)
        parent_id = node.parent_id
        if parent_id is None:
            continue
        parent = tree.get_node(parent_id)

        parent_results = parent.retrieval_results or []
        child_results = node.retrieval_results or []

        if not parent_results and not child_results:
            continue

        diff = compute_diff(parent_results, child_results, k=k)
        displacements_by_type[edge.fact_type.value].append(
            diff.mean_displacement,
        )

    scores: dict[str, float] = {}
    for fact_type_name, values in displacements_by_type.items():
        scores[fact_type_name] = sum(values) / len(values) if values else 0.0

    # Ensure all four canonical fact types are present (with 0.0 default)
    for canonical in ("Numerical", "Section", "PartyType", "Evidence"):
        scores.setdefault(canonical, 0.0)

    logger.info("Sensitivity scores: %s", scores)
    return scores


def compute_per_case_sensitivity(
    tree: PerturbationTree,
    case_id: str,
    k: int = 10,
) -> dict[str, Any]:
    """Compute how a specific case's rank changes across perturbation paths.

    For every parent-child pair in the tree whose results contain
    *case_id*, record the rank in each set and the perturbation that
    caused the change.

    Args:
        tree: A populated perturbation tree.
        case_id: The case_id to track.
        k: Top-K size for rank extraction.

    Returns:
        Dict with:
            - ``appearances``: total count of nodes whose top-K contains
              the case.
            - ``rank_by_node``: dict[node_id, rank].
            - ``displacements``: list of dicts with edge description,
              parent rank, child rank, and displacement.
    """
    rank_by_node: dict[int, int] = {}

    for nid, node in tree._nodes.items():
        results = node.retrieval_results or []
        ids = _extract_case_ids(results, k)
        if case_id in ids:
            rank_by_node[nid] = ids.index(case_id) + 1

    displacements: list[dict[str, Any]] = []
    for node_id, edge in tree.get_all_edges():
        node = tree.get_node(node_id)
        parent_id = node.parent_id
        if parent_id is None:
            continue
        p_rank = rank_by_node.get(parent_id)
        c_rank = rank_by_node.get(node_id)
        if p_rank is None and c_rank is None:
            continue  # case not in either set
        p_rank_val = p_rank if p_rank is not None else k + 1
        c_rank_val = c_rank if c_rank is not None else k + 1
        disp = abs(p_rank_val - c_rank_val)
        status = "stable"
        if p_rank is not None and c_rank is None:
            status = "dropped"
        elif p_rank is None and c_rank is not None:
            status = "new"
        displacements.append({
            "edge_description": edge.description,
            "fact_type": edge.fact_type.value,
            "parent_rank": p_rank_val,
            "child_rank": c_rank_val,
            "displacement": disp,
            "status": status,
        })

    return {
        "case_id": case_id,
        "appearances": len(rank_by_node),
        "rank_by_node": rank_by_node,
        "displacements": displacements,
    }
