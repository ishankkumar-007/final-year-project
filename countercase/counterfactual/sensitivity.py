"""Sensitivity scoring and diff computation for perturbation nodes.

Compares retrieval result sets between parent and child nodes to
quantify the impact of each fact change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

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
