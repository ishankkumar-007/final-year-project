"""Perturbation tree: directed acyclic graph of fact sheet states.

Each node holds a fact sheet state and (optionally) retrieval results.
Edges represent single-fact perturbations with metadata.  The tree is
the core data structure for counterfactual reasoning in COUNTERCASE.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from countercase.counterfactual.llm_validator import PerturbationValidator
from countercase.counterfactual.perturbation_rules import (
    FactType,
    PerturbationEdge,
    perturb_evidence,
    perturb_numerical,
    perturb_party_type,
    perturb_section,
)
from countercase.counterfactual.section_adjacency import ADJACENCY_MAP
from countercase.fact_extraction.ner_tagger import (
    EntityType,
    TaggedSpan,
    tag_perturbation_candidates,
)
from countercase.fact_extraction.schema import FactSheet

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Tree node
# -------------------------------------------------------------------

@dataclass
class TreeNode:
    """A single node in the perturbation tree.

    Attributes:
        node_id: Unique integer identifier.
        fact_sheet: The fact sheet state at this node.
        parent_id: ID of the parent node (None for root).
        edge: The perturbation edge from parent (None for root).
        retrieval_results: Ranked retrieval results at this node.
        children_ids: IDs of child nodes.
    """

    node_id: int
    fact_sheet: FactSheet
    parent_id: int | None = None
    edge: PerturbationEdge | None = None
    retrieval_results: list[Any] | None = None
    children_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        result_dicts = None
        if self.retrieval_results is not None:
            result_dicts = []
            for r in self.retrieval_results:
                if hasattr(r, "__dict__"):
                    result_dicts.append(
                        {k: v for k, v in r.__dict__.items()}
                    )
                elif isinstance(r, dict):
                    result_dicts.append(r)
                else:
                    result_dicts.append(str(r))

        return {
            "node_id": self.node_id,
            "fact_sheet": json.loads(self.fact_sheet.model_dump_json()),
            "parent_id": self.parent_id,
            "edge": self.edge.to_dict() if self.edge else None,
            "retrieval_results": result_dicts,
            "children_ids": list(self.children_ids),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TreeNode:
        """Deserialize from a dictionary."""
        edge = None
        if d.get("edge"):
            edge = PerturbationEdge.from_dict(d["edge"])

        return cls(
            node_id=d["node_id"],
            fact_sheet=FactSheet.model_validate(d["fact_sheet"]),
            parent_id=d.get("parent_id"),
            edge=edge,
            retrieval_results=d.get("retrieval_results"),
            children_ids=d.get("children_ids", []),
        )


# -------------------------------------------------------------------
# Retriever protocol (to avoid circular imports)
# -------------------------------------------------------------------

class _RetrieverProtocol:
    """Minimal protocol for the retriever interface.

    Avoids importing HybridRetriever directly so the tree module
    can be tested without the full retrieval stack.
    """

    def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> list[Any]:
        raise NotImplementedError


# -------------------------------------------------------------------
# Perturbation tree
# -------------------------------------------------------------------

class PerturbationTree:
    """Directed acyclic graph of fact sheet perturbations.

    Each node is a fact sheet state.  Edges are single-fact changes.
    The tree supports breadth-first expansion with LLM validation
    and retrieval at each node.

    Args:
        retriever: Object with a ``retrieve(query, top_k)`` method.
            If ``None``, retrieval is skipped (useful for testing
            the tree structure alone).
        top_k: Number of retrieval results per node.
    """

    def __init__(
        self,
        retriever: Any | None = None,
        top_k: int = 10,
    ) -> None:
        self._retriever = retriever
        self._top_k = top_k
        self._nodes: dict[int, TreeNode] = {}
        self._next_id = 0

    # -----------------------------------------------------------------
    # Node access
    # -----------------------------------------------------------------

    def get_node(self, node_id: int) -> TreeNode:
        """Return a tree node by ID.

        Raises:
            KeyError: If the node_id does not exist.
        """
        return self._nodes[node_id]

    def get_children(self, node_id: int) -> list[TreeNode]:
        """Return the child nodes of a given node."""
        node = self._nodes[node_id]
        return [self._nodes[cid] for cid in node.children_ids]

    @property
    def node_count(self) -> int:
        """Total number of nodes in the tree."""
        return len(self._nodes)

    @property
    def root_id(self) -> int | None:
        """ID of the root node, or None if tree is empty."""
        if not self._nodes:
            return None
        return 0

    # -----------------------------------------------------------------
    # Tree construction
    # -----------------------------------------------------------------

    def build_root(self, fact_sheet: FactSheet) -> int:
        """Create the root node and run retrieval on it.

        Args:
            fact_sheet: The original (unperturbed) fact sheet.

        Returns:
            The root node ID (always 0).
        """
        node_id = self._allocate_id()
        node = TreeNode(
            node_id=node_id,
            fact_sheet=fact_sheet,
            parent_id=None,
            edge=None,
        )
        node.retrieval_results = self._run_retrieval(fact_sheet)
        self._nodes[node_id] = node

        logger.info(
            "Built root node %d with %d retrieval results",
            node_id,
            len(node.retrieval_results) if node.retrieval_results else 0,
        )
        return node_id

    def expand_node(
        self,
        node_id: int,
        validator: PerturbationValidator,
        max_children: int = 5,
    ) -> list[int]:
        """Generate perturbations, validate, and create child nodes.

        Generates all perturbations from the four perturbation
        functions, validates each with the LLM validator, keeps the
        top ``max_children`` by acceptance, creates child nodes, and
        runs retrieval for each.

        Args:
            node_id: ID of the node to expand.
            validator: LLM validation filter.
            max_children: Maximum child nodes to create.

        Returns:
            List of newly created child node IDs.
        """
        parent = self._nodes[node_id]
        fact_sheet = parent.fact_sheet

        # Collect all candidate perturbations
        candidates: list[tuple[FactSheet, PerturbationEdge]] = []

        # NER-based candidates for numerical and section perturbations
        facts_text = _fact_sheet_to_query(fact_sheet)
        spans = tag_perturbation_candidates(facts_text)

        for span in spans:
            if span.entity_type in (
                EntityType.MONETARY_AMOUNT,
                EntityType.AGE,
                EntityType.DURATION,
            ):
                candidates.extend(perturb_numerical(span, fact_sheet))
            elif span.entity_type == EntityType.LEGAL_SECTION:
                candidates.extend(
                    perturb_section(span, fact_sheet, ADJACENCY_MAP)
                )

        # Section perturbations from sections_cited directly
        for section in fact_sheet.sections_cited:
            dummy_span = TaggedSpan(
                text=section, start=0, end=len(section),
                entity_type=EntityType.LEGAL_SECTION,
            )
            section_perturbs = perturb_section(
                dummy_span, fact_sheet, ADJACENCY_MAP,
            )
            # Avoid duplicates
            existing_keys = {
                (c[1].original_value, c[1].perturbed_value)
                for c in candidates
                if c[1].fact_type == FactType.Section
            }
            for fs, edge in section_perturbs:
                key = (edge.original_value, edge.perturbed_value)
                if key not in existing_keys:
                    candidates.append((fs, edge))
                    existing_keys.add(key)

        # Party type perturbations
        candidates.extend(perturb_party_type(fact_sheet))

        # Evidence perturbations
        candidates.extend(perturb_evidence(fact_sheet))

        logger.info(
            "Node %d: generated %d candidate perturbations",
            node_id,
            len(candidates),
        )

        # Validate and accept
        accepted: list[tuple[FactSheet, PerturbationEdge]] = []
        for perturbed_fs, edge in candidates:
            result = validator.validate(fact_sheet, perturbed_fs, edge)
            if result.accepted:
                accepted.append((perturbed_fs, edge))

        logger.info(
            "Node %d: %d accepted out of %d candidates (validator stats: %s)",
            node_id,
            len(accepted),
            len(candidates),
            validator.stats(),
        )

        # Limit to max_children
        children_to_create = accepted[:max_children]

        # Create child nodes
        child_ids: list[int] = []
        for perturbed_fs, edge in children_to_create:
            child_id = self._allocate_id()
            child = TreeNode(
                node_id=child_id,
                fact_sheet=perturbed_fs,
                parent_id=node_id,
                edge=edge,
            )
            child.retrieval_results = self._run_retrieval(perturbed_fs)
            self._nodes[child_id] = child
            parent.children_ids.append(child_id)
            child_ids.append(child_id)

            logger.info(
                "  Created child %d: %s (%d results)",
                child_id,
                edge.description,
                len(child.retrieval_results) if child.retrieval_results else 0,
            )

        return child_ids

    def expand_tree(
        self,
        validator: PerturbationValidator,
        max_depth: int = 3,
        max_children_per_node: int = 5,
        min_displacement_threshold: float = 1.0,
    ) -> None:
        """Breadth-first expansion of the tree up to *max_depth*.

        For each node at the current frontier, compute the diff vs its
        parent.  If the ``mean_displacement`` is below
        *min_displacement_threshold* the node is pruned (not expanded
        further).  Expansion is interruptible via ``KeyboardInterrupt``;
        on interrupt the partial tree is preserved.

        Args:
            validator: LLM validation filter.
            max_depth: Maximum tree depth (root is depth 0).
            max_children_per_node: Children cap per node.
            min_displacement_threshold: Minimum mean displacement to
                justify further expansion.
        """
        from countercase.counterfactual.sensitivity import compute_diff

        if self.root_id is None:
            logger.warning("expand_tree called on empty tree -- nothing to do")
            return

        # Frontier: list of (node_id, depth)
        frontier: deque[tuple[int, int]] = deque()
        root = self._nodes[self.root_id]

        # Depth 0 is the root -- expand it to get Level 1 children
        if not root.children_ids:
            logger.info("Expanding root to depth 1")
            t0 = time.perf_counter()
            try:
                self.expand_node(
                    self.root_id, validator, max_children=max_children_per_node,
                )
            except KeyboardInterrupt:
                logger.warning("Interrupted during root expansion -- returning partial tree")
                return
            elapsed = time.perf_counter() - t0
            logger.info(
                "Depth 1 expansion: %d children in %.1fs",
                len(root.children_ids),
                elapsed,
            )

        # Seed frontier with Level 1 nodes (depth 1)
        for cid in root.children_ids:
            frontier.append((cid, 1))

        current_depth = 1
        while frontier and current_depth < max_depth:
            next_frontier: deque[tuple[int, int]] = deque()
            level_start = time.perf_counter()
            expanded_count = 0
            pruned_count = 0

            while frontier:
                node_id, depth = frontier.popleft()
                if depth != current_depth:
                    # Belongs to a deeper level -- re-queue
                    next_frontier.append((node_id, depth))
                    continue

                node = self._nodes[node_id]
                parent = self._nodes[node.parent_id] if node.parent_id is not None else None

                # Compute diff vs parent for pruning decision
                if parent is not None and parent.retrieval_results and node.retrieval_results:
                    diff = compute_diff(
                        parent.retrieval_results,
                        node.retrieval_results,
                        k=self._top_k,
                    )
                    if diff.mean_displacement < min_displacement_threshold:
                        pruned_count += 1
                        logger.debug(
                            "Pruning node %d: mean_displacement=%.2f < %.2f",
                            node_id,
                            diff.mean_displacement,
                            min_displacement_threshold,
                        )
                        continue

                # Expand this node
                try:
                    new_children = self.expand_node(
                        node_id, validator, max_children=max_children_per_node,
                    )
                except KeyboardInterrupt:
                    logger.warning(
                        "Interrupted during expansion of node %d -- returning partial tree",
                        node_id,
                    )
                    return

                expanded_count += 1
                for child_id in new_children:
                    next_frontier.append((child_id, depth + 1))

            level_elapsed = time.perf_counter() - level_start
            logger.info(
                "Depth %d -> %d: expanded %d nodes, pruned %d, total nodes %d (%.1fs)",
                current_depth,
                current_depth + 1,
                expanded_count,
                pruned_count,
                self.node_count,
                level_elapsed,
            )

            frontier = next_frontier
            current_depth += 1

        logger.info(
            "Tree expansion complete: %d total nodes, max depth reached = %d",
            self.node_count,
            current_depth,
        )

    def add_manual_node(
        self,
        parent_id: int,
        edited_fact_sheet: FactSheet,
        description: str = "Manual edit",
    ) -> int:
        """Create a child node from a user-edited fact sheet.

        Runs retrieval on the edited fact sheet and attaches the node
        to the tree as a child of *parent_id*.

        Args:
            parent_id: ID of the parent node.
            edited_fact_sheet: User-modified fact sheet.
            description: Human-readable description of the edit.

        Returns:
            The new child node ID.
        """
        parent = self._nodes[parent_id]
        edge = PerturbationEdge(
            fact_type=FactType.Numerical,  # generic placeholder
            original_value="(manual)",
            perturbed_value="(manual)",
            description=description,
        )

        child_id = self._allocate_id()
        child = TreeNode(
            node_id=child_id,
            fact_sheet=edited_fact_sheet,
            parent_id=parent_id,
            edge=edge,
        )
        child.retrieval_results = self._run_retrieval(edited_fact_sheet)
        self._nodes[child_id] = child
        parent.children_ids.append(child_id)

        logger.info(
            "Manual node %d created under parent %d: %s (%d results)",
            child_id,
            parent_id,
            description,
            len(child.retrieval_results) if child.retrieval_results else 0,
        )
        return child_id

    def get_depth(self, node_id: int) -> int:
        """Return the depth of a node (root = 0)."""
        depth = 0
        nid = node_id
        while self._nodes[nid].parent_id is not None:
            nid = self._nodes[nid].parent_id
            depth += 1
        return depth

    def get_all_edges(self) -> list[tuple[int, PerturbationEdge]]:
        """Return all (node_id, edge) pairs for non-root nodes."""
        return [
            (node.node_id, node.edge)
            for node in self._nodes.values()
            if node.edge is not None
        ]

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Serialize the entire tree to a JSON-safe dictionary."""
        return {
            "node_count": self.node_count,
            "top_k": self._top_k,
            "nodes": {
                str(nid): node.to_dict()
                for nid, node in self._nodes.items()
            },
        }

    @classmethod
    def from_json(
        cls,
        data: dict[str, Any],
        retriever: Any | None = None,
    ) -> PerturbationTree:
        """Deserialize a tree from a JSON dictionary."""
        tree = cls(retriever=retriever, top_k=data.get("top_k", 10))
        for nid_str, node_dict in data.get("nodes", {}).items():
            node = TreeNode.from_dict(node_dict)
            tree._nodes[node.node_id] = node
            tree._next_id = max(tree._next_id, node.node_id + 1)
        return tree

    def to_json_string(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_json(), indent=indent, default=str)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _allocate_id(self) -> int:
        """Allocate the next node ID."""
        nid = self._next_id
        self._next_id += 1
        return nid

    def _run_retrieval(self, fact_sheet: FactSheet) -> list[Any]:
        """Run retrieval for a fact sheet state.

        Constructs a structured query string from the fact sheet
        and passes it through the retriever.
        """
        if self._retriever is None:
            return []

        query = _fact_sheet_to_query(fact_sheet)
        try:
            return self._retriever.retrieve(query, top_k=self._top_k)
        except Exception:
            logger.exception(
                "Retrieval failed for case %s", fact_sheet.case_id,
            )
            return []


# -------------------------------------------------------------------
# Query construction from fact sheet
# -------------------------------------------------------------------

def _fact_sheet_to_query(fact_sheet: FactSheet) -> str:
    """Convert a fact sheet to a structured query string for retrieval.

    Concatenates party types, sections cited, numerical facts summary,
    and evidence items into a single query string.
    """
    parts: list[str] = []

    # Party types
    pet = fact_sheet.parties.petitioner_type or "Unknown"
    resp = fact_sheet.parties.respondent_type or "Unknown"
    if pet != "Unknown" or resp != "Unknown":
        parts.append(f"Petitioner: {pet}, Respondent: {resp}")

    # Sections cited
    if fact_sheet.sections_cited:
        parts.append("Sections: " + ", ".join(fact_sheet.sections_cited))

    # Numerical facts
    nf = fact_sheet.numerical_facts
    if nf.amounts:
        amt_strs = [
            f"{a.get('value', 0)} {a.get('unit', 'rupees')}"
            for a in nf.amounts
        ]
        parts.append("Amounts: " + ", ".join(amt_strs))
    if nf.ages:
        age_strs = [
            f"age {a.get('value', 0)} ({a.get('descriptor', '')})"
            for a in nf.ages
        ]
        parts.append("Ages: " + ", ".join(age_strs))
    if nf.durations:
        dur_strs = [
            f"{d.get('value', 0)} {d.get('unit', 'years')}"
            for d in nf.durations
        ]
        parts.append("Durations: " + ", ".join(dur_strs))

    # Evidence items
    if fact_sheet.evidence_items:
        ev_strs = [e.evidence_type for e in fact_sheet.evidence_items]
        parts.append("Evidence: " + ", ".join(ev_strs))

    # Outcome
    if fact_sheet.outcome:
        parts.append(f"Outcome: {fact_sheet.outcome}")

    return ". ".join(parts) if parts else fact_sheet.case_id
