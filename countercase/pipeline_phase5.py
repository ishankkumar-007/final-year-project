"""Phase 5 pipeline: multi-level tree, sensitivity scoring, export.

Usage:
    python -m countercase.pipeline_phase5 [--max-depth 3] [--max-children 5]
    python -m countercase.pipeline_phase5 --case-id "Criminal Appeal 1031/2024"

To launch the Streamlit UI:
    streamlit run countercase/app/streamlit_app.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Sample fact sheet (reused from Phase 4)
# -------------------------------------------------------------------

def _make_sample_fact_sheet() -> Any:
    from countercase.fact_extraction.schema import (
        EvidenceItem,
        FactSheet,
        NumericalFacts,
        PartyInfo,
    )

    return FactSheet(
        case_id="Criminal Appeal 1031/2024",
        parties=PartyInfo(
            petitioner_type="Individual",
            respondent_type="State",
        ),
        evidence_items=[
            EvidenceItem(evidence_type="FIR", description="FIR No. 123/2020"),
            EvidenceItem(
                evidence_type="MedicalReport",
                description="Post-mortem report of the deceased",
            ),
            EvidenceItem(
                evidence_type="Witness",
                description="Eyewitness testimony of PW-1",
            ),
        ],
        sections_cited=["IPC-302", "IPC-34", "IPC-304"],
        numerical_facts=NumericalFacts(
            amounts=[
                {"value": 500000.0, "unit": "rupees", "context": "compensation"},
            ],
            ages=[{"value": 25, "descriptor": "accused"}],
            durations=[
                {"value": 3.0, "unit": "years", "context": "imprisonment term"},
            ],
        ),
        outcome="appeal dismissed",
    )


# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

def run_pipeline(
    case_id: str | None = None,
    max_depth: int = 3,
    max_children: int = 5,
    min_displacement: float = 1.0,
) -> None:
    """End-to-end Phase 5 pipeline."""
    from countercase.counterfactual.llm_validator import (
        PerturbationValidator,
        mock_validation_llm_fn,
    )
    from countercase.counterfactual.perturbation_tree import PerturbationTree
    from countercase.counterfactual.sensitivity import (
        compute_diff,
        compute_per_case_sensitivity,
        compute_sensitivity_scores,
    )

    # -----------------------------------------------------------------
    # Step 1: Get or build fact sheet
    # -----------------------------------------------------------------
    print("\n=== Step 1: Fact sheet ===")
    fact_sheet = None

    if case_id:
        from countercase.fact_extraction.fact_store import load_fact_sheet
        fact_sheet = load_fact_sheet(case_id)
        if fact_sheet:
            print(f"Loaded fact sheet for: {case_id}")

    if fact_sheet is None:
        fact_sheet = _make_sample_fact_sheet()
        print(f"Using sample fact sheet: {fact_sheet.case_id}")

    print(f"  Parties: {fact_sheet.parties.petitioner_type} v {fact_sheet.parties.respondent_type}")
    print(f"  Sections: {fact_sheet.sections_cited}")
    print(f"  Evidence: {[e.evidence_type for e in fact_sheet.evidence_items]}")

    # -----------------------------------------------------------------
    # Step 2: Build tree root
    # -----------------------------------------------------------------
    print("\n=== Step 2: Build perturbation tree root ===")
    tree = PerturbationTree(retriever=None, top_k=10)
    root_id = tree.build_root(fact_sheet)
    print(f"Root node {root_id} created")

    # -----------------------------------------------------------------
    # Step 3: Multi-level expansion
    # -----------------------------------------------------------------
    print(f"\n=== Step 3: Expand tree (depth {max_depth}, max_children {max_children}) ===")
    validator = PerturbationValidator(llm_fn=mock_validation_llm_fn)
    t_start = time.perf_counter()

    tree.expand_tree(
        validator=validator,
        max_depth=max_depth,
        max_children_per_node=max_children,
        min_displacement_threshold=min_displacement,
    )

    t_elapsed = time.perf_counter() - t_start
    print(f"Tree expansion complete: {tree.node_count} total nodes in {t_elapsed:.1f}s")
    print(f"Validator stats: {validator.stats()}")

    # -----------------------------------------------------------------
    # Step 4: Tree structure summary
    # -----------------------------------------------------------------
    print("\n=== Step 4: Tree structure ===")
    _print_tree(tree, root_id, indent=0)

    # Depth distribution
    depth_counts: dict[int, int] = {}
    for nid in tree._nodes:
        d = tree.get_depth(nid)
        depth_counts[d] = depth_counts.get(d, 0) + 1
    print(f"\nDepth distribution: {dict(sorted(depth_counts.items()))}")

    # -----------------------------------------------------------------
    # Step 5: Sensitivity scores
    # -----------------------------------------------------------------
    print("\n=== Step 5: Sensitivity scores ===")
    scores = compute_sensitivity_scores(tree, k=10)
    for fact_type, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fact_type}: {score:.4f}")

    if scores:
        top_fact = max(scores, key=scores.get)
        print(
            f"\n  Most operative fact dimension: {top_fact} "
            f"(score: {scores[top_fact]:.4f})"
        )

    # -----------------------------------------------------------------
    # Step 6: Diff for Level 1 children
    # -----------------------------------------------------------------
    print("\n=== Step 6: Diffs for root's children ===")
    root_node = tree.get_node(root_id)
    for child_id in root_node.children_ids:
        child = tree.get_node(child_id)
        diff = compute_diff(
            root_node.retrieval_results or [],
            child.retrieval_results or [],
            k=10,
        )
        print(
            f"  Node {child_id} ({child.edge.description}): "
            f"dropped={len(diff.dropped_cases)}, "
            f"new={len(diff.new_cases)}, "
            f"stable={len(diff.stable_cases)}, "
            f"mean_disp={diff.mean_displacement:.2f}"
        )

    # -----------------------------------------------------------------
    # Step 7: Manual node editing demo
    # -----------------------------------------------------------------
    print("\n=== Step 7: Manual node editing demo ===")
    from countercase.fact_extraction.schema import (
        FactSheet,
        NumericalFacts,
        PartyInfo,
    )

    edited_fs = FactSheet(
        case_id=fact_sheet.case_id,
        parties=PartyInfo(
            petitioner_type="Minor",
            respondent_type="State",
        ),
        evidence_items=fact_sheet.evidence_items,
        sections_cited=fact_sheet.sections_cited,
        numerical_facts=fact_sheet.numerical_facts,
        outcome=fact_sheet.outcome,
    )
    manual_id = tree.add_manual_node(
        parent_id=root_id,
        edited_fact_sheet=edited_fs,
        description="Manual: changed petitioner from Individual to Minor",
    )
    print(f"Created manual node {manual_id} under root")
    print(f"Total nodes after manual edit: {tree.node_count}")

    # -----------------------------------------------------------------
    # Step 8: Export
    # -----------------------------------------------------------------
    print("\n=== Step 8: Export tree ===")
    output_dir = Path("countercase/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase5_tree.json"

    tree_json = tree.to_json()
    tree_str = json.dumps(tree_json, indent=2, default=str)
    output_path.write_text(tree_str, encoding="utf-8")
    print(f"Tree saved to {output_path} ({len(tree_str):,} bytes)")

    # Round-trip verification
    tree2 = PerturbationTree.from_json(json.loads(tree_str))
    assert tree2.node_count == tree.node_count, "Round-trip node count mismatch"
    print(f"Round-trip verification: {tree2.node_count} nodes (OK)")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n=== Phase 5 pipeline complete ===")
    print(f"  Total nodes: {tree.node_count}")
    print(f"  Depth levels: {sorted(depth_counts.keys())}")
    print(f"  Sensitivity scores: {scores}")
    print(f"\nTo launch the Streamlit UI:")
    print(f"  streamlit run countercase/app/streamlit_app.py")


def _print_tree(tree: Any, node_id: int, indent: int) -> None:
    """Print tree structure recursively."""
    node = tree.get_node(node_id)
    prefix = "  " * indent
    n_results = len(node.retrieval_results) if node.retrieval_results else 0
    if node.edge:
        label = f"{node.edge.description} ({n_results} results)"
    else:
        label = f"Root ({n_results} results)"
    n_children = len(node.children_ids)
    print(f"{prefix}[{node_id}] {label} [{n_children} children]")
    for child_id in node.children_ids:
        _print_tree(tree, child_id, indent + 1)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: Multi-level tree, sensitivity, and export",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Load fact sheet from the fact store by case_id",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum tree depth (default: 3)",
    )
    parser.add_argument(
        "--max-children",
        type=int,
        default=5,
        help="Maximum children per node (default: 5)",
    )
    parser.add_argument(
        "--min-displacement",
        type=float,
        default=1.0,
        help="Minimum mean displacement to expand a node (default: 1.0)",
    )

    args = parser.parse_args()
    run_pipeline(
        case_id=args.case_id,
        max_depth=args.max_depth,
        max_children=args.max_children,
        min_displacement=args.min_displacement,
    )


if __name__ == "__main__":
    main()
