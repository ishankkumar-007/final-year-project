"""Phase 4 -- Perturbation Logic and Single-Level Tree pipeline.

Demonstrates single-level counterfactual reasoning end-to-end:
extracts a fact sheet from a case, builds a perturbation tree root,
expands to depth 1, computes diffs, and prints sensitivity results.

Usage:
    python -m countercase.pipeline_phase4 [--case-id CASE_ID] [--text FILE] [--max-children 5]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from countercase.counterfactual.llm_validator import (
    PerturbationValidator,
    mock_validation_llm_fn,
)
from countercase.counterfactual.perturbation_rules import PerturbationEdge
from countercase.counterfactual.perturbation_tree import PerturbationTree
from countercase.counterfactual.section_adjacency import get_adjacent_sections
from countercase.counterfactual.sensitivity import DiffResult, compute_diff
from countercase.fact_extraction.fact_sheet_extractor import (
    FactSheetExtractor,
    mock_llm_fn,
)
from countercase.fact_extraction.fact_store import load_fact_sheet
from countercase.fact_extraction.schema import (
    EvidenceItem,
    FactSheet,
    NumericalFacts,
    PartyInfo,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("__main__")

SEPARATOR = "=" * 72
THIN_SEP = "-" * 72


# -------------------------------------------------------------------
# Sample fact sheet for demo when no case is available
# -------------------------------------------------------------------

def _make_sample_fact_sheet() -> FactSheet:
    """Create a realistic sample fact sheet for demonstration."""
    return FactSheet(
        case_id="Criminal Appeal 1031/2024",
        parties=PartyInfo(
            petitioner_type="Individual",
            respondent_type="State",
        ),
        evidence_items=[
            EvidenceItem(
                evidence_type="FIR",
                description="FIR No. 123/2020 filed at PS Sadar",
            ),
            EvidenceItem(
                evidence_type="MedicalReport",
                description="Post-mortem report indicating cause of death",
            ),
            EvidenceItem(
                evidence_type="Witness",
                description="Eyewitness testimony of PW-1 and PW-2",
            ),
        ],
        sections_cited=[
            "IPC-302",
            "IPC-34",
            "IPC-304",
            "Evidence-Section-27",
        ],
        numerical_facts=NumericalFacts(
            amounts=[
                {"value": 500000.0, "unit": "rupees", "context": "compensation"},
            ],
            ages=[
                {"value": 25, "descriptor": "accused"},
            ],
            durations=[
                {"value": 3.0, "unit": "years", "context": "imprisonment"},
            ],
        ),
        outcome="Convicted",
    )


# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

def run_pipeline(
    case_id: str | None = None,
    text_file: str | None = None,
    max_children: int = 5,
) -> None:
    """Execute the Phase 4 pipeline.

    Args:
        case_id: Load fact sheet from the fact store by case_id.
        text_file: Path to a text file to extract a fact sheet from.
        max_children: Maximum child perturbations per node.
    """
    print(f"\n{SEPARATOR}")
    print("  COUNTERCASE Phase 4 -- Perturbation Logic and Single-Level Tree")
    print(SEPARATOR)

    # -- Step 1: Get or build fact sheet --------------------------------
    print(f"\n{THIN_SEP}")
    print("  Step 1: Obtaining fact sheet")
    print(THIN_SEP)

    fact_sheet: FactSheet | None = None

    if case_id:
        fact_sheet = load_fact_sheet(case_id)
        if fact_sheet:
            print(f"  Loaded fact sheet for: {case_id}")
        else:
            print(f"  No fact sheet found for: {case_id}")

    if fact_sheet is None and text_file:
        text_path = Path(text_file)
        if text_path.exists():
            text = text_path.read_text(encoding="utf-8")
            extractor = FactSheetExtractor(llm_fn=mock_llm_fn)
            fact_sheet = extractor.extract(text, case_id=text_path.stem)
            if fact_sheet:
                print(f"  Extracted fact sheet from: {text_file}")
            else:
                print(f"  Failed to extract fact sheet from: {text_file}")

    if fact_sheet is None:
        print("  Using sample fact sheet for demonstration")
        fact_sheet = _make_sample_fact_sheet()

    print(f"\n  Case ID: {fact_sheet.case_id}")
    print(f"  Parties: {fact_sheet.parties.petitioner_type} v {fact_sheet.parties.respondent_type}")
    print(f"  Sections: {fact_sheet.sections_cited}")
    print(f"  Evidence: {[e.evidence_type for e in fact_sheet.evidence_items]}")
    print(f"  Amounts: {fact_sheet.numerical_facts.amounts}")
    print(f"  Ages: {fact_sheet.numerical_facts.ages}")
    print(f"  Durations: {fact_sheet.numerical_facts.durations}")
    print(f"  Outcome: {fact_sheet.outcome}")

    # -- Step 2: Build perturbation tree root ---------------------------
    print(f"\n{THIN_SEP}")
    print("  Step 2: Building perturbation tree root")
    print(THIN_SEP)

    t0 = time.perf_counter()

    # No retriever -- we test tree structure and perturbation logic here.
    # Phase 5+ will plug in the HybridRetriever.
    tree = PerturbationTree(retriever=None, top_k=10)
    root_id = tree.build_root(fact_sheet)

    print(f"  Root node created (id={root_id})")
    print(f"  Retrieval results: {len(tree.get_node(root_id).retrieval_results or [])}")

    # -- Step 3: Expand to depth 1 -------------------------------------
    print(f"\n{THIN_SEP}")
    print("  Step 3: Expanding root to depth 1 (max_children={})".format(max_children))
    print(THIN_SEP)

    validator = PerturbationValidator(
        llm_fn=mock_validation_llm_fn,
        timeout=60,
    )

    child_ids = tree.expand_node(
        root_id,
        validator=validator,
        max_children=max_children,
    )

    t_expand = time.perf_counter() - t0
    print(f"\n  Expansion complete in {t_expand:.2f}s")
    print(f"  Created {len(child_ids)} child nodes")
    print(f"  Validator stats: {validator.stats()}")

    # -- Step 4: Display perturbations and diffs ------------------------
    print(f"\n{THIN_SEP}")
    print("  Step 4: Perturbation details and diff analysis")
    print(THIN_SEP)

    root_results = tree.get_node(root_id).retrieval_results or []

    for child_id in child_ids:
        child = tree.get_node(child_id)
        edge = child.edge
        child_results = child.retrieval_results or []

        diff = compute_diff(root_results, child_results, k=10)

        print(f"\n  --- Child {child_id} ---")
        print(f"  Fact type:  {edge.fact_type.value if edge else 'N/A'}")
        print(f"  Change:     {edge.description if edge else 'N/A'}")
        print(f"  Original:   {edge.original_value if edge else 'N/A'}")
        print(f"  Perturbed:  {edge.perturbed_value if edge else 'N/A'}")
        print(f"  Dropped:    {diff.dropped_cases}")
        print(f"  New:        {diff.new_cases}")
        print(f"  Stable:     {diff.stable_cases}")
        print(f"  Mean displacement: {diff.mean_displacement:.2f}")

    # -- Step 5: Section adjacency demo ---------------------------------
    print(f"\n{THIN_SEP}")
    print("  Step 5: Section adjacency map demo")
    print(THIN_SEP)

    for section in ["IPC-302", "IPC-498A", "Constitution-Article-21", "Evidence-Section-27"]:
        adjacent = get_adjacent_sections(section)
        print(f"  {section} -> {adjacent}")

    # -- Step 6: Tree serialization round-trip --------------------------
    print(f"\n{THIN_SEP}")
    print("  Step 6: Tree serialization round-trip")
    print(THIN_SEP)

    tree_json = tree.to_json()
    tree_str = json.dumps(tree_json, indent=2, default=str)
    tree_size = len(tree_str)

    # Round-trip
    tree_loaded = PerturbationTree.from_json(tree_json)

    print(f"  Serialized tree: {tree_size:,} bytes, {tree.node_count} nodes")
    print(f"  Deserialized tree: {tree_loaded.node_count} nodes")
    print(f"  Round-trip match: {tree.node_count == tree_loaded.node_count}")

    # Verify root fact sheet round-trips
    orig_fs = tree.get_node(0).fact_sheet
    loaded_fs = tree_loaded.get_node(0).fact_sheet
    match = orig_fs.model_dump() == loaded_fs.model_dump()
    print(f"  Root fact sheet match: {match}")

    # Save tree to file
    output_dir = Path("countercase/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase4_tree.json"
    output_path.write_text(tree_str, encoding="utf-8")
    print(f"  Saved tree to: {output_path}")

    # -- Summary --------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("  PHASE 4 SUMMARY")
    print(SEPARATOR)
    print(f"  Tree nodes:          {tree.node_count}")
    print(f"  Root children:       {len(child_ids)}")
    print(f"  Validator accepted:  {validator.accept_count}")
    print(f"  Validator rejected:  {validator.reject_count}")
    print(f"  Expansion time:      {t_expand:.2f}s")
    print(f"  Tree serialized OK:  {tree.node_count == tree_loaded.node_count}")
    print()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: Single-level perturbation tree demo",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Load fact sheet from the fact store by case_id",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Path to a text file to extract a fact sheet from",
    )
    parser.add_argument(
        "--max-children",
        type=int,
        default=5,
        help="Maximum child perturbations per node (default: 5)",
    )

    args = parser.parse_args()
    run_pipeline(
        case_id=args.case_id,
        text_file=args.text,
        max_children=args.max_children,
    )


if __name__ == "__main__":
    main()
