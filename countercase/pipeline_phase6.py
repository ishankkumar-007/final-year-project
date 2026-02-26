"""Phase 6 pipeline: explanation engine and output format.

End-to-end demonstration:
    1. Get or build a fact sheet for a sample case.
    2. Build the perturbation tree to depth 2.
    3. Generate per-result explanations for every node.
    4. Generate counterfactual summaries for every edge.
    5. Compute sensitivity scores.
    6. Format the tree as structured JSON.
    7. Generate a Markdown text summary.
    8. Write both to countercase/output/.

Usage::

    python -m countercase.pipeline_phase6 [--case-id ID] [--max-depth 2]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s -- %(message)s",
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[0]
OUTPUT_DIR = _ROOT / "output"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="COUNTERCASE Phase 6 -- Explanation Engine",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default="",
        help="Case ID to load from the fact store.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum tree depth (default 2).",
    )
    parser.add_argument(
        "--max-children",
        type=int,
        default=5,
        help="Maximum children per node.",
    )
    parser.add_argument(
        "--min-displacement",
        type=float,
        default=1.0,
        help="Minimum mean displacement threshold for expansion.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()

    # ---------------------------------------------------------------
    # Step 1: Fact sheet
    # ---------------------------------------------------------------
    from countercase.fact_extraction.schema import (
        EvidenceItem,
        FactSheet,
        NumericalFacts,
        PartyInfo,
    )

    fs: FactSheet | None = None

    if args.case_id:
        from countercase.fact_extraction.fact_store import load_fact_sheet
        fs = load_fact_sheet(args.case_id)
        if fs:
            logger.info("Step 1: Loaded fact sheet for %s", args.case_id)

    if fs is None:
        logger.info("Step 1: Building sample fact sheet")
        fs = FactSheet(
            case_id=args.case_id or "Sample Case 1/2024",
            parties=PartyInfo(
                petitioner_type="Individual",
                respondent_type="State",
            ),
            evidence_items=[
                EvidenceItem(evidence_type="FIR", description="FIR filed"),
                EvidenceItem(
                    evidence_type="MedicalReport",
                    description="Post-mortem report",
                ),
            ],
            sections_cited=["IPC-302", "IPC-34", "IPC-304"],
            numerical_facts=NumericalFacts(
                amounts=[
                    {"value": 500000.0, "unit": "rupees", "context": "compensation"},
                ],
                ages=[{"value": 25, "descriptor": "accused"}],
                durations=[
                    {"value": 3.0, "unit": "years", "context": "imprisonment"},
                ],
            ),
            outcome="appeal dismissed",
        )

    # ---------------------------------------------------------------
    # Step 2: Build perturbation tree
    # ---------------------------------------------------------------
    logger.info("Step 2: Building perturbation tree (depth %d)", args.max_depth)
    from countercase.counterfactual.perturbation_tree import PerturbationTree

    retriever = None
    try:
        from countercase.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        logger.info("  Retriever loaded.")
    except Exception as exc:
        logger.warning("  Retriever unavailable: %s (continuing without)", exc)

    from countercase.counterfactual.llm_validator import (
        PerturbationValidator,
        mock_validation_llm_fn,
    )
    validator = PerturbationValidator(llm_fn=mock_validation_llm_fn)

    tree = PerturbationTree(retriever=retriever, top_k=10)
    tree.build_root(fs)
    tree.expand_tree(
        validator=validator,
        max_depth=args.max_depth,
        max_children_per_node=args.max_children,
        min_displacement_threshold=args.min_displacement,
    )
    logger.info("  Tree nodes: %d", tree.node_count)

    # ---------------------------------------------------------------
    # Step 3: Per-result explanations
    # ---------------------------------------------------------------
    logger.info("Step 3: Generating per-result explanations")
    from countercase.explanation.per_result import explain_result

    expl_count = 0
    for nid in sorted(tree._nodes.keys()):
        node = tree.get_node(nid)
        for r in node.retrieval_results or []:
            explain_result(fs, r)
            expl_count += 1
    logger.info("  Generated %d per-result explanations.", expl_count)

    # ---------------------------------------------------------------
    # Step 4: Counterfactual summaries
    # ---------------------------------------------------------------
    logger.info("Step 4: Generating counterfactual summaries")
    from countercase.counterfactual.sensitivity import compute_diff
    from countercase.explanation.counterfactual_summary import explain_edge

    cf_count = 0
    for nid in sorted(tree._nodes.keys()):
        node = tree.get_node(nid)
        if node.parent_id is None:
            continue
        parent = tree.get_node(node.parent_id)
        diff = compute_diff(
            parent.retrieval_results or [],
            node.retrieval_results or [],
            k=10,
        )
        summary = explain_edge(parent, node, diff)
        if summary:
            cf_count += 1
            logger.info("  Node %d: %s", nid, summary[:120])
    logger.info("  Generated %d counterfactual summaries.", cf_count)

    # ---------------------------------------------------------------
    # Step 5: Sensitivity scores
    # ---------------------------------------------------------------
    logger.info("Step 5: Computing sensitivity scores")
    from countercase.counterfactual.sensitivity import compute_sensitivity_scores

    scores = compute_sensitivity_scores(tree, k=10)
    for ft, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        logger.info("  %s: %.2f", ft, sc)

    # ---------------------------------------------------------------
    # Step 6: JSON output
    # ---------------------------------------------------------------
    logger.info("Step 6: Formatting JSON output")
    from countercase.explanation.output_formatter import format_tree_output

    tree_json = format_tree_output(tree, sensitivity_scores=scores, k=10)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "phase6_tree.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(tree_json, fh, indent=2, ensure_ascii=False)
    logger.info("  JSON written to %s", json_path)

    # ---------------------------------------------------------------
    # Step 7: Markdown summary
    # ---------------------------------------------------------------
    logger.info("Step 7: Generating Markdown summary")
    from countercase.explanation.output_formatter import generate_text_summary

    md_text = generate_text_summary(tree_json)
    md_path = OUTPUT_DIR / "phase6_report.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(md_text)
    logger.info("  Markdown written to %s", md_path)

    # ---------------------------------------------------------------
    # Step 8: Summary
    # ---------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("Phase 6 pipeline complete in %.1f seconds.", elapsed)
    logger.info("  Tree nodes: %d", tree.node_count)
    logger.info("  JSON file: %s", json_path)
    logger.info("  Markdown file: %s", md_path)
    logger.info("=" * 60)

    # Print a short sample of the Markdown
    print()
    print("=== Markdown Preview (first 40 lines) ===")
    for line in md_text.split("\n")[:40]:
        print(line)
    print()

    print("Launch the Streamlit app:")
    print("  streamlit run countercase/app/streamlit_app.py")
    print()


if __name__ == "__main__":
    main()
