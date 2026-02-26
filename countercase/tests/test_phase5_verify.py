"""Phase 5 verification script."""
from dataclasses import dataclass
import json

from countercase.fact_extraction.schema import (
    EvidenceItem, FactSheet, NumericalFacts, PartyInfo,
)
from countercase.counterfactual.perturbation_tree import PerturbationTree
from countercase.counterfactual.llm_validator import (
    PerturbationValidator, mock_validation_llm_fn,
)
from countercase.counterfactual.sensitivity import (
    compute_diff, compute_sensitivity_scores, compute_per_case_sensitivity,
)


@dataclass
class MockResult:
    case_id: str


def main() -> None:
    tree = PerturbationTree(retriever=None, top_k=10)

    fs = FactSheet(
        case_id="test",
        parties=PartyInfo(petitioner_type="Individual", respondent_type="State"),
        evidence_items=[EvidenceItem(evidence_type="FIR", description="FIR")],
        sections_cited=["IPC-302", "IPC-34"],
        numerical_facts=NumericalFacts(
            amounts=[{"value": 500000.0, "unit": "rupees", "context": "amount"}],
            ages=[{"value": 25, "descriptor": "accused"}],
        ),
    )

    root_id = tree.build_root(fs)
    tree.get_node(root_id).retrieval_results = [
        MockResult("A"), MockResult("B"), MockResult("C"),
        MockResult("D"), MockResult("E"),
    ]

    validator = PerturbationValidator(llm_fn=mock_validation_llm_fn)
    children = tree.expand_node(root_id, validator, max_children=3)

    # Inject different results for children
    tree.get_node(children[0]).retrieval_results = [
        MockResult("A"), MockResult("F"), MockResult("C"),
        MockResult("G"), MockResult("E"),
    ]
    tree.get_node(children[1]).retrieval_results = [
        MockResult("B"), MockResult("A"), MockResult("C"),
        MockResult("D"), MockResult("E"),
    ]
    tree.get_node(children[2]).retrieval_results = [
        MockResult("A"), MockResult("B"), MockResult("C"),
        MockResult("D"), MockResult("E"),
    ]

    # Test sensitivity scores
    scores = compute_sensitivity_scores(tree, k=5)
    print("Sensitivity scores:", scores)
    assert scores["Numerical"] > 0, "Expected non-zero Numerical sensitivity"
    print("  PASS: non-zero sensitivity for Numerical")

    # Verify diffs
    diff0 = compute_diff(
        tree.get_node(root_id).retrieval_results,
        tree.get_node(children[0]).retrieval_results,
        k=5,
    )
    print(
        f"Child {children[0]} diff: dropped={diff0.dropped_cases}, "
        f"new={diff0.new_cases}, mean_disp={diff0.mean_displacement:.2f}"
    )
    assert diff0.dropped_cases == ["B", "D"]
    assert diff0.new_cases == ["F", "G"]
    print("  PASS: diff dropped/new correct")

    # Test per-case sensitivity
    cas = compute_per_case_sensitivity(tree, "A", k=5)
    appearances = cas["appearances"]
    n_disps = len(cas["displacements"])
    print(f"Per-case sensitivity for A: appearances={appearances}, displacements={n_disps}")
    assert appearances >= 3
    print("  PASS: per-case sensitivity")

    # Test manual node editing
    fs2 = FactSheet(
        case_id="test",
        parties=PartyInfo(petitioner_type="Minor", respondent_type="State"),
        evidence_items=fs.evidence_items,
        sections_cited=fs.sections_cited,
        numerical_facts=fs.numerical_facts,
    )
    manual_id = tree.add_manual_node(root_id, fs2, "Changed pet to Minor")
    manual_node = tree.get_node(manual_id)
    assert manual_node.parent_id == root_id
    assert manual_node.edge.description == "Changed pet to Minor"
    assert manual_id in tree.get_node(root_id).children_ids
    print(f"  PASS: manual node {manual_id} created correctly")

    # Test get_depth
    assert tree.get_depth(root_id) == 0
    assert tree.get_depth(children[0]) == 1
    print("  PASS: get_depth")

    # Test get_all_edges
    edges = tree.get_all_edges()
    assert len(edges) >= 4
    print(f"  PASS: get_all_edges returned {len(edges)} edges")

    # Round-trip
    tree_json = tree.to_json()
    tree_str = json.dumps(tree_json, default=str)
    tree2 = PerturbationTree.from_json(json.loads(tree_str))
    assert tree2.node_count == tree.node_count
    print("  PASS: round-trip serialization")

    print(f"\nALL PHASE 5 VERIFICATIONS PASSED")
    print(f"  Tree nodes: {tree.node_count}")
    print(f"  Sensitivity scores: {scores}")
    print(f"  Manual node: {manual_id}")


if __name__ == "__main__":
    main()
