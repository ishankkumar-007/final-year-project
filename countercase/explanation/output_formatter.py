"""Output formatter: JSON and Markdown report generation.

Provides structured JSON output for the full perturbation tree and a
deterministic Markdown text summary derived entirely from the JSON.
No LLM calls -- the text is a strict projection of the data.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from countercase.counterfactual.sensitivity import DiffResult, compute_diff
from countercase.explanation.counterfactual_summary import explain_edge
from countercase.explanation.per_result import explain_result

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Field accessor
# -------------------------------------------------------------------

def _field(obj: Any, name: str, default: Any = "") -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


# -------------------------------------------------------------------
# Per-node JSON
# -------------------------------------------------------------------

def format_node_output(
    node: Any,
    diff: DiffResult | None,
    explanations: dict[str, str],
    sensitivity_scores: dict[str, float] | None = None,
    counterfactual_summary: str | None = None,
) -> dict[str, Any]:
    """Format a single tree node into the canonical JSON schema.

    Args:
        node: A TreeNode with fact_sheet, retrieval_results, edge.
        diff: DiffResult vs parent (None for root).
        explanations: Mapping of chunk_id -> per-result explanation.
        sensitivity_scores: Aggregate scores (typically from root).
        counterfactual_summary: Edge explanation text.

    Returns:
        Dict conforming to the plan.md section 8 JSON schema.
    """
    # Fact sheet state
    fs = node.fact_sheet
    fs_dict = json.loads(fs.model_dump_json()) if hasattr(fs, "model_dump_json") else {}

    # Retrieval results
    results_out: list[dict[str, Any]] = []
    for rank_idx, r in enumerate(node.retrieval_results or []):
        chunk_id = str(_field(r, "chunk_id", ""))
        results_out.append({
            "chunk_id": chunk_id,
            "source_pdf": str(_field(r, "source_pdf", "")),
            "page_number": int(_field(r, "page_number", 0)),
            "section_type": str(_field(r, "section_type", "Unknown")),
            "rank": rank_idx + 1,
            "rrf_score": float(_field(r, "rrf_score", 0.0)),
            "reranker_score": float(_field(r, "reranker_score", 0.0)),
            "case_id": str(_field(r, "case_id", "")),
            "year": int(_field(r, "year", 0)),
            "explanation": explanations.get(chunk_id, ""),
        })

    # Diff vs parent
    diff_dict: dict[str, Any] | None = None
    if diff is not None:
        diff_dict = diff.to_dict()
        # Rename for consistency with spec
        diff_dict["newly_appeared_cases"] = diff_dict.pop("new_cases", [])

    output: dict[str, Any] = {
        "node_id": node.node_id,
        "fact_sheet_state": fs_dict,
        "retrieval_results": results_out,
        "diff_vs_parent": diff_dict,
        "sensitivity_scores": sensitivity_scores,
        "counterfactual_summary": counterfactual_summary or "",
    }
    return output


# -------------------------------------------------------------------
# Full tree JSON
# -------------------------------------------------------------------

def format_tree_output(
    tree: Any,
    sensitivity_scores: dict[str, float] | None = None,
    k: int = 10,
) -> dict[str, Any]:
    """Format the entire perturbation tree as a nested JSON structure.

    Computes diffs and explanations for every node in the tree.

    Args:
        tree: A populated PerturbationTree.
        sensitivity_scores: Pre-computed aggregate scores (optional).
        k: Top-K for diff computation.

    Returns:
        Dict with ``root``, ``nodes`` (flat list), and ``metadata``.
    """
    root_id = tree.root_id
    root_node = tree.get_node(root_id) if root_id is not None else None
    root_fs = root_node.fact_sheet if root_node else None

    nodes_out: list[dict[str, Any]] = []

    for nid in sorted(tree._nodes.keys()):
        node = tree.get_node(nid)

        # Diff
        diff: DiffResult | None = None
        cf_summary: str | None = None
        if node.parent_id is not None:
            parent = tree.get_node(node.parent_id)
            diff = compute_diff(
                parent.retrieval_results or [],
                node.retrieval_results or [],
                k=k,
            )
            cf_summary = explain_edge(parent, node, diff)

        # Per-result explanations
        expl: dict[str, str] = {}
        if root_fs is not None and node.retrieval_results:
            for r in node.retrieval_results:
                chunk_id = str(_field(r, "chunk_id", ""))
                if chunk_id:
                    expl[chunk_id] = explain_result(root_fs, r)

        # Sensitivity only on root node
        scores = sensitivity_scores if nid == root_id else None

        node_dict = format_node_output(
            node, diff, expl,
            sensitivity_scores=scores,
            counterfactual_summary=cf_summary,
        )
        nodes_out.append(node_dict)

    return {
        "root_id": root_id,
        "node_count": len(nodes_out),
        "sensitivity_scores": sensitivity_scores,
        "nodes": nodes_out,
    }


# -------------------------------------------------------------------
# Markdown text summary (6.4)
# -------------------------------------------------------------------

def generate_text_summary(tree_output: dict[str, Any]) -> str:
    """Generate a human-readable Markdown report from JSON output.

    The report has four sections:
        1. Query case fact sheet
        2. Top retrieved precedents with explanations
        3. Counterfactual analysis
        4. Conclusion

    This is a pure string-formatting function.  No LLM calls.
    The text is derivable from the JSON alone.

    Args:
        tree_output: Dict produced by :func:`format_tree_output`.

    Returns:
        Markdown-formatted report string.
    """
    lines: list[str] = []
    nodes = tree_output.get("nodes", [])
    root_id = tree_output.get("root_id", 0)
    sensitivity = tree_output.get("sensitivity_scores") or {}

    root_node = None
    for n in nodes:
        if n.get("node_id") == root_id:
            root_node = n
            break

    # ----- Section 1: Fact sheet -----
    lines.append("# COUNTERCASE Analysis Report")
    lines.append("")
    lines.append("## 1. Query Case Fact Sheet")
    lines.append("")

    if root_node:
        fs = root_node.get("fact_sheet_state", {})
        lines.append(f"**Case ID:** {fs.get('case_id', 'N/A')}")
        lines.append("")

        parties = fs.get("parties", {})
        lines.append("| Role | Type |")
        lines.append("|------|------|")
        lines.append(
            f"| Petitioner | {parties.get('petitioner_type', 'N/A')} |"
        )
        lines.append(
            f"| Respondent | {parties.get('respondent_type', 'N/A')} |"
        )
        lines.append("")

        sections = fs.get("sections_cited", [])
        if sections:
            lines.append(
                f"**Sections cited:** {', '.join(sections)}"
            )
            lines.append("")

        outcome = fs.get("outcome")
        if outcome:
            lines.append(f"**Outcome:** {outcome}")
            lines.append("")

        evidence = fs.get("evidence_items", [])
        if evidence:
            lines.append("**Evidence items:**")
            lines.append("")
            for ev in evidence:
                etype = ev.get("evidence_type", "")
                desc = ev.get("description", "")
                lines.append(f"- {etype}: {desc}")
            lines.append("")

        numerical = fs.get("numerical_facts", {})
        _format_numerical(lines, numerical)

    # ----- Section 2: Retrieved precedents -----
    lines.append("## 2. Top Retrieved Precedents")
    lines.append("")

    if root_node:
        results = root_node.get("retrieval_results", [])
        if results:
            lines.append(
                "| Rank | Case ID | Section type | RRF score | Explanation |"
            )
            lines.append(
                "|------|---------|-------------|-----------|-------------|"
            )
            for r in results[:10]:
                lines.append(
                    f"| {r.get('rank', '')} "
                    f"| {r.get('case_id', '')} "
                    f"| {r.get('section_type', '')} "
                    f"| {r.get('rrf_score', 0.0):.4f} "
                    f"| {r.get('explanation', '')} |"
                )
            lines.append("")
        else:
            lines.append("No retrieval results available.")
            lines.append("")

    # ----- Section 3: Counterfactual analysis -----
    lines.append("## 3. Counterfactual Analysis")
    lines.append("")

    child_nodes = [n for n in nodes if n.get("diff_vs_parent") is not None]

    if child_nodes:
        lines.append(
            "| Node | Perturbation | Dropped | New | Stable "
            "| Mean displacement |"
        )
        lines.append(
            "|------|-------------|---------|-----|--------"
            "|-------------------|"
        )
        for cn in child_nodes:
            nid = cn.get("node_id", "")
            desc = cn.get("counterfactual_summary", "")[:80]
            diff_data = cn.get("diff_vs_parent", {})
            n_dropped = len(diff_data.get("dropped_cases", []))
            n_new = len(diff_data.get("newly_appeared_cases", []))
            n_stable = len(diff_data.get("stable_cases", []))
            mean_d = diff_data.get("mean_displacement", 0.0)
            lines.append(
                f"| {nid} | {desc} | {n_dropped} | {n_new} "
                f"| {n_stable} | {mean_d:.2f} |"
            )
        lines.append("")

        # Counterfactual summaries
        for cn in child_nodes:
            nid = cn.get("node_id", "")
            summary = cn.get("counterfactual_summary", "")
            if summary:
                lines.append(f"**Node {nid}:** {summary}")
                lines.append("")

    else:
        lines.append("No perturbation children in the tree.")
        lines.append("")

    # Sensitivity scores
    if sensitivity:
        lines.append("### Sensitivity Scores")
        lines.append("")
        lines.append("| Fact dimension | Score |")
        lines.append("|----------------|-------|")
        for ft in sorted(sensitivity.keys(), key=lambda k: sensitivity[k], reverse=True):
            lines.append(f"| {ft} | {sensitivity[ft]:.2f} |")
        lines.append("")

    # ----- Section 4: Conclusion -----
    lines.append("## 4. Conclusion")
    lines.append("")

    if sensitivity:
        top_fact = max(sensitivity, key=sensitivity.get)  # type: ignore[arg-type]
        top_score = sensitivity[top_fact]
        lines.append(
            f"The most legally operative fact for this case is "
            f"**{top_fact}** with a sensitivity score of **{top_score:.2f}**."
        )
    else:
        lines.append(
            "Sensitivity scores are not available.  Build a deeper "
            "perturbation tree with retrieval results to compute scores."
        )
    lines.append("")

    total_nodes = tree_output.get("node_count", 0)
    lines.append(
        f"Total perturbation tree nodes: {total_nodes}."
    )
    lines.append("")
    lines.append("---")
    lines.append("*Report generated by COUNTERCASE.*")
    lines.append("")

    return "\n".join(lines)


# -------------------------------------------------------------------
# Numerical facts formatting helper
# -------------------------------------------------------------------

def _format_numerical(lines: list[str], numerical: dict[str, Any]) -> None:
    """Append numerical facts to the markdown lines."""
    amounts = numerical.get("amounts", [])
    ages = numerical.get("ages", [])
    durations = numerical.get("durations", [])

    if amounts or ages or durations:
        lines.append("**Numerical facts:**")
        lines.append("")
        for a in amounts:
            lines.append(
                f"- Amount: {a.get('value', '')} {a.get('unit', '')} "
                f"({a.get('context', '')})"
            )
        for a in ages:
            lines.append(
                f"- Age: {a.get('value', '')} ({a.get('descriptor', '')})"
            )
        for d in durations:
            lines.append(
                f"- Duration: {d.get('value', '')} {d.get('unit', '')} "
                f"({d.get('context', '')})"
            )
        lines.append("")
