"""COUNTERCASE -- Streamlit application.

Six-page interface for counterfactual legal case retrieval analysis.

Pages:
    1. Query Input -- enter case text or case_id, extract fact sheet.
    2. Retrieval Results -- root node top-K results table.
    3. Perturbation Tree -- tree visualisation with node details.
    4. Diff View -- side-by-side comparison of parent/child results.
    5. Sensitivity Dashboard -- bar chart of fact dimension scores.
    6. Manual Edit -- edit a node's fact sheet and re-retrieve.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so that `countercase` is importable
# when Streamlit runs this script from its own process.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="COUNTERCASE",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports (avoid heavy loads on every rerun)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _load_retriever() -> Any:
    """Load the HybridRetriever once and cache across reruns."""
    try:
        from countercase.retrieval.hybrid_retriever import HybridRetriever
        return HybridRetriever()
    except Exception as exc:
        logger.warning("Could not load retriever: %s", exc)
        return None


def _get_retriever() -> Any:
    return _load_retriever()


def _get_validator() -> Any:
    from countercase.counterfactual.llm_validator import (
        PerturbationValidator,
        mock_validation_llm_fn,
    )
    return PerturbationValidator(llm_fn=mock_validation_llm_fn)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "fact_sheet": None,
    "tree": None,
    "tree_json": None,
    "selected_node_id": 0,
    "diff_parent_id": 0,
    "diff_child_id": None,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = [
    "Query Input",
    "Retrieval Results",
    "Perturbation Tree",
    "Diff View",
    "Sensitivity Dashboard",
    "Manual Edit",
]

page = st.sidebar.radio("Navigation", PAGES)

st.sidebar.markdown("---")
st.sidebar.markdown("**COUNTERCASE**")
st.sidebar.caption("Counterfactual Legal Case Retrieval")

if st.session_state.tree is not None:
    st.sidebar.info(
        f"Tree: {st.session_state.tree.node_count} nodes"
    )


# ===================================================================
# Page 1 -- Query Input
# ===================================================================

def _page_query_input() -> None:
    st.header("Query Input")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Option A: Paste case text")
        case_text = st.text_area(
            "Case text",
            height=250,
            placeholder="Paste the facts section of a judgment here...",
        )

    with col_right:
        st.subheader("Option B: Case ID lookup")
        case_id_input = st.text_input(
            "Case ID",
            placeholder="e.g. Criminal Appeal 1031/2024",
        )

    st.markdown("---")

    st.subheader("Metadata filters (optional)")
    yr_col1, yr_col2 = st.columns(2)
    with yr_col1:
        start_year = st.slider("Start year", 1950, 2026, 2020)
    with yr_col2:
        end_year = st.slider("End year", 1950, 2026, 2025)

    max_depth = st.slider("Tree depth", 1, 5, 2)
    max_children = st.slider("Max children per node", 1, 10, 5)
    min_disp = st.number_input(
        "Min displacement threshold for expansion",
        min_value=0.0,
        value=1.0,
        step=0.5,
    )

    st.markdown("---")

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("Extract Fact Sheet"):
            _extract_fact_sheet(case_text, case_id_input)

    with btn_col2:
        if st.button("Build Perturbation Tree"):
            _build_tree(max_depth, max_children, min_disp)

    # Display current fact sheet if available
    if st.session_state.fact_sheet is not None:
        st.subheader("Current Fact Sheet")
        fs = st.session_state.fact_sheet
        edited_json = st.text_area(
            "Fact sheet JSON (editable)",
            value=fs.model_dump_json(indent=2),
            height=300,
        )
        if st.button("Update Fact Sheet from JSON"):
            try:
                from countercase.fact_extraction.schema import FactSheet
                updated = FactSheet.model_validate_json(edited_json)
                st.session_state.fact_sheet = updated
                st.success("Fact sheet updated.")
            except Exception as exc:
                st.error(f"Invalid JSON: {exc}")


def _extract_fact_sheet(case_text: str, case_id_input: str) -> None:
    """Run fact sheet extraction from text or load from store."""
    from countercase.fact_extraction.schema import FactSheet

    with st.spinner("Extracting fact sheet..."):
        # Try case_id lookup first
        if case_id_input.strip():
            from countercase.fact_extraction.fact_store import load_fact_sheet
            fs = load_fact_sheet(case_id_input.strip())
            if fs is not None:
                st.session_state.fact_sheet = fs
                st.success(f"Loaded fact sheet for: {case_id_input}")
                return
            st.warning(f"No fact sheet found for '{case_id_input}' in store.")

        # Try extracting from text
        if case_text.strip():
            try:
                from countercase.fact_extraction.fact_sheet_extractor import (
                    FactSheetExtractor,
                    mock_llm_fn,
                )
                extractor = FactSheetExtractor(llm_fn=mock_llm_fn)
                fs = extractor.extract(case_text.strip(), case_id="user_input")
                if fs is not None:
                    st.session_state.fact_sheet = fs
                    st.success("Fact sheet extracted from text.")
                    return
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")

        # Fallback: build a sample
        if not case_text.strip() and not case_id_input.strip():
            from countercase.fact_extraction.schema import (
                EvidenceItem,
                NumericalFacts,
                PartyInfo,
            )
            fs = FactSheet(
                case_id="Sample Case 1/2024",
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
                    durations=[{"value": 3.0, "unit": "years", "context": "imprisonment"}],
                ),
                outcome="appeal dismissed",
            )
            st.session_state.fact_sheet = fs
            st.info("Using sample fact sheet (no input provided).")


def _build_tree(
    max_depth: int,
    max_children: int,
    min_disp: float,
) -> None:
    """Build the perturbation tree from the current fact sheet."""
    if st.session_state.fact_sheet is None:
        st.error("Extract a fact sheet first.")
        return

    from countercase.counterfactual.perturbation_tree import PerturbationTree

    retriever = _get_retriever()
    validator = _get_validator()
    fs = st.session_state.fact_sheet

    with st.spinner(f"Building perturbation tree (depth {max_depth})..."):
        tree = PerturbationTree(retriever=retriever, top_k=10)
        tree.build_root(fs)
        tree.expand_tree(
            validator=validator,
            max_depth=max_depth,
            max_children_per_node=max_children,
            min_displacement_threshold=min_disp,
        )

    st.session_state.tree = tree
    st.session_state.tree_json = tree.to_json()
    st.success(f"Tree built: {tree.node_count} nodes.")


# ===================================================================
# Page 2 -- Retrieval Results
# ===================================================================

def _page_retrieval_results() -> None:
    st.header("Retrieval Results")

    tree = st.session_state.tree
    if tree is None:
        st.warning("Build a perturbation tree first (Page 1).")
        return

    root = tree.get_node(tree.root_id)
    results = root.retrieval_results or []

    if not results:
        st.info("No retrieval results at root node (retriever may be unavailable).")
        return

    st.subheader(f"Root node -- top {len(results)} results")

    rows = []
    for idx, r in enumerate(results):
        row = _result_to_row(r, idx + 1)
        rows.append(row)

    # Display as a table
    import pandas as pd
    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
    )

    # Expandable full text
    st.subheader("Result details")
    for idx, r in enumerate(results):
        text = _get_result_field(r, "text", "")
        case_id = _get_result_field(r, "case_id", f"Result {idx + 1}")
        with st.expander(f"Rank {idx + 1}: {case_id}"):
            st.text(text[:2000] if text else "(no text)")


# ===================================================================
# Page 3 -- Perturbation Tree
# ===================================================================

def _page_perturbation_tree() -> None:
    st.header("Perturbation Tree")

    tree = st.session_state.tree
    if tree is None:
        st.warning("Build a perturbation tree first (Page 1).")
        return

    st.subheader("Tree structure")
    _render_tree_view(tree, tree.root_id, indent=0)

    st.markdown("---")

    # Node detail selector
    node_ids = sorted(tree._nodes.keys())
    selected = st.selectbox(
        "Select a node to inspect",
        node_ids,
        format_func=lambda nid: _node_label(tree, nid),
    )

    if selected is not None:
        st.session_state.selected_node_id = selected
        node = tree.get_node(selected)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Node {selected}")
            if node.edge:
                st.markdown(f"**Edge:** {node.edge.description}")
                st.markdown(f"**Fact type:** {node.edge.fact_type.value}")
            else:
                st.markdown("**Root node**")
            st.markdown(
                f"**Children:** {len(node.children_ids)} | "
                f"**Results:** {len(node.retrieval_results) if node.retrieval_results else 0}"
            )

        with col2:
            st.subheader("Fact sheet")
            st.json(json.loads(node.fact_sheet.model_dump_json()))

        # Show retrieval results for selected node
        results = node.retrieval_results or []
        if results:
            st.subheader(f"Retrieval results for node {selected}")
            import pandas as pd
            rows = [_result_to_row(r, i + 1) for i, r in enumerate(results)]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_tree_view(tree: Any, node_id: int, indent: int) -> None:
    """Recursively render the tree as indented markdown."""
    node = tree.get_node(node_id)
    prefix = "&nbsp;" * (indent * 4)
    n_results = len(node.retrieval_results) if node.retrieval_results else 0

    if node.edge:
        label = f"{prefix}**[{node_id}]** {node.edge.description} ({n_results} results)"
    else:
        label = f"{prefix}**[{node_id}] Root** ({n_results} results)"

    st.markdown(label, unsafe_allow_html=True)

    for child_id in node.children_ids:
        _render_tree_view(tree, child_id, indent + 1)


# ===================================================================
# Page 4 -- Diff View
# ===================================================================

def _page_diff_view() -> None:
    st.header("Diff View")

    tree = st.session_state.tree
    if tree is None:
        st.warning("Build a perturbation tree first (Page 1).")
        return

    node_ids = sorted(tree._nodes.keys())

    col1, col2 = st.columns(2)
    with col1:
        parent_id = st.selectbox(
            "Parent node",
            node_ids,
            format_func=lambda nid: _node_label(tree, nid),
            key="diff_parent",
        )
    with col2:
        child_candidates = tree.get_node(parent_id).children_ids if parent_id is not None else []
        if not child_candidates:
            child_candidates = [n for n in node_ids if n != parent_id]
        child_id = st.selectbox(
            "Child node",
            child_candidates,
            format_func=lambda nid: _node_label(tree, nid),
            key="diff_child",
        )

    if parent_id is not None and child_id is not None:
        parent_node = tree.get_node(parent_id)
        child_node = tree.get_node(child_id)

        # Edge description
        if child_node.edge:
            st.markdown(f"### {child_node.edge.description}")
        st.markdown(f"Comparing **Node {parent_id}** (parent) vs **Node {child_id}** (child)")

        from countercase.counterfactual.sensitivity import compute_diff
        diff = compute_diff(
            parent_node.retrieval_results or [],
            child_node.retrieval_results or [],
            k=10,
        )

        # Summary metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Dropped", len(diff.dropped_cases))
        mc2.metric("New", len(diff.new_cases))
        mc3.metric("Stable", len(diff.stable_cases))
        mc4.metric("Mean displacement", f"{diff.mean_displacement:.2f}")

        # Build diff table
        parent_results = parent_node.retrieval_results or []
        child_results = child_node.retrieval_results or []
        parent_ids = _extract_result_ids(parent_results, 10)
        child_ids = _extract_result_ids(child_results, 10)
        p_rank_map = {cid: i + 1 for i, cid in enumerate(parent_ids)}
        c_rank_map = {cid: i + 1 for i, cid in enumerate(child_ids)}

        rows = []
        all_ids = list(dict.fromkeys(parent_ids + child_ids))
        for cid in all_ids:
            p_rank = p_rank_map.get(cid, "-")
            c_rank = c_rank_map.get(cid, "-")
            if cid in diff.dropped_cases:
                status = "DROPPED"
            elif cid in diff.new_cases:
                status = "NEW"
            else:
                status = "STABLE"
            disp = diff.rank_displacements.get(cid, 0.0)
            rows.append({
                "Case ID": cid,
                "Parent rank": p_rank,
                "Child rank": c_rank,
                "Status": status,
                "Displacement": disp,
            })

        import pandas as pd
        df = pd.DataFrame(rows)

        def _color_status(row: Any) -> list[str]:
            if row["Status"] == "DROPPED":
                return ["background-color: #ffcccc"] * len(row)
            elif row["Status"] == "NEW":
                return ["background-color: #ccffcc"] * len(row)
            return ["background-color: #e0e0e0"] * len(row)

        styled = df.style.apply(_color_status, axis=1)
        st.dataframe(styled, width="stretch", hide_index=True)


# ===================================================================
# Page 5 -- Sensitivity Dashboard
# ===================================================================

def _page_sensitivity_dashboard() -> None:
    st.header("Sensitivity Dashboard")

    tree = st.session_state.tree
    if tree is None:
        st.warning("Build a perturbation tree first (Page 1).")
        return

    from countercase.counterfactual.sensitivity import compute_sensitivity_scores

    scores = compute_sensitivity_scores(tree, k=10)

    if not any(v > 0.0 for v in scores.values()):
        st.info(
            "All sensitivity scores are 0.0. This is expected when "
            "the retriever is unavailable (no retrieval results to compare)."
        )

    # Bar chart
    st.subheader("Aggregate sensitivity by fact dimension")
    try:
        import plotly.graph_objects as go

        fact_types = sorted(scores.keys())
        values = [scores[ft] for ft in fact_types]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=fact_types,
                    y=values,
                    marker_color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"],
                )
            ],
        )
        fig.update_layout(
            xaxis_title="Fact dimension",
            yaxis_title="Mean rank displacement",
            title="Sensitivity scores per fact dimension",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")
    except ImportError:
        import pandas as pd
        st.bar_chart(pd.Series(scores))

    # Table
    st.subheader("Scores table")
    import pandas as pd
    score_df = pd.DataFrame(
        [{"Fact type": k, "Sensitivity score": v} for k, v in sorted(
            scores.items(), key=lambda x: x[1], reverse=True,
        )]
    )
    st.dataframe(score_df, width="stretch", hide_index=True)

    # Interpretation
    if scores:
        top_fact = max(scores, key=scores.get)
        top_score = scores[top_fact]
        st.markdown(
            f"**Interpretation:** The most legally operative fact for this "
            f"case is **{top_fact}** with a sensitivity score of "
            f"**{top_score:.2f}**."
        )


# ===================================================================
# Page 6 -- Manual Edit
# ===================================================================

def _page_manual_edit() -> None:
    st.header("Manual Edit")

    tree = st.session_state.tree
    if tree is None:
        st.warning("Build a perturbation tree first (Page 1).")
        return

    node_ids = sorted(tree._nodes.keys())
    selected_id = st.selectbox(
        "Select a node to edit",
        node_ids,
        format_func=lambda nid: _node_label(tree, nid),
        key="manual_edit_node",
    )

    if selected_id is not None:
        node = tree.get_node(selected_id)
        fs = node.fact_sheet

        st.subheader(f"Editing fact sheet at Node {selected_id}")

        # Editable fields
        col1, col2 = st.columns(2)
        with col1:
            petitioner_type = st.text_input(
                "Petitioner type",
                value=fs.parties.petitioner_type or "",
                key="edit_pet",
            )
            respondent_type = st.text_input(
                "Respondent type",
                value=fs.parties.respondent_type or "",
                key="edit_resp",
            )

        with col2:
            sections_str = st.text_input(
                "Sections cited (comma-separated)",
                value=", ".join(fs.sections_cited),
                key="edit_sections",
            )
            outcome = st.text_input(
                "Outcome",
                value=fs.outcome or "",
                key="edit_outcome",
            )

        # Evidence items
        st.subheader("Evidence items")
        evidence_json = st.text_area(
            "Evidence items JSON",
            value=json.dumps(
                [e.model_dump() for e in fs.evidence_items], indent=2,
            ),
            height=150,
            key="edit_evidence",
        )

        # Numerical facts
        st.subheader("Numerical facts")
        numerical_json = st.text_area(
            "Numerical facts JSON",
            value=fs.numerical_facts.model_dump_json(indent=2),
            height=150,
            key="edit_numerical",
        )

        # Description
        description = st.text_input(
            "Description of this edit",
            value="Manual edit",
            key="edit_description",
        )

        if st.button("Apply Edit and Re-retrieve"):
            _apply_manual_edit(
                tree,
                selected_id,
                petitioner_type,
                respondent_type,
                sections_str,
                outcome,
                evidence_json,
                numerical_json,
                description,
            )


def _apply_manual_edit(
    tree: Any,
    parent_id: int,
    petitioner_type: str,
    respondent_type: str,
    sections_str: str,
    outcome: str,
    evidence_json: str,
    numerical_json: str,
    description: str,
) -> None:
    """Parse edited fields, create manual node, refresh tree state."""
    from countercase.fact_extraction.schema import (
        EvidenceItem,
        FactSheet,
        NumericalFacts,
        PartyInfo,
    )

    try:
        sections = [
            s.strip() for s in sections_str.split(",") if s.strip()
        ]
        evidence_items = [
            EvidenceItem.model_validate(e)
            for e in json.loads(evidence_json)
        ]
        numerical_facts = NumericalFacts.model_validate_json(numerical_json)

        parent_fs = tree.get_node(parent_id).fact_sheet
        edited_fs = FactSheet(
            case_id=parent_fs.case_id,
            parties=PartyInfo(
                petitioner_type=petitioner_type or None,
                respondent_type=respondent_type or None,
            ),
            evidence_items=evidence_items,
            sections_cited=sections,
            numerical_facts=numerical_facts,
            outcome=outcome or None,
        )

        new_id = tree.add_manual_node(
            parent_id=parent_id,
            edited_fact_sheet=edited_fs,
            description=description,
        )

        st.session_state.tree_json = tree.to_json()
        st.success(f"Created manual node {new_id} under node {parent_id}.")

    except Exception as exc:
        st.error(f"Failed to apply edit: {exc}")


# ===================================================================
# Helpers
# ===================================================================

def _result_to_row(r: Any, rank: int) -> dict[str, Any]:
    """Convert a retrieval result to a display dict."""
    return {
        "Rank": rank,
        "Case ID": _get_result_field(r, "case_id", ""),
        "Section type": _get_result_field(r, "section_type", ""),
        "Source PDF": _get_result_field(r, "source_pdf", ""),
        "Page": _get_result_field(r, "page_number", 0),
        "RRF score": round(_get_result_field(r, "rrf_score", 0.0), 4),
        "Snippet": str(_get_result_field(r, "text", ""))[:200],
    }


def _get_result_field(r: Any, field: str, default: Any = None) -> Any:
    """Get a field from a result object or dict."""
    if hasattr(r, field):
        return getattr(r, field)
    if isinstance(r, dict):
        return r.get(field, default)
    return default


def _extract_result_ids(results: list[Any], k: int) -> list[str]:
    """Extract case IDs from results list."""
    ids: list[str] = []
    seen: set[str] = set()
    for r in results[:k]:
        cid = _get_result_field(r, "case_id", "")
        if cid and cid not in seen:
            ids.append(cid)
            seen.add(cid)
    return ids


def _node_label(tree: Any, node_id: int) -> str:
    """Human-readable label for a node."""
    node = tree.get_node(node_id)
    if node.edge:
        return f"Node {node_id}: {node.edge.description}"
    return f"Node {node_id}: Root"


# ===================================================================
# Page dispatch
# ===================================================================

_PAGE_MAP = {
    "Query Input": _page_query_input,
    "Retrieval Results": _page_retrieval_results,
    "Perturbation Tree": _page_perturbation_tree,
    "Diff View": _page_diff_view,
    "Sensitivity Dashboard": _page_sensitivity_dashboard,
    "Manual Edit": _page_manual_edit,
}

_PAGE_MAP[page]()
