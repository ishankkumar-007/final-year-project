# COUNTERCASE — Streamlit App Description

## Overview

COUNTERCASE is a six-page Streamlit application for **counterfactual legal case retrieval analysis**. Given a Supreme Court of India judgment, it extracts structured facts, retrieves similar cases, systematically perturbs each fact dimension, and measures how retrieval results change — revealing which facts are most legally operative.

---

## Pages

### Page 1: Query Input

The starting point. Two ways to provide a case:

- **Option A — Paste case text:** Copy/paste the facts section of a judgment into the text area. Click **"Extract Fact Sheet"** to send the text to the Mistral LLM API, which returns a structured fact sheet (parties, evidence, legal sections, numerical facts, outcome).
- **Option B — Case ID lookup:** Enter a case ID that already exists in the fact store (from prior Phase 3 pipeline runs). Click **"Extract Fact Sheet"** to load the pre-extracted fact sheet from disk.
- If both fields are left blank, a **sample fact sheet** is loaded for demonstration purposes.

#### Configuration

- **Year range sliders** — filter retrieval to cases from a specific period.
- **Tree depth** — how many levels of perturbation to explore (1–5).
- **Max children per node** — how many perturbation variants to generate per fact dimension.
- **Min displacement threshold** — minimum retrieval change required for a perturbation to be considered significant.

#### Editing

Once a fact sheet is loaded, it appears as an editable JSON text area. Modify any field and click **"Update Fact Sheet from JSON"** to apply changes before building the tree.

#### Building the Tree

Click **"Build Perturbation Tree"** to:

1. Retrieve the top-K similar cases for the root (original) fact sheet.
2. Systematically perturb each fact dimension (parties, sections, evidence, numerical values).
3. Re-retrieve for each perturbation, building a tree of counterfactual variants.

---

### Page 2: Retrieval Results

Displays the **root node's top-K retrieved cases** in a sortable table with columns:

| Column       | Description                          |
|--------------|--------------------------------------|
| Rank         | Position in the result list          |
| Case ID      | Identifier of the retrieved case     |
| Section type | Type of matched section              |
| Source PDF   | Original PDF filename                |
| Page         | Page number in the source document   |
| RRF score    | Reciprocal Rank Fusion score         |
| Snippet      | First 200 characters of matched text |

Expandable rows show **per-result explanations** describing *why* each case was retrieved relative to the query fact sheet.

---

### Page 3: Perturbation Tree

Visualises the **full tree structure** as indented labels (root → children → grandchildren), showing how many retrieval results each node has.

Select any node to inspect:

- **Edge description** — what fact was changed (e.g., "Changed petitioner from Individual to Corporation").
- **Fact type** — which dimension was perturbed (Party, Section, Evidence, Numerical).
- **Fact sheet JSON** — the complete fact sheet at that node.
- **Retrieval results table** — the top-K cases retrieved for that node's modified fact sheet.

---

### Page 4: Diff View

**Side-by-side comparison** of any parent–child node pair. Select a parent and child node to see:

#### Summary Metrics

| Metric            | Description                                    |
|-------------------|------------------------------------------------|
| Dropped           | Cases in parent results but not in child        |
| New               | Cases in child results but not in parent        |
| Stable            | Cases present in both result sets               |
| Mean displacement | Average rank change across stable cases         |

#### Diff Table

Colour-coded rows:
- **Red** — case was dropped from results after the perturbation.
- **Green** — case is new (appeared only after the perturbation).
- **Grey** — case is stable (present in both, possibly re-ranked).

#### Counterfactual Summary

A natural language explanation of *why* changing that particular fact caused the observed retrieval shift.

#### Export

Download buttons for the full analysis as **JSON** or a **Markdown report**.

---

### Page 5: Sensitivity Dashboard

Aggregates perturbation results into a **sensitivity score per fact dimension**.

- **Bar chart** (via Plotly) showing mean rank displacement for each fact type.
- **Scores table** sorted by sensitivity (highest first).
- **Interpretation** — e.g., *"The most legally operative fact is Sections with a sensitivity score of 2.34."*

Higher sensitivity = changing that fact type causes the most disruption to retrieved results, indicating it is the most legally operative dimension for the query case.

---

### Page 6: Manual Edit

Allows **free-form "what if" testing** on any node in the tree.

Editable fields:
- Petitioner type / Respondent type
- Sections cited (comma-separated)
- Outcome
- Evidence items (JSON)
- Numerical facts (JSON)
- Description of the edit

Click **"Apply Edit and Re-retrieve"** to create a new child node with the manual changes and run retrieval, letting you see the impact of arbitrary hypothetical modifications.

---

## Typical Workflow

```
Page 1 (Input + Build Tree)
    │
    ▼
Page 2 (Review base retrieval results)
    │
    ▼
Page 3 (Explore perturbation tree)
    │
    ▼
Page 4 (Compare specific parent/child pairs)
    │
    ▼
Page 5 (See which facts matter most)
    │
    ▼
Page 6 (Test custom hypotheses)
    │
    ▼
Export (Download JSON / Markdown report)
```

## Running the App

```bash
conda activate torch2
streamlit run countercase/app/streamlit_app.py
```

The app opens at `http://localhost:8501`.

### Required Environment Variables

Set these in a `.env` file at the project root:

| Variable         | Description                              | Default                        |
|------------------|------------------------------------------|--------------------------------|
| `LLM_API_KEY`    | Mistral API key                          | *(required for real extraction)*|
| `LLM_API_URL`    | API base URL                             | `https://api.mistral.ai/v1`   |
| `LLM_API_MODEL`  | Model name                               | `open-mistral-7b`             |
| `HF_TOKEN`       | Hugging Face token (for DPR embeddings)  | *(optional)*                   |
