"""Per-result explanation generator.

Generates one-to-two sentence explanations for each retrieval result
by grounding in shared metadata, TF-IDF term overlap, and party type
matching.  No LLM calls -- all explanations are deterministic and
traceable to source data.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

from countercase.fact_extraction.schema import FactSheet

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Lightweight TF-IDF helper
# -------------------------------------------------------------------

_STOP_WORDS: set[str] = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "on",
    "is", "was", "were", "are", "be", "been", "being", "with", "at",
    "by", "from", "as", "that", "this", "it", "its", "not", "but",
    "which", "who", "whom", "their", "they", "he", "she", "his",
    "her", "had", "has", "have", "do", "does", "did", "will", "shall",
    "would", "could", "should", "may", "might", "can", "than", "no",
    "nor", "so", "if", "then", "into", "upon", "about", "also", "such",
    "any", "all", "each", "both", "more", "most", "other", "some",
}

_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    """Lowercase alpha-numeric tokenisation, filtering stop words."""
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOP_WORDS and len(t) > 2
    ]


class _TfIdfTermScorer:
    """Lightweight IDF scorer built over a small corpus.

    Not meant to replace sklearn -- just used to identify high-IDF
    overlap terms between two short documents without heavy deps.
    """

    def __init__(self) -> None:
        self._doc_freq: Counter[str] = Counter()
        self._n_docs: int = 0

    def fit(self, documents: list[str]) -> None:
        """Compute document frequencies from a list of texts."""
        self._n_docs = len(documents)
        for doc in documents:
            unique_tokens = set(_tokenize(doc))
            for t in unique_tokens:
                self._doc_freq[t] += 1

    def idf(self, term: str) -> float:
        """Inverse document frequency with add-one smoothing."""
        df = self._doc_freq.get(term, 0)
        return math.log((self._n_docs + 1) / (df + 1)) + 1.0

    def top_overlap_terms(
        self,
        text_a: str,
        text_b: str,
        top_n: int = 3,
    ) -> list[str]:
        """Return the top-N overlapping terms ranked by IDF."""
        tokens_a = set(_tokenize(text_a))
        tokens_b = set(_tokenize(text_b))
        overlap = tokens_a & tokens_b
        if not overlap:
            return []
        scored = [(t, self.idf(t)) for t in overlap]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:top_n]]


# -------------------------------------------------------------------
# Module-level scorer (lazy init)
# -------------------------------------------------------------------

_scorer: _TfIdfTermScorer | None = None


def init_tfidf_scorer(corpus_texts: list[str]) -> None:
    """Initialise the module-level TF-IDF scorer.

    Call this once during startup with a list of chunk texts from the
    index.  If never called, TF-IDF overlap explanations are skipped.

    Args:
        corpus_texts: List of document/chunk text strings.
    """
    global _scorer  # noqa: PLW0603
    _scorer = _TfIdfTermScorer()
    _scorer.fit(corpus_texts)
    logger.info(
        "TF-IDF scorer initialised with %d documents.", len(corpus_texts),
    )


def _get_scorer() -> _TfIdfTermScorer:
    """Return the module-level scorer, or a fallback empty one."""
    global _scorer  # noqa: PLW0603
    if _scorer is None:
        _scorer = _TfIdfTermScorer()
        _scorer._n_docs = 1  # avoid division by zero
    return _scorer


# -------------------------------------------------------------------
# Result field accessor
# -------------------------------------------------------------------

def _field(result: Any, name: str, default: Any = "") -> Any:
    """Get a field from a RetrievalResult object or dict."""
    if hasattr(result, name):
        return getattr(result, name)
    if isinstance(result, dict):
        return result.get(name, default)
    return default


# -------------------------------------------------------------------
# DPR similarity helper (optional)
# -------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (pure Python)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# -------------------------------------------------------------------
# Main explanation function
# -------------------------------------------------------------------

def explain_result(
    query_fact_sheet: FactSheet,
    result: Any,
    dpr_index: Any | None = None,
) -> str:
    """Generate a grounded one-to-two sentence explanation.

    Strategy order:
        1. Shared ``act_sections`` between query and result metadata.
        2. Matching party types.
        3. TF-IDF top-overlap terms between combined fact-sheet text
           and result chunk text.
        4. Fallback: generic semantic similarity statement.

    Args:
        query_fact_sheet: The query case's structured fact sheet.
        result: A retrieval result (RetrievalResult or dict).
        dpr_index: Optional DPR index for embedding-based similarity.

    Returns:
        One-to-two sentence explanation string.
    """
    explanations: list[str] = []

    # --- 1. Shared act sections ---
    query_sections = set(query_fact_sheet.sections_cited)
    result_sections_raw = _field(result, "act_sections", "")
    if isinstance(result_sections_raw, list):
        result_sections = set(result_sections_raw)
    else:
        result_sections = {
            s.strip()
            for s in str(result_sections_raw).split(",")
            if s.strip()
        }

    shared_sections = query_sections & result_sections
    if shared_sections:
        sect_str = ", ".join(sorted(shared_sections))
        explanations.append(
            f"This case involves {sect_str}, which is also cited in your case."
        )

    # --- 2. Matching party types ---
    q_pet = query_fact_sheet.parties.petitioner_type or ""
    q_resp = query_fact_sheet.parties.respondent_type or ""

    r_text = str(_field(result, "text", "")).lower()
    r_case_id = _field(result, "case_id", "")

    if q_pet and q_pet.lower() in r_text:
        explanations.append(
            f"This case also involves a {q_pet} as the petitioner."
        )
    if q_resp and q_resp.lower() in r_text:
        explanations.append(
            f"This case also involves a {q_resp} as the respondent."
        )

    # --- 3. TF-IDF term overlap ---
    if len(explanations) < 2:
        scorer = _get_scorer()
        # Build query text from fact sheet fields
        query_text_parts = [
            query_fact_sheet.case_id,
            q_pet,
            q_resp,
            " ".join(query_fact_sheet.sections_cited),
            query_fact_sheet.outcome or "",
        ]
        for ev in query_fact_sheet.evidence_items:
            query_text_parts.append(ev.description)
        query_text = " ".join(query_text_parts)

        chunk_text = str(_field(result, "text", ""))
        top_terms = scorer.top_overlap_terms(query_text, chunk_text, top_n=3)

        if top_terms:
            terms_str = ", ".join(top_terms)
            explanations.append(
                f"This case shares similar factual patterns "
                f"involving {terms_str}."
            )

    # --- 4. Fallback ---
    if not explanations:
        return (
            "This case was retrieved based on overall semantic "
            "similarity to your case."
        )

    return " ".join(explanations[:2])


# -------------------------------------------------------------------
# Batch explanation helper
# -------------------------------------------------------------------

def explain_results(
    query_fact_sheet: FactSheet,
    results: list[Any],
    dpr_index: Any | None = None,
) -> dict[str, str]:
    """Generate explanations for a list of results.

    Args:
        query_fact_sheet: The query case's fact sheet.
        results: List of retrieval results.
        dpr_index: Optional DPR index.

    Returns:
        Dict mapping chunk_id (str) to explanation (str).
    """
    explanations: dict[str, str] = {}
    for r in results:
        chunk_id = str(_field(r, "chunk_id", ""))
        if not chunk_id:
            continue
        explanations[chunk_id] = explain_result(
            query_fact_sheet, r, dpr_index=dpr_index,
        )
    return explanations
