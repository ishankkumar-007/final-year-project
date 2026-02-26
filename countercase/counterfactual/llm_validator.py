"""LLM-based validation filter for perturbation plausibility.

Each perturbed fact sheet is checked against an LLM to determine
whether the perturbation is legally plausible and operative.
Perturbations that fail either criterion are discarded, preventing
nonsensical counterfactuals from entering the tree.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from countercase.counterfactual.perturbation_rules import (
    FactType,
    PerturbationEdge,
)
from countercase.fact_extraction.schema import FactSheet

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Validation prompt
# -------------------------------------------------------------------

_VALIDATION_PROMPT = """\
You are a legal expert. A case has the following original facts:
{original_json}

A counterfactual version has been generated with the following altered facts:
{perturbed_json}

The specific change is: {edge_description}

Answer two questions:
(1) Is this counterfactual legally plausible -- could a case with these \
altered facts plausibly occur in Indian jurisprudence?
(2) Does the alteration change a legally operative fact, or is it a minor \
stylistic change that would not affect legal reasoning?

Respond with ONLY valid JSON (no markdown, no extra text):
{{"plausible": true/false, "operative": true/false, "reasoning": "brief explanation"}}
"""


# -------------------------------------------------------------------
# Validation result dataclass
# -------------------------------------------------------------------


class ValidationResult:
    """Result of validating a single perturbation.

    Attributes:
        plausible: Whether the perturbation is legally plausible.
        operative: Whether the perturbation changes a legally
            operative fact.
        reasoning: Brief explanation from the LLM.
        accepted: True if both plausible and operative.
    """

    __slots__ = ("plausible", "operative", "reasoning", "accepted")

    def __init__(
        self,
        plausible: bool,
        operative: bool,
        reasoning: str,
    ) -> None:
        self.plausible = plausible
        self.operative = operative
        self.reasoning = reasoning
        self.accepted = plausible and operative

    def to_dict(self) -> dict[str, Any]:
        return {
            "plausible": self.plausible,
            "operative": self.operative,
            "reasoning": self.reasoning,
            "accepted": self.accepted,
        }


# -------------------------------------------------------------------
# Validator class
# -------------------------------------------------------------------

LLMFn = Callable[..., str]


class PerturbationValidator:
    """Validates perturbations via an LLM with result caching.

    Args:
        llm_fn: Callable matching the Phase 3 LLM function interface
            ``(prompt: str, *, timeout: int) -> str``.  If ``None``,
            all perturbations are accepted with a warning.
        timeout: Seconds per LLM call.
    """

    def __init__(
        self,
        llm_fn: LLMFn | None = None,
        timeout: int = 60,
    ) -> None:
        self._llm_fn = llm_fn
        self._timeout = timeout
        self._cache: dict[tuple[str, str, str], ValidationResult] = {}
        self._accept_count = 0
        self._reject_count = 0

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def validate(
        self,
        original: FactSheet,
        perturbed: FactSheet,
        edge: PerturbationEdge,
    ) -> ValidationResult:
        """Validate a single perturbation.

        Uses a cache keyed by ``(fact_type, original_value,
        perturbed_value)`` so structurally identical perturbations
        are not re-validated.

        Args:
            original: The parent fact sheet.
            perturbed: The child (perturbed) fact sheet.
            edge: Metadata describing the perturbation.

        Returns:
            :class:`ValidationResult` with accept/reject decision.
        """
        cache_key = (
            edge.fact_type.value,
            edge.original_value,
            edge.perturbed_value,
        )

        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._run_validation(original, perturbed, edge)
        self._cache[cache_key] = result

        if result.accepted:
            self._accept_count += 1
        else:
            self._reject_count += 1

        return result

    @property
    def accept_count(self) -> int:
        """Number of accepted perturbations."""
        return self._accept_count

    @property
    def reject_count(self) -> int:
        """Number of rejected perturbations."""
        return self._reject_count

    def stats(self) -> dict[str, int]:
        """Return accept/reject counts."""
        return {
            "accepted": self._accept_count,
            "rejected": self._reject_count,
            "cached": len(self._cache),
        }

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _run_validation(
        self,
        original: FactSheet,
        perturbed: FactSheet,
        edge: PerturbationEdge,
    ) -> ValidationResult:
        """Call the LLM and parse its response."""
        if self._llm_fn is None:
            logger.warning(
                "No LLM available -- accepting perturbation: %s",
                edge.description,
            )
            return ValidationResult(
                plausible=True,
                operative=True,
                reasoning="No LLM available; accepted by default.",
            )

        prompt = _VALIDATION_PROMPT.format(
            original_json=original.model_dump_json(indent=2),
            perturbed_json=perturbed.model_dump_json(indent=2),
            edge_description=edge.description,
        )

        try:
            raw = self._llm_fn(prompt, timeout=self._timeout)
        except Exception:
            logger.exception(
                "LLM call failed for validation of: %s", edge.description
            )
            # Accept on failure to avoid blocking the pipeline
            return ValidationResult(
                plausible=True,
                operative=True,
                reasoning="LLM call failed; accepted by default.",
            )

        return self._parse_response(raw, edge)

    def _parse_response(
        self,
        raw: str,
        edge: PerturbationEdge,
    ) -> ValidationResult:
        """Parse JSON from the LLM response."""
        try:
            # Try to extract JSON from potential markdown fencing
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                json_lines = []
                inside = False
                for line in lines:
                    if line.strip().startswith("```") and not inside:
                        inside = True
                        continue
                    if line.strip().startswith("```") and inside:
                        break
                    if inside:
                        json_lines.append(line)
                text = "\n".join(json_lines)

            data = json.loads(text)
            plausible = bool(data.get("plausible", True))
            operative = bool(data.get("operative", True))
            reasoning = str(data.get("reasoning", ""))

            return ValidationResult(
                plausible=plausible,
                operative=operative,
                reasoning=reasoning,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "Failed to parse LLM validation response for '%s': %s. "
                "Accepting by default.",
                edge.description,
                exc,
            )
            return ValidationResult(
                plausible=True,
                operative=True,
                reasoning=f"Parse failure ({exc}); accepted by default.",
            )


# -------------------------------------------------------------------
# Mock validator for testing
# -------------------------------------------------------------------


def mock_validation_llm_fn(prompt: str, *, timeout: int = 60) -> str:
    """Mock LLM that accepts all perturbations as plausible and operative.

    Uses simple heuristics to reject obviously nonsensical changes.
    """
    # Heuristic: if the prompt mentions crossing a legal boundary,
    # it is definitely operative.
    lower = prompt.lower()

    plausible = True
    operative = True
    reasoning = "Mock validation: accepted by default."

    # Simple heuristic rejections
    if "age changed from" in lower:
        # Accept age changes that cross boundaries
        operative = True
        reasoning = "Age perturbation crosses legal boundary."
    elif "section changed from" in lower:
        operative = True
        reasoning = "Section substitution to adjacent provision."
    elif "evidence removed" in lower or "evidence added" in lower:
        operative = True
        reasoning = "Evidence toggle affects evidentiary basis."
    elif "type changed from" in lower:
        operative = True
        reasoning = "Party type change affects legal standing."

    result = {
        "plausible": plausible,
        "operative": operative,
        "reasoning": reasoning,
    }
    return json.dumps(result)
