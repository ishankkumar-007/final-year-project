"""LLM-based fact sheet extraction from case facts text.

Provides a pluggable LLM backend: ``local_llm_fn`` for local Mistral-7B
inference and ``api_llm_fn`` for OpenAI-compatible API calls.  The
:class:`FactSheetExtractor` wraps either backend and handles prompt
construction, JSON parsing, schema validation, and retry logic.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable

from countercase.fact_extraction.schema import FactSheet

logger = logging.getLogger(__name__)

# -- Prompt templates ---------------------------------------------------

_SCHEMA_JSON: str = json.dumps(
    {
        "case_id": "<string>",
        "parties": {
            "petitioner_type": "<one of: Individual, Corporation, State, "
            "UnionOfIndia, Minor, Partnership, ForeignNational, "
            "PublicSector, PrivateSector, Employee, Contractor, "
            "Tenant, Licensee, Unknown>",
            "respondent_type": "<same vocabulary>",
        },
        "evidence_items": [
            {
                "evidence_type": "<DyingDeclaration, Confession, MedicalReport, "
                "Document, Witness, FIR, PostMortem, ExpertOpinion, "
                "ElectronicRecord, RecoveryMemo, SiteInspection, "
                "Circumstantial>",
                "description": "<brief description>",
            }
        ],
        "sections_cited": ["<ACT-Section, e.g. IPC-302, Constitution-Article-21>"],
        "numerical_facts": {
            "amounts": [{"value": 0.0, "unit": "rupees", "context": "..."}],
            "ages": [{"value": 0, "descriptor": "..."}],
            "durations": [{"value": 0.0, "unit": "years", "context": "..."}],
        },
        "outcome": "<Allowed | Dismissed | Disposed | Partly Allowed | Unknown | null>",
    },
    indent=2,
)

_EXTRACTION_PROMPT: str = (
    "You are a legal analyst specializing in Indian Supreme Court cases. "
    "Extract a structured fact sheet from the following case facts section. "
    "Return ONLY valid JSON matching this exact schema, with no additional text:\n\n"
    "{schema_json}\n\n"
    "Rules:\n"
    "- Only include information explicitly stated in the text.\n"
    "- For sections_cited, normalize to format \"ACT-Section\" "
    '(e.g., "IPC-302", "Constitution-Article-21").\n'
    "- For party types, use one of: Individual, Corporation, State, "
    "UnionOfIndia, Minor, Partnership, ForeignNational, PublicSector, "
    "PrivateSector, Employee, Contractor, Tenant, Licensee, Unknown.\n"
    "- If a field is not present in the text, use null or an empty list.\n\n"
    "Case facts:\n{facts_text}"
)

_EXAMPLE_FACT_SHEET: str = json.dumps(
    {
        "case_id": "Criminal Appeal 1234/2020",
        "parties": {
            "petitioner_type": "Individual",
            "respondent_type": "State",
        },
        "evidence_items": [
            {
                "evidence_type": "FIR",
                "description": "FIR No. 123/2019 registered at PS Kotwali",
            },
            {
                "evidence_type": "MedicalReport",
                "description": "Post-mortem report showing cause of death as "
                "hemorrhagic shock",
            },
        ],
        "sections_cited": ["IPC-302", "IPC-34", "Evidence-Section-27"],
        "numerical_facts": {
            "amounts": [],
            "ages": [{"value": 25, "descriptor": "age of the deceased"}],
            "durations": [
                {"value": 3, "unit": "years", "context": "period of incarceration"}
            ],
        },
        "outcome": "Dismissed",
    },
    indent=2,
)

_RETRY_PROMPT: str = (
    "Your previous response was not valid JSON that matches the schema. "
    "Here is an example of a correctly filled fact sheet:\n\n"
    "{example}\n\n"
    "Now extract the fact sheet again from the same case facts. "
    "Return ONLY valid JSON, nothing else:\n\n"
    "Case facts:\n{facts_text}"
)


# -- LLM backend implementations ---------------------------------------


def local_llm_fn(
    prompt: str,
    *,
    timeout: int = 60,
) -> str:
    """Call a local Mistral-7B model via ``transformers``.

    The model is lazy-loaded on first call to avoid import-time overhead.

    Args:
        prompt: The full prompt string.
        timeout: Maximum seconds to wait (best-effort).

    Returns:
        Raw model output as a string.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Local LLM requires 'transformers' and 'torch'. "
            "Install them or use api_llm_fn instead."
        ) from exc

    model_name = os.getenv("LOCAL_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

    if not hasattr(local_llm_fn, "_model"):
        logger.info("Loading local LLM: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        model.eval()
        local_llm_fn._model = model  # type: ignore[attr-defined]
        local_llm_fn._tokenizer = tokenizer  # type: ignore[attr-defined]

    tokenizer = local_llm_fn._tokenizer  # type: ignore[attr-defined]
    model = local_llm_fn._model  # type: ignore[attr-defined]

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def api_llm_fn(
    prompt: str,
    *,
    timeout: int = 60,
) -> str:
    """Call an OpenAI-compatible API endpoint.

    Configuration via environment variables:
        ``LLM_API_URL``  -- Base URL (default: ``https://api.openai.com/v1``)
        ``LLM_API_KEY``  -- API key
        ``LLM_API_MODEL`` -- Model name (default: ``gpt-3.5-turbo``)

    Args:
        prompt: The full prompt string.
        timeout: Request timeout in seconds.

    Returns:
        Raw model output as a string.
    """
    import requests

    base_url = os.getenv("LLM_API_URL", "https://api.openai.com/v1")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_API_MODEL", "gpt-3.5-turbo")

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY environment variable is not set. "
            "Set it or use local_llm_fn instead."
        )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 2048,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


# -- JSON extraction helpers --------------------------------------------


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from raw LLM output.

    Handles common issues: markdown code fences, leading/trailing text.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block
    brace_start = text.find("{")
    if brace_start == -1:
        return None

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start : i + 1])
                except json.JSONDecodeError:
                    return None

    return None


# -- Extractor class ----------------------------------------------------


class FactSheetExtractor:
    """Extract structured fact sheets from case facts text using an LLM.

    Args:
        llm_fn: Callable that takes a prompt string and returns the
            LLM response string.  Use :func:`local_llm_fn` or
            :func:`api_llm_fn`, or provide a custom callable.
        timeout: Maximum seconds per LLM call.
    """

    def __init__(
        self,
        llm_fn: Callable[..., str],
        timeout: int = 60,
    ) -> None:
        self._llm_fn = llm_fn
        self._timeout = timeout

    def extract(
        self,
        facts_text: str,
        case_id: str = "unknown",
    ) -> FactSheet | None:
        """Extract a fact sheet from facts section text.

        Makes up to two LLM calls: one with the base prompt, and a
        retry with an example-augmented prompt if the first fails.

        Args:
            facts_text: Raw text of the facts section.
            case_id: Case identifier to embed in the fact sheet.

        Returns:
            A validated ``FactSheet``, or ``None`` if extraction fails.
        """
        if not facts_text or not facts_text.strip():
            logger.warning("Empty facts text for case %s", case_id)
            return None

        # Truncate very long texts to avoid exceeding context windows.
        max_chars = 12_000
        truncated = facts_text[:max_chars]

        # -- Attempt 1: base prompt -------------------------------------
        prompt = _EXTRACTION_PROMPT.format(
            schema_json=_SCHEMA_JSON,
            facts_text=truncated,
        )

        raw = self._call_llm(prompt, case_id, attempt=1)
        if raw is not None:
            result = self._parse_and_validate(raw, case_id)
            if result is not None:
                return result

        # -- Attempt 2: example-augmented retry -------------------------
        retry_prompt = _RETRY_PROMPT.format(
            example=_EXAMPLE_FACT_SHEET,
            facts_text=truncated,
        )

        raw = self._call_llm(retry_prompt, case_id, attempt=2)
        if raw is not None:
            result = self._parse_and_validate(raw, case_id)
            if result is not None:
                return result

        logger.error(
            "Fact sheet extraction failed for case %s after 2 attempts",
            case_id,
        )
        return None

    def _call_llm(
        self,
        prompt: str,
        case_id: str,
        attempt: int,
    ) -> str | None:
        """Call the LLM with error handling."""
        try:
            t0 = time.perf_counter()
            result = self._llm_fn(prompt, timeout=self._timeout)
            elapsed = time.perf_counter() - t0
            logger.info(
                "LLM call for %s (attempt %d) completed in %.1fs",
                case_id,
                attempt,
                elapsed,
            )
            return result
        except Exception:
            logger.exception(
                "LLM call failed for case %s (attempt %d)",
                case_id,
                attempt,
            )
            return None

    def _parse_and_validate(
        self,
        raw: str,
        case_id: str,
    ) -> FactSheet | None:
        """Parse LLM output as JSON and validate against schema."""
        data = _extract_json_from_text(raw)
        if data is None:
            logger.warning(
                "Could not extract JSON from LLM output for case %s", case_id
            )
            return None

        # Inject case_id if missing or placeholder
        if "case_id" not in data or not data["case_id"]:
            data["case_id"] = case_id

        try:
            return FactSheet.model_validate(data)
        except Exception:
            logger.exception(
                "Pydantic validation failed for case %s", case_id
            )
            return None


# -- Mock LLM for testing without a real model --------------------------


def mock_llm_fn(prompt: str, *, timeout: int = 60) -> str:
    """Deterministic mock LLM that extracts facts using regex heuristics.

    Produces a best-effort fact sheet without any actual LLM.  Useful
    for testing the pipeline when no model is available.
    """
    # Try to find case_id in the prompt text
    case_match = re.search(
        r"(?:Criminal|Civil|Writ|SLP|Transfer)\s+"
        r"(?:Appeal|Petition|Case)\s*(?:No\.?\s*)?"
        r"[\d/\-]+",
        prompt,
        re.IGNORECASE,
    )
    case_id_str = case_match.group() if case_match else "unknown"

    # Extract sections cited
    section_matches = re.findall(
        r"\b(?:Section|Sec\.?|S\.)\s*(\d+[A-Za-z]?)\s*"
        r"(?:of\s+(?:the\s+)?)?(?:IPC|CPC|CrPC|Cr\.P\.C)?\b",
        prompt,
        re.IGNORECASE,
    )
    sections = []
    for s in section_matches[:10]:
        sections.append(f"IPC-{s}")

    article_matches = re.findall(
        r"\b(?:Article|Art\.?)\s*(\d+[A-Za-z]?)\b", prompt, re.IGNORECASE
    )
    for a in article_matches[:5]:
        sections.append(f"Constitution-Article-{a}")

    # Deduplicate
    sections = list(dict.fromkeys(sections))

    # Detect evidence types
    evidence_items = []
    evidence_map = {
        r"dying\s+declaration": ("DyingDeclaration", "Dying declaration"),
        r"confess(?:ion|ional)": ("Confession", "Confession statement"),
        r"FIR|first\s+information\s+report": ("FIR", "First Information Report"),
        r"post[\s-]*mortem": ("PostMortem", "Post-mortem report"),
        r"medical\s+(?:report|evidence)": ("MedicalReport", "Medical evidence"),
        r"eye[\s-]*witness|witness": ("Witness", "Witness testimony"),
    }
    for pat, (etype, desc) in evidence_map.items():
        if re.search(pat, prompt, re.IGNORECASE):
            evidence_items.append({"evidence_type": etype, "description": desc})

    # Detect outcome
    outcome = None
    if re.search(r"appeal\s+(?:is\s+)?allowed", prompt, re.IGNORECASE):
        outcome = "Allowed"
    elif re.search(r"appeal\s+(?:is\s+)?dismissed", prompt, re.IGNORECASE):
        outcome = "Dismissed"
    elif re.search(r"disposed\s+of", prompt, re.IGNORECASE):
        outcome = "Disposed"

    # Detect ages
    ages = []
    for m in re.finditer(r"aged?\s+(?:about\s+)?(\d+)", prompt, re.IGNORECASE):
        ages.append({"value": int(m.group(1)), "descriptor": "person"})

    # Detect amounts
    amounts = []
    for m in re.finditer(
        r"Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(?:(lakh|crore)s?)?",
        prompt,
        re.IGNORECASE,
    ):
        raw = m.group(1).replace(",", "").strip()
        if not raw:
            continue
        val = float(raw)
        unit = m.group(2) or "rupees"
        amounts.append({"value": val, "unit": unit, "context": "monetary amount"})

    result = {
        "case_id": case_id_str,
        "parties": {"petitioner_type": "Unknown", "respondent_type": "Unknown"},
        "evidence_items": evidence_items,
        "sections_cited": sections,
        "numerical_facts": {
            "amounts": amounts,
            "ages": ages,
            "durations": [],
        },
        "outcome": outcome,
    }
    return json.dumps(result, indent=2)
