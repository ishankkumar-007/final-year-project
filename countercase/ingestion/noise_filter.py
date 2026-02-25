"""Noise filtering for Indian Supreme Court judgment text.

Removes cause-list boilerplate, normalizes unicode characters, and
collapses excessive whitespace.
"""

from __future__ import annotations

import re
import unicodedata

# Patterns that appear at the very start of a judgment file and should
# be stripped.  These include cause-list metadata, reportability headers,
# and similar boilerplate.
_CAUSELIST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*Diary\s+No\..*?\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*ITEM\s+NO\..*?\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*COURT\s+NO\..*?\n", re.IGNORECASE | re.MULTILINE),
    re.compile(
        r"^\s*(?:Reportable|Non[- ]?Reportable)\s*\n",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*IN\s+THE\s+SUPREME\s+COURT\s+OF\s+INDIA\s*\n",
        re.IGNORECASE | re.MULTILINE,
    ),
]

# Patterns that can appear in the leading portion of a document.
_LEADING_NOISE = re.compile(
    r"^(?:\s*(?:Diary\s+No\.|ITEM\s+NO\.|COURT\s+NO\.|"
    r"(?:Non[- ]?)?Reportable|IN\s+THE\s+SUPREME\s+COURT).*?\n)+",
    re.IGNORECASE | re.MULTILINE,
)

# Unicode replacements: curly quotes, em-dashes, non-breaking spaces, etc.
_UNICODE_MAP: dict[str, str] = {
    "\u2018": "'",   # left single curly quote
    "\u2019": "'",   # right single curly quote
    "\u201c": '"',   # left double curly quote
    "\u201d": '"',   # right double curly quote
    "\u2014": "--",  # em dash
    "\u2013": "-",   # en dash
    "\u00a0": " ",   # non-breaking space
    "\u2002": " ",   # en space
    "\u2003": " ",   # em space
    "\u200b": "",    # zero-width space
    "\ufeff": "",    # BOM
}

_UNICODE_RE = re.compile("|".join(re.escape(k) for k in _UNICODE_MAP))

# Excessive whitespace: 3+ consecutive blank lines, or 2+ spaces inline.
_MULTI_NEWLINES = re.compile(r"\n{3,}")
_MULTI_SPACES = re.compile(r"[ \t]{2,}")


def remove_causelist_noise(text: str) -> str:
    """Remove cause-list and boilerplate noise from the start of a judgment.

    Detects and strips patterns like "Diary No.", "ITEM NO.",
    "COURT NO.", and "Reportable"/"Non-Reportable" headers that precede
    the actual judgment text.

    Args:
        text: Raw judgment text.

    Returns:
        Text with leading boilerplate removed.
    """
    # Strip each known boilerplate pattern from the beginning.
    for pattern in _CAUSELIST_PATTERNS:
        text = pattern.sub("", text, count=1)
    return text.lstrip("\n")


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII equivalents.

    Replaces curly quotes, em-dashes, non-breaking spaces, and other
    typographic characters with their plain ASCII counterparts.  Also
    applies NFC normalization.

    Args:
        text: Input text with potential unicode anomalies.

    Returns:
        Normalized text.
    """
    text = unicodedata.normalize("NFC", text)
    text = _UNICODE_RE.sub(lambda m: _UNICODE_MAP[m.group()], text)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse excessive whitespace.

    Reduces runs of 3+ newlines to 2, and runs of 2+ spaces/tabs to a
    single space.

    Args:
        text: Input text.

    Returns:
        Text with collapsed whitespace.
    """
    text = _MULTI_NEWLINES.sub("\n\n", text)
    text = _MULTI_SPACES.sub(" ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Apply all noise-removal and normalization steps.

    Convenience function that runs cause-list removal, unicode
    normalization, and whitespace collapsing in sequence.

    Args:
        text: Raw judgment text.

    Returns:
        Fully cleaned text.
    """
    text = remove_causelist_noise(text)
    text = normalize_unicode(text)
    text = collapse_whitespace(text)
    return text
