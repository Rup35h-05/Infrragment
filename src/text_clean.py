"""
text_clean.py
-------------
Mirrors the "Data cleaning and transformation" step described in the
reference article: lowercase, strip non-alphanumeric noise, tokenize,
drop stopwords. Cheap, deterministic, and doesn't need a model -- it just
makes the downstream vectorizer/classifier see signal instead of OCR noise.
"""
from __future__ import annotations

import re

_STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "with", "as", "by", "at", "from", "this", "that", "be", "are", "was",
    "were", "it", "its", "if", "any", "all", "such", "shall", "may",
    "will", "would", "should", "has", "have", "had", "not", "no", "do",
    "does", "did", "you", "your", "i", "we", "our", "us", "he", "she",
    "they", "them", "their",
}

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """Lowercase + strip punctuation/noise + drop stopwords.

    Returns a single cleaned string (space-joined tokens), which is what
    the TF-IDF vectorizer expects.
    """
    if not raw:
        return ""
    lowered = raw.lower()
    no_punct = _NON_ALNUM.sub(" ", lowered)
    tokens = [t for t in no_punct.split() if t and t not in _STOPWORDS]
    return " ".join(tokens)


def tokenize(raw: str) -> list[str]:
    return clean_text(raw).split()
