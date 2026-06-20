"""
doc_signatures.py
------------------
A small, editable registry of document types. For each type we keep:
  - first_page_patterns: regexes that strongly indicate "this page opens
    a document of this type" (mirrors the article's "First Page Classes")
  - instance_key_pattern: a regex used to tell apart multiple instances of
    the *same* document type sitting back-to-back in the file (e.g. three
    years of Form 1040 -- the instance key is the tax year). This directly
    targets the "logical pagination" requirement in the InfrX brief: same
    type, different instance, must not be merged into one block.

This is intentionally a plain Python dict, not a trained model -- it's the
zero-labeled-data path. `page_classifier.TrainableClassifier` is the
alternative path for when real labeled samples (like the article's 300+
samples per class) are available.

A generic "Page X of Y" footer pattern is also defined here; it is the
single strongest, domain-independent signal for first/continuation/last
page roles and is checked before any type-specific signature.
"""
from __future__ import annotations

import re

PAGE_OF_PATTERN = re.compile(r"page\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)

DOC_TYPES: dict[str, dict] = {
    "Loan_Application": {
        "first_page_patterns": [
            re.compile(r"uniform\s+residential\s+loan\s+application", re.I),
            re.compile(r"loan\s+application\s+summary", re.I),
        ],
        "instance_key_pattern": None,
    },
    "Form_4506-T": {
        "first_page_patterns": [
            re.compile(r"form\s+4506-?t", re.I),
            re.compile(r"request\s+for\s+transcript\s+of\s+tax\s+return", re.I),
        ],
        "instance_key_pattern": None,
    },
    "Form_1040": {
        "first_page_patterns": [
            re.compile(r"form\s+1040\b", re.I),
            re.compile(r"u\.?s\.?\s+individual\s+income\s+tax\s+return", re.I),
        ],
        "instance_key_pattern": re.compile(r"tax\s+year[:\s]+(\d{4})", re.I),
    },
    "W-2": {
        "first_page_patterns": [
            re.compile(r"wage\s+and\s+tax\s+statement", re.I),
            re.compile(r"\bform\s+w-?2\b", re.I),
        ],
        "instance_key_pattern": re.compile(r"tax\s+year[:\s]+(\d{4})", re.I),
    },
    "Pay_Stub": {
        "first_page_patterns": [
            re.compile(r"earnings\s+statement", re.I),
            re.compile(r"pay\s*stub", re.I),
        ],
        "instance_key_pattern": re.compile(r"pay\s+period[:\s]+([0-9/\-]+)", re.I),
    },
    "Bank_Statement": {
        "first_page_patterns": [
            re.compile(r"account\s+statement", re.I),
            re.compile(r"statement\s+period", re.I),
        ],
        "instance_key_pattern": re.compile(r"statement\s+period[:\s]+([a-z0-9 ,/\-]+)", re.I),
    },
    "Borrower_Letter": {
        # Unstructured free text -- no reliable title signature. Falls
        # back to the embedding-similarity boundary detector instead.
        "first_page_patterns": [
            re.compile(r"dear\s+(mr|mrs|ms|borrower)", re.I),
        ],
        "instance_key_pattern": None,
    },
}


def match_doc_type(text: str) -> str | None:
    """Return the first document type whose first-page signature fires, else None."""
    for doc_type, cfg in DOC_TYPES.items():
        for pattern in cfg["first_page_patterns"]:
            if pattern.search(text):
                return doc_type
    return None


def extract_instance_key(doc_type: str, text: str) -> str | None:
    cfg = DOC_TYPES.get(doc_type, {})
    pattern = cfg.get("instance_key_pattern")
    if not pattern:
        return None
    m = pattern.search(text)
    return m.group(1).strip() if m else None
