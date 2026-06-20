"""
pdf_splitter.py
----------------
Takes the original package PDF + the list of DocumentRange objects and:
  1. writes one sub-PDF per detected document instance
  2. resolves a human-readable, collision-free filename per instance using
     the instance-key extraction from doc_signatures (e.g. distinguishing
     three Form_1040 instances as Form_1040_2019.pdf / _2020.pdf / _2021.pdf
     instead of Form_1040_1.pdf / _2.pdf / _3.pdf when a key is found)
  3. returns the manifest (list of dicts) that pipeline.py writes to disk
     as manifest.json -- this is the artifact a downstream reviewer or
     system actually consumes, per the InfrX brief's "machine-usable"
     requirement.
"""
from __future__ import annotations

import os
import re

from pypdf import PdfReader, PdfWriter

from . import doc_signatures as sig
from .boundary_detector import DocumentRange

_SAFE = re.compile(r"[^A-Za-z0-9_\-]+")


def _safe_token(value: str) -> str:
    return _SAFE.sub("", value.replace(" ", "_"))[:40]


def split_pdf(
    pdf_path: str,
    page_texts: list[str],
    ranges: list[DocumentRange],
    output_dir: str,
) -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)
    reader = PdfReader(pdf_path)

    manifest: list[dict] = []
    type_counters: dict[str, int] = {}

    for rng in ranges:
        first_page_text = page_texts[rng.start_page - 1]
        instance_key = sig.extract_instance_key(rng.doc_type, first_page_text)

        if instance_key:
            suffix = _safe_token(instance_key)
        else:
            type_counters[rng.doc_type] = type_counters.get(rng.doc_type, 0) + 1
            suffix = str(type_counters[rng.doc_type])

        filename = f"{rng.doc_type}_{suffix}.pdf"
        out_path = os.path.join(output_dir, filename)

        # Guard against an instance-key collision (e.g. OCR misread the
        # same year twice) by falling back to a numeric suffix.
        if os.path.exists(out_path):
            type_counters[rng.doc_type] = type_counters.get(rng.doc_type, 0) + 1
            filename = f"{rng.doc_type}_{suffix}_{type_counters[rng.doc_type]}.pdf"
            out_path = os.path.join(output_dir, filename)

        writer = PdfWriter()
        for p in range(rng.start_page - 1, rng.end_page):  # 0-indexed, inclusive end
            writer.add_page(reader.pages[p])
        with open(out_path, "wb") as f:
            writer.write(f)

        manifest.append(
            {
                "file": filename,
                "doc_type": rng.doc_type,
                "instance_key": instance_key,
                "start_page": rng.start_page,
                "end_page": rng.end_page,
                "page_count": rng.page_count,
                "avg_confidence": round(rng.avg_confidence, 4),
                "closing_reason": rng.closing_reason,
                "needs_review": rng.closing_reason
                in {"mismatched_last_page_type", "orphan_last_page"}
                or rng.avg_confidence < 0.6,
            }
        )

    return manifest
