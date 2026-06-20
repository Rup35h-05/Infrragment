"""
pipeline.py
-----------
End-to-end orchestration:

    PDF -> per-page text -> per-page FIRST/OTHER/LAST classification
        -> document ranges -> split PDFs + manifest.json (+ optional
           stitched tables for tabular doc types)

This mirrors the article's "Machine Learning Engine" diagram: Text
Vectorizer -> Classifier -> Label -> Post-processing -> Document Type,
just with the heuristic/trainable classifier swap described in
page_classifier.py.
"""
from __future__ import annotations

import csv
import json
import os
import time

from . import table_extractor
from .boundary_detector import detect_boundaries
from .page_classifier import HeuristicClassifier, PagePrediction, TrainableClassifier
from .pdf_extract import extract_pages
from .pdf_splitter import split_pdf


def run_pipeline(
    pdf_path: str,
    output_dir: str,
    classifier=None,
    extract_tables: bool = True,
) -> dict:
    """Run the full split pipeline on a single PDF.

    Parameters
    ----------
    pdf_path: input PDF (a whole loan package).
    output_dir: directory to write split PDFs + manifest.json into.
    classifier: an object with .classify(list[str]) -> list[PagePrediction].
        Defaults to HeuristicClassifier() (no training data required).
    extract_tables: if True, also writes a stitched CSV per document
        instance whose doc_type is in table_extractor.TABULAR_DOC_TYPES.

    Returns
    -------
    The manifest dict that's also written to {output_dir}/manifest.json.
    """
    t0 = time.time()
    classifier = classifier or HeuristicClassifier()

    pages = extract_pages(pdf_path)
    page_texts = [p.text for p in pages]

    predictions: list[PagePrediction] = classifier.classify(page_texts)
    ranges = detect_boundaries(predictions)

    os.makedirs(output_dir, exist_ok=True)
    manifest_entries = split_pdf(pdf_path, page_texts, ranges, output_dir)

    if extract_tables:
        for entry in manifest_entries:
            if entry["doc_type"] in table_extractor.TABULAR_DOC_TYPES:
                rows = table_extractor.extract_tables_for_range(
                    pdf_path, entry["start_page"], entry["end_page"]
                )
                if rows:
                    csv_name = entry["file"].replace(".pdf", "_table.csv")
                    with open(os.path.join(output_dir, csv_name), "w", newline="") as f:
                        csv.writer(f).writerows(rows)
                    entry["table_file"] = csv_name

    manifest = {
        "source_pdf": os.path.basename(pdf_path),
        "total_pages": len(pages),
        "documents_found": len(manifest_entries),
        "processing_seconds": round(time.time() - t0, 3),
        "ocr_pages_used": sum(1 for p in pages if p.source == "ocr"),
        "documents": manifest_entries,
    }

    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
