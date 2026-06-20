"""
evaluate.py
-----------
Scores a manifest.json against a ground_truth.json the same spirit as the
article's "Cumulative Error Evaluation Metric" + "Confusion Matrix"
sections, but at the level this project actually operates on: per-page
role classification (FIRST/OTHER/LAST) and whole-document boundary
recovery, since that's what "logical pagination" success or failure
looks like in practice.

Three numbers are reported, deliberately kept separate rather than
collapsed into one score, because they catch different failure modes:

  1. Page-role accuracy   - did we get FIRST/OTHER/LAST right per page?
  2. Boundary exact-match - did we recover the *exact* (start, end) page
                             range for every ground-truth document?
  3. Type-label accuracy  - among correctly-bounded documents, did we
                             also get the document TYPE right? (A type
                             the signature registry has never seen, like
                             Hazard_Insurance_Notice in the demo data,
                             will correctly bound but show up here as a
                             miss -- that's an honest result, not a bug.)
"""
from __future__ import annotations

import json


def _page_role_labels(ground_truth: list[dict], total_pages: int) -> list[str]:
    """Expand ground-truth document ranges into a per-page FIRST/OTHER/LAST label list."""
    labels = ["OTHER"] * total_pages
    for doc in ground_truth:
        start, end = doc["start_page"], doc["end_page"]
        labels[start - 1] = "FIRST"
        if end > start:
            labels[end - 1] = "LAST"
        # single-page documents keep their one page as FIRST, matching
        # the article's note that single-page classes have no "-last" class
    return labels


def evaluate(manifest: dict, ground_truth: list[dict], predicted_roles: list[str] | None = None) -> dict:
    total_pages = manifest["total_pages"]
    gt_labels = _page_role_labels(ground_truth, total_pages)

    report: dict = {"total_pages": total_pages, "ground_truth_documents": len(ground_truth)}

    if predicted_roles is not None:
        correct = sum(1 for a, b in zip(gt_labels, predicted_roles) if a == b)
        report["page_role_accuracy"] = round(correct / total_pages, 4)

    gt_ranges = {(d["start_page"], d["end_page"]) for d in ground_truth}
    pred_ranges = {(d["start_page"], d["end_page"]): d["doc_type"] for d in manifest["documents"]}

    exact_matches = gt_ranges & set(pred_ranges.keys())
    report["boundary_exact_match_count"] = len(exact_matches)
    report["boundary_exact_match_rate"] = round(len(exact_matches) / len(gt_ranges), 4) if gt_ranges else None

    gt_type_by_range = {(d["start_page"], d["end_page"]): d["doc_type"] for d in ground_truth}
    type_correct = sum(
        1 for r in exact_matches if pred_ranges[r] == gt_type_by_range[r]
    )
    report["type_accuracy_on_matched_boundaries"] = (
        round(type_correct / len(exact_matches), 4) if exact_matches else None
    )

    missed = sorted(gt_ranges - exact_matches)
    extra = sorted(set(pred_ranges.keys()) - exact_matches)
    report["missed_ground_truth_ranges"] = missed
    report["unexpected_predicted_ranges"] = extra

    return report


def load_and_evaluate(manifest_path: str, ground_truth_path: str) -> dict:
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)
    return evaluate(manifest, ground_truth)
