"""
boundary_detector.py
---------------------
Post-processing: turns a per-page FIRST/OTHER/LAST stream into a clean
list of document instances with exact page ranges. This is a
reimplementation of the article's "Document Range Identification"
algorithm (FirstPageMemory / LastPageMemory walk over the page stream),
generalized to also close out:

  - single-page document classes (a FIRST with no matching LAST -- the
    article notes Last Page Classes only exist for classes with >1 page
    samples, so a single-page doc's range must close implicitly, either
    when the next FIRST appears or at end-of-package)
  - back-to-back instances of the *same* document type (three years of
    Form 1040 stacked together) -- because every FIRST unconditionally
    closes whatever was previously open, three FIRST hits in a row for
    "Form_1040" correctly yield three separate ranges instead of one
    six-page block. This is the specific failure mode called out in the
    InfrX brief for Problem B.
  - mismatched / orphan LAST pages (logged, not silently dropped), so a
    reviewer can see exactly where the page-level model was unsure rather
    than the boundary getting silently merged into its neighbor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .page_classifier import PagePrediction


@dataclass
class DocumentRange:
    doc_type: str
    start_page: int
    end_page: int
    page_count: int = field(init=False)
    closing_reason: str = "explicit_last_page"
    avg_confidence: float = 0.0

    def __post_init__(self):
        self.page_count = self.end_page - self.start_page + 1


def detect_boundaries(predictions: list[PagePrediction]) -> list[DocumentRange]:
    if not predictions:
        return []

    ranges: list[DocumentRange] = []
    open_doc: dict | None = None  # {"doc_type", "start_page", "confidences": [...]}

    def close(end_page: int, reason: str) -> None:
        nonlocal open_doc
        confs = open_doc["confidences"]
        ranges.append(
            DocumentRange(
                doc_type=open_doc["doc_type"],
                start_page=open_doc["start_page"],
                end_page=end_page,
                closing_reason=reason,
                avg_confidence=sum(confs) / len(confs) if confs else 0.0,
            )
        )
        open_doc = None

    for pred in predictions:
        if pred.role == "FIRST":
            if open_doc is not None:
                # Previous document never hit an explicit LAST page
                # (typically a single-page document class) -> close it
                # implicitly at the page right before this new FIRST.
                close(pred.page_number - 1, "implicit_next_first_page")
            open_doc = {
                "doc_type": pred.doc_type,
                "start_page": pred.page_number,
                "confidences": [pred.confidence],
            }

        elif pred.role == "LAST":
            if open_doc is not None and open_doc["doc_type"] == pred.doc_type:
                open_doc["confidences"].append(pred.confidence)
                close(pred.page_number, "explicit_last_page")
            elif open_doc is not None:
                # A LAST page fired for a different type than what's
                # currently open. Most likely the model mis-typed the
                # closing page; we still trust the page-position signal
                # and close the open document here, but flag it so a
                # reviewer can double check.
                open_doc["confidences"].append(pred.confidence)
                close(pred.page_number, "mismatched_last_page_type")
            else:
                # LAST with nothing open at all -- the FIRST page was
                # likely missed. Emit a single "orphan" page range rather
                # than silently dropping it.
                ranges.append(
                    DocumentRange(
                        doc_type=pred.doc_type,
                        start_page=pred.page_number,
                        end_page=pred.page_number,
                        closing_reason="orphan_last_page",
                        avg_confidence=pred.confidence,
                    )
                )

        else:  # OTHER
            if open_doc is None:
                # A middle page with no document open at all -- start an
                # "Unclassified" block rather than dropping the page.
                open_doc = {
                    "doc_type": "Unclassified",
                    "start_page": pred.page_number,
                    "confidences": [pred.confidence],
                }
            else:
                open_doc["confidences"].append(pred.confidence)

    if open_doc is not None:
        close(predictions[-1].page_number, "end_of_package")

    return ranges
