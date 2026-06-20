"""
table_extractor.py
--------------------
Secondary to the pagination work, but the InfrX brief calls out tables
explicitly, so this gives each detected document instance a best-effort
shot at structured tables -- useful for things like a multi-page bank
statement's transaction history.

Approach: run pdfplumber's table detector per page within the document's
page range, then stitch pages together by dropping a repeated header row
when it reappears on a continuation page (a very common real-world
pattern: the header repeats verbatim at the top of every page of the
table). This is intentionally simple -- a real system would want column
alignment/fuzzy-header-matching across pages, which is its own project.
"""
from __future__ import annotations

import pdfplumber

TABULAR_DOC_TYPES = {"Bank_Statement", "Pay_Stub"}


def extract_tables_for_range(pdf_path: str, start_page: int, end_page: int) -> list[list[str]]:
    """Return a single stitched table (list of rows) for the given 1-indexed page range."""
    rows: list[list[str]] = []
    header: list[str] | None = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_no in range(start_page, end_page + 1):
            page = pdf.pages[page_no - 1]
            for table in page.extract_tables():
                if not table:
                    continue
                this_header, body = table[0], table[1:]
                if header is None:
                    header = this_header
                    rows.append(header)
                if this_header == header:
                    rows.extend(body)
                else:
                    # Header didn't repeat verbatim on this page; treat
                    # the whole table (including its first row) as data.
                    rows.extend(table)
    return rows
