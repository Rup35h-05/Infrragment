"""
synth_data.py
--------------
We don't have access to real mortgage loan files (and shouldn't -- they're
sensitive, which is exactly why the original article couldn't publish its
code or data either). This module builds a synthetic but structurally
realistic "loan package" PDF: several distinct document types, three
back-to-back instances of the *same* type (Form 1040 across three tax
years, the exact stress case called out in the InfrX brief), a multi-page
table (bank statement), and one page type with no known signature at all
(to honestly exercise the similarity-fallback path instead of only
testing the easy cases).

It also writes out ground_truth.json: the answer key, used by evaluate.py
to score the pipeline the same way the article scores its model
(accuracy / F1, here at the document-boundary level).
"""
from __future__ import annotations

import json
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

styles = getSampleStyleSheet()
TITLE = styles["Heading1"]
BODY = styles["Normal"]
SMALL = styles["Normal"]


def _para(text, style=BODY):
    return Paragraph(text, style)


def _bank_table(rows):
    t = Table(rows, colWidths=[1.3 * inch, 2.6 * inch, 1.3 * inch])
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def build_demo_package(pdf_path: str) -> list[dict]:
    """Writes a 16-page synthetic loan package to pdf_path.

    Returns the ground-truth document list:
        [{"doc_type": ..., "start_page": ..., "end_page": ..., "instance_key": ...}, ...]
    """
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    all_pages: list[list] = []  # list of flowable-lists, one per physical page
    ground_truth = []
    page = 1

    def emit(doc_type, n_pages, instance_key, page_flowables_per_page):
        """page_flowables_per_page: list of flowable-lists, each WITHOUT a
        trailing PageBreak -- page breaks are inserted centrally below so
        every physical page gets exactly one."""
        nonlocal page
        start = page
        all_pages.extend(page_flowables_per_page)
        page += len(page_flowables_per_page)
        ground_truth.append(
            {
                "doc_type": doc_type,
                "start_page": start,
                "end_page": start + n_pages - 1,
                "instance_key": instance_key,
            }
        )

    # 1. Loan Application (1pg, no "Page X of Y")
    emit(
        "Loan_Application",
        1,
        None,
        [[
            _para("UNIFORM RESIDENTIAL LOAN APPLICATION", TITLE),
            Spacer(1, 12),
            _para("Borrower: Jane Doe. Property address: 12 Birch Street. Loan amount requested: $310,000."),
        ]],
    )

    # 2. Form 4506-T (1pg)
    emit(
        "Form_4506-T",
        1,
        None,
        [[
            _para("FORM 4506-T", TITLE),
            _para("Request for Transcript of Tax Return"),
            Spacer(1, 12),
            _para("Use this form to request a transcript of your federal tax return."),
        ]],
    )

    # 3-4 / 5-6 / 7-8: three years of Form 1040, back to back, same type
    for year in (2019, 2020, 2021):
        emit(
            "Form_1040",
            2,
            str(year),
            [
                [
                    _para("FORM 1040", TITLE),
                    _para("U.S. Individual Income Tax Return"),
                    _para(f"Tax Year: {year}"),
                    Spacer(1, 12),
                    _para("Filing status: Single. Wages, salaries, tips: $84,200."),
                    Spacer(1, 200),
                    _para("Page 1 of 2", SMALL),
                ],
                [
                    _para("Schedule details continued from page 1."),
                    _para("Total tax: $11,420. Refund: $640."),
                    Spacer(1, 200),
                    _para("Page 2 of 2", SMALL),
                ],
            ],
        )

    # 9. W-2 (1pg)
    emit(
        "W-2",
        1,
        "2021",
        [[
            _para("WAGE AND TAX STATEMENT", TITLE),
            _para("Form W-2"),
            _para("Tax Year: 2021"),
            Spacer(1, 12),
            _para("Employer: Acme Robotics Inc. Wages: $84,200. Federal income tax withheld: $9,870."),
        ]],
    )

    # 10-11. Two pay stubs, different pay periods, same type, back to back
    for period in ("01/01/2022-01/15/2022", "01/16/2022-01/31/2022"):
        emit(
            "Pay_Stub",
            1,
            period,
            [[
                _para("EARNINGS STATEMENT", TITLE),
                _para(f"Pay Period: {period}"),
                Spacer(1, 12),
                _para("Gross pay: $3,508.33. Net pay: $2,690.10."),
            ]],
        )

    # 12. Hazard Insurance Notice -- deliberately NOT in doc_signatures.py,
    # so the heuristic classifier has to fall back to similarity-drop
    # detection rather than a title match. Ground truth keeps the real
    # name so evaluate.py can honestly report this as a type the
    # registry doesn't know, separate from a boundary error.
    emit(
        "Hazard_Insurance_Notice",
        1,
        None,
        [[
            _para("Notice regarding hazard insurance coverage requirements", TITLE),
            Spacer(1, 12),
            _para("Borrower must maintain hazard insurance covering the replacement cost of the dwelling."),
        ]],
    )

    # 13-15. Bank statement, 3 pages, multi-page table with a repeated header
    header = ["Date", "Description", "Amount"]
    page1_rows = [header, ["01/03", "Payroll deposit", "$1,200.00"], ["01/05", "Grocery Mart", "-$84.21"]]
    page2_rows = [header, ["01/09", "Electric Co", "-$112.40"], ["01/14", "Payroll deposit", "$1,200.00"]]
    page3_rows = [header, ["01/22", "Mortgage Co", "-$1,840.00"], ["01/29", "ATM Withdrawal", "-$200.00"]]
    emit(
        "Bank_Statement",
        3,
        "January 2022",
        [
            [
                _para("ACCOUNT STATEMENT", TITLE),
                _para("Statement Period: January 2022"),
                Spacer(1, 10),
                _bank_table(page1_rows),
                Spacer(1, 150),
                _para("Page 1 of 3", SMALL),
            ],
            [
                _para("Transactions continued"),
                Spacer(1, 6),
                _bank_table(page2_rows),
                Spacer(1, 150),
                _para("Page 2 of 3", SMALL),
            ],
            [
                _para("Transactions continued"),
                Spacer(1, 6),
                _bank_table(page3_rows),
                Spacer(1, 150),
                _para("Page 3 of 3", SMALL),
            ],
        ],
    )

    # 16. Borrower letter (unstructured, free text)
    emit(
        "Borrower_Letter",
        1,
        None,
        [[
            _para("October 18, 2022", SMALL),
            Spacer(1, 12),
            _para("Dear Mr. White,"),
            Spacer(1, 8),
            _para(
                "I am writing to confirm that the down payment funds were a gift from "
                "my parents and are not expected to be repaid."
            ),
            Spacer(1, 8),
            _para("Sincerely, Jane Doe"),
        ]],
    )

    flow = []
    for i, page_flowables in enumerate(all_pages):
        flow.extend(page_flowables)
        if i != len(all_pages) - 1:
            flow.append(PageBreak())

    doc.build(flow)
    return ground_truth


def write_demo_assets(out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "synthetic_loan_package.pdf")
    gt_path = os.path.join(out_dir, "ground_truth.json")
    ground_truth = build_demo_package(pdf_path)
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    return pdf_path, gt_path
