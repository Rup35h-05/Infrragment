"""
pdf_extract.py
--------------
Step 1-2 of the pipeline (mirrors the article's "Package -> OCR -> Text" stage):
turn a multi-page PDF into one text blob per page.

Most pages in a born-digital PDF already have a selectable text layer, so we
use it directly (fast, free, no model needed). When a page has no text layer
(a scanned image, the case the original article is built around) we fall
back to Tesseract OCR for that page only -- this keeps the common case cheap
and only pays the OCR cost where it's actually required, which matters at
2,000-page scale.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pdfplumber

logger = logging.getLogger(__name__)

# A page with fewer than this many alphanumeric characters is treated as
# "no usable text layer" and routed to OCR.
MIN_TEXT_CHARS = 20


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str
    source: str  # "text_layer" or "ocr"


def _ocr_page(page) -> str:
    """OCR a single pdfplumber page. Returns '' if OCR is unavailable."""
    try:
        import pytesseract

        image = page.to_image(resolution=200).original
        return pytesseract.image_to_string(image)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("OCR fallback failed on page %s: %s", page.page_number, exc)
        return ""


def extract_pages(pdf_path: str, use_ocr_fallback: bool = True) -> list[PageText]:
    """Extract per-page text from a PDF.

    Parameters
    ----------
    pdf_path: path to the input PDF (a whole "loan package").
    use_ocr_fallback: if True, pages with no usable text layer are OCR'd.

    Returns
    -------
    A list of PageText, one per page, in page order.
    """
    pages: list[PageText] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            source = "text_layer"
            usable_chars = sum(c.isalnum() for c in text)
            if usable_chars < MIN_TEXT_CHARS and use_ocr_fallback:
                ocr_text = _ocr_page(page)
                if len(ocr_text.strip()) > len(text.strip()):
                    text, source = ocr_text, "ocr"
            pages.append(PageText(page_number=i, text=text, source=source))
    return pages
