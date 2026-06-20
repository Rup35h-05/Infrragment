# Loan-Package Logical Pagination Splitter

A working prototype for **InfrX 2026 — Problem B: Structuring the 2,000-Page File: Tables & Logical Pagination**.

It takes one giant, undifferentiated loan-package PDF and recovers the individual
documents inside it — exact start/end page for each instance, including telling
apart several documents of the *same type* sitting back to back (e.g. three years
of Form 1040) — then writes each one out as its own PDF plus a machine-readable
`manifest.json`. As a secondary feature (the brief also calls out tables), it
stitches multi-page tables (e.g. a bank statement's transaction history) into a
single clean CSV per document.

## Where this comes from

The approach is adapted from the "Multi Page Document Classification using
Machine Learning and NLP" article (Qaisar Tanvir), which frames document
splitting as a page-level classification problem with three label types —
**First page**, **Last page**, and **Other (middle page)** of each document
class — followed by a post-processing pass that walks the predicted labels and
turns them into page ranges. This project re-implements that same two-stage
shape (classify → walk → range), but adapted to run with **zero labeled
training data**, since (like the article's own dataset) real loan files aren't
something this project has access to:

| Article's component | This project |
|---|---|
| OCR → text per page | `pdf_extract.py` (text layer first, Tesseract OCR fallback only for image-only pages) |
| Doc2Vec text vectorizer | `vectorizer.py` — TF-IDF stand-in, same `fit`/`transform` shape so a real Doc2Vec model drops in later |
| Logistic Regression classifier over First/Last/Other | `page_classifier.py` — **two interchangeable classifiers**: a zero-data `HeuristicClassifier` (default) and a `TrainableClassifier` that is a direct re-implementation of the article's supervised pipeline for when labeled samples exist |
| Post-processing range-finding state machine | `boundary_detector.py` — generalized to also close out single-page document classes and back-to-back same-type instances |

## Why a heuristic default instead of training a model first

The article had 300+ labeled samples per document class to train Doc2Vec +
Logistic Regression. A hackathon prototype doesn't have that, and won't have it
for a brand-new customer's document set on day one either. So the default path
(`HeuristicClassifier`) needs no training data at all:

1. **Generic structural cue, checked first:** a `Page X of Y` footer is the
   single strongest, domain-independent signal for First/Other/Last role —
   most real templated documents carry one.
2. **Per-type title signatures** (`doc_signatures.py`): a small, editable
   registry of regexes per document type (`FORM 1040`, `WAGE AND TAX
   STATEMENT`, etc.) — the zero-data equivalent of the article's "First Page
   Class".
3. **Unsupervised fallback:** for a document type with no known signature at
   all (tested in the demo via a `Hazard_Insurance_Notice` page that is
   deliberately *not* registered), a sharp drop in TF-IDF cosine similarity
   between consecutive pages is used as a generic "something changed here"
   boundary cue, the same idea the article used Doc2Vec embeddings for.

`TrainableClassifier` exists for the path where labeled samples do become
available — it is the article's pipeline (vectorize → Logistic Regression
over `DocType` / `DocType-last` / `Other` labels), included so the project
doesn't stop being useful once real training data shows up.

## Why this matters for the brief's efficiency constraint

The whole classify-and-split pass runs on TF-IDF + regex + a state machine —
no LLM calls, no GPU, no per-page embedding from a large hosted model. On the
16-page synthetic demo it runs in well under a second; the cost scales
roughly linearly with page count and stays cheap at 2,000 pages, which is the
explicit ask in Problem B ("approaches light enough to imagine running close
to where the documents live").

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate a synthetic 16-page test loan package + answer key
#    (no real mortgage data is used or required anywhere in this project)
python cli.py demo --out sample_data

# 2. Split it
python cli.py split sample_data/synthetic_loan_package.pdf -o sample_output

# 3. Score the result against the answer key
python cli.py evaluate sample_output/manifest.json sample_data/ground_truth.json \
    --source-pdf sample_data/synthetic_loan_package.pdf

# Or all three in one go:
python cli.py full-demo --out sample_output
```

To run it on your own PDF:

```bash
python cli.py split /path/to/your_loan_package.pdf -o output_dir
```

Output directory contains one PDF per detected document instance, a
`manifest.json` describing every instance (type, page range, confidence,
whether it's flagged for manual review), and a `_table.csv` for any
document whose type is in `table_extractor.TABULAR_DOC_TYPES`.

## Demo results (synthetic 16-page package, 11 ground-truth documents)

The synthetic package deliberately includes the two hard cases called out in
the brief: three back-to-back instances of the same type (`Form_1040`, tax
years 2019/2020/2021) and a multi-page table (a 3-page bank statement). It
also includes one document type with no signature registered at all, to
honestly test the no-training-data fallback rather than only the easy cases.

```
page_role_accuracy:                 1.0   (16/16 pages)
boundary_exact_match_rate:          1.0   (11/11 documents, exact start/end page)
type_accuracy_on_matched_boundaries: 0.91 (10/11 — the one miss is the
                                            deliberately-unregistered type,
                                            correctly bounded, labeled
                                            "Unclassified", and flagged
                                            needs_review: true rather than
                                            silently mismerged)
```

This is a clean run because the synthetic data is well-behaved by
construction. The honest caveat: real scanned mortgage documents will have
noisier OCR, rotated pages, and templates that vary by state — the
`HeuristicClassifier`'s regex signatures will need to be extended
(`doc_signatures.py` is a plain dict, deliberately easy to grow), and the
`TrainableClassifier` path will matter a lot more once real labeled samples
are available, the same way the article reports a jump to ~93-94% accuracy
once Doc2Vec + Logistic Regression is trained on a real labeled corpus.

## How boundary detection actually works

`page_classifier.py` labels every page `FIRST` / `OTHER` / `LAST` plus a
document type guess. `boundary_detector.py` then walks that list and tracks
one "currently open document." The key rules (see the module docstring for
the full reasoning):

- A `FIRST` page always closes whatever was previously open and starts a new
  document — this is what correctly splits three back-to-back Form_1040
  instances into three ranges instead of merging them into one six-page block.
- A `LAST` page closes the open document if the type matches; a type mismatch
  or an orphan `LAST` with nothing open is still resolved into a range, but
  flagged (`closing_reason`) so a human reviewer can see exactly where the
  page-level model was unsure, instead of it being silently merged away.
- Anything still open at the end of the file is closed at the last page —
  this is what correctly handles single-page document classes, which (per
  the article) never get a dedicated "last page" class in the first place.

## Project structure

```
cli.py                  entry point (demo / split / evaluate / full-demo)
requirements.txt
src/
  pdf_extract.py         per-page text extraction + OCR fallback
  text_clean.py          lowercase / strip noise / stopwords (article's "Data cleaning" step)
  vectorizer.py           TF-IDF page vectorizer (Doc2Vec stand-in) + similarity helper
  doc_signatures.py       editable registry of document-type regex signatures
  page_classifier.py      HeuristicClassifier (default) + TrainableClassifier
  boundary_detector.py    FIRST/OTHER/LAST -> page-range state machine
  pdf_splitter.py         writes per-document PDFs + manifest.json
  table_extractor.py      multi-page table stitching (secondary feature)
  pipeline.py             orchestrates the full split
  synth_data.py           synthetic test PDF + ground truth generator
  evaluate.py             scores a manifest against ground truth
sample_data/, sample_output/   generated by `python cli.py full-demo`
```

## Honest limitations / what's next

- **OCR fallback is wired in but untuned** for real scan noise (skew,
  low-contrast, multi-column layouts) — it works (verified against a
  synthetic image-only page in testing), but a production version would want
  deskew/denoise preprocessing before Tesseract, or a dedicated OCR engine.
- **`doc_signatures.py` only knows the document types listed in it.** That's
  by design (it's the zero-training-data path), but it means recall on an
  unfamiliar document type depends entirely on the similarity-fallback path,
  which is weaker than a trained classifier — exactly why `TrainableClassifier`
  exists as the upgrade path once labeled samples are available.
- **Table stitching is intentionally simple** (exact header-row match across
  pages). Real bank statements sometimes reformat or omit the header on
  continuation pages; a more robust version would do fuzzy column alignment.
- **Mismatched/orphan boundary cases are surfaced, not resolved.** The
  `needs_review` flag in the manifest is the intended hook for a human-in-the-
  loop step, matching the confidence-threshold philosophy the article uses:
  let the model handle the confident majority, route the rest to a reviewer.
