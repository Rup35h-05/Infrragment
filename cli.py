"""
cli.py
------
Command-line entry point.

    python cli.py demo                         # generate synthetic test PDF + ground truth
    python cli.py split input.pdf -o out_dir    # split a real PDF
    python cli.py evaluate out_dir/manifest.json sample_data/ground_truth.json
    python cli.py full-demo                     # demo + split + evaluate, one shot
"""
from __future__ import annotations

import argparse
import json
import sys

from src.evaluate import evaluate
from src.page_classifier import HeuristicClassifier
from src.pdf_extract import extract_pages
from src.pipeline import run_pipeline
from src.synth_data import write_demo_assets


def cmd_demo(args):
    pdf_path, gt_path = write_demo_assets(args.out)
    print(f"Wrote synthetic package: {pdf_path}")
    print(f"Wrote ground truth:      {gt_path}")


def cmd_split(args):
    classifier = HeuristicClassifier()
    manifest = run_pipeline(args.pdf, args.out, classifier=classifier, extract_tables=not args.no_tables)
    print(json.dumps(manifest, indent=2))


def cmd_evaluate(args):
    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    predicted_roles = None
    if args.source_pdf:
        pages = extract_pages(args.source_pdf)
        preds = HeuristicClassifier().classify([p.text for p in pages])
        predicted_roles = [p.role for p in preds]

    report = evaluate(manifest, ground_truth, predicted_roles)
    print(json.dumps(report, indent=2))


def cmd_full_demo(args):
    pdf_path, gt_path = write_demo_assets(args.out)
    print(f"[1/3] Generated synthetic package -> {pdf_path}\n")

    classifier = HeuristicClassifier()
    manifest = run_pipeline(pdf_path, args.out, classifier=classifier)
    print(f"[2/3] Split into {manifest['documents_found']} documents -> {args.out}/")
    for d in manifest["documents"]:
        flag = "  <- needs review" if d["needs_review"] else ""
        print(f"    {d['file']:<40} pages {d['start_page']:>3}-{d['end_page']:<3} conf={d['avg_confidence']}{flag}")
    print()

    pages = extract_pages(pdf_path)
    preds = classifier.classify([p.text for p in pages])
    with open(gt_path) as f:
        ground_truth = json.load(f)
    report = evaluate(manifest, ground_truth, [p.role for p in preds])
    print("[3/3] Evaluation against ground truth:")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Loan-package logical pagination splitter")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="generate a synthetic test loan package")
    p_demo.add_argument("--out", default="sample_data")
    p_demo.set_defaults(func=cmd_demo)

    p_split = sub.add_parser("split", help="split a PDF into per-document files + manifest.json")
    p_split.add_argument("pdf")
    p_split.add_argument("-o", "--out", default="output")
    p_split.add_argument("--no-tables", action="store_true")
    p_split.set_defaults(func=cmd_split)

    p_eval = sub.add_parser("evaluate", help="score a manifest against ground truth")
    p_eval.add_argument("manifest")
    p_eval.add_argument("ground_truth")
    p_eval.add_argument("--source-pdf", help="also report page-role accuracy")
    p_eval.set_defaults(func=cmd_evaluate)

    p_full = sub.add_parser("full-demo", help="demo + split + evaluate in one go")
    p_full.add_argument("--out", default="sample_output")
    p_full.set_defaults(func=cmd_full_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
