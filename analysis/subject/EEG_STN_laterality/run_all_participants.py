#!/usr/bin/env python
"""Review-confirm-then-run the participant STN laterality analysis for selected subjects."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from A00_review_stimulation_blocks import (
    review_rows, preserve_confirmations, write_html, events_path
)
from A01_participant_laterality_analysis import analyze_subject
from A00_review_stimulation_blocks import read_reference
from pipeline_config import resolve_project_root

VALID = {"off", "stim_left", "stim_right", "exclude"}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", required=True,
                   help="Participant IDs, e.g. 102 103 104. Use IDs without sub- prefix.")
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="SpAtt")
    p.add_argument("--run", default="01")
    p.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    p.add_argument("--project-root", default=None)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--review-only", action="store_true",
                   help="Create/update the BIDS confirmation report but do not analyse.")
    args = p.parse_args()
    subjects = [str(s).removeprefix("sub-") for s in args.subjects]
    root = resolve_project_root(args.platform, args.project_root)
    out = root / "derivatives" / "reports" / "group" / "EEG_STN_laterality"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "confirm_stimulation_blocks.csv"
    html_path = out / "confirm_stimulation_blocks.html"

    rows = review_rows(root, subjects, args.session, args.task, args.run)
    rows = preserve_confirmations(rows, csv_path)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_html(rows, html_path, csv_path)
    print(f"\nBIDS block review:\n  {html_path}\n  {csv_path}")
    if args.review_only:
        print("Review-only requested. No TFR analysis has been run.")
        return

    review = pd.read_csv(csv_path, dtype=str).fillna("")
    errors, selected, excluded = [], [], []
    for sub in subjects:
        rows_sub = review[review["subject"].astype(str) == sub]
        if len(rows_sub) != 4:
            errors.append(f"sub-{sub}: expected 4 identifiable block rows, found {len(rows_sub)}. Check BIDS block_onset/block_end events.")
            continue
        confirmed = rows_sub["confirmed"].astype(str).str.strip().str.lower()
        if not confirmed.isin(["yes", "y", "true", "1"]).all():
            errors.append(f"sub-{sub}: one or more blocks are not explicitly confirmed in the CSV.")
            continue
        labels = rows_sub["confirmed_condition"].astype(str).str.strip().str.lower().tolist()
        if all(x == "exclude" for x in labels):
            excluded.append(sub)
            continue
        if any(x == "exclude" for x in labels):
            errors.append(f"sub-{sub}: use 'exclude' for all four rows to exclude the participant, or label all four blocks.")
            continue
        if any(x not in VALID for x in labels):
            errors.append(f"sub-{sub}: confirmed_condition must be off, stim_left, stim_right, or exclude.")
            continue
        if labels.count("off") != 2 or labels.count("stim_left") != 1 or labels.count("stim_right") != 1:
            errors.append(f"sub-{sub}: expected two off blocks, one stim_left and one stim_right; got {labels}.")
            continue
        selected.append(sub)

    if errors:
        print("\nNOT RUN: the following review issues must be resolved first:")
        for msg in errors:
            print(" - " + msg)
        print(f"\nEdit the CSV after checking the HTML report, then rerun this same command:\n  {csv_path}")
        raise SystemExit(2)
    if excluded:
        print("Participants explicitly marked for exclusion (not analysed): " + ", ".join("sub-"+s for s in excluded))
    if not selected:
        print("No confirmed participants remain for analysis.")
        return

    run_rows = []
    for sub in selected:
        try:
            report = analyze_subject(root, sub, review, args)
            run_rows.append({"subject": "sub-"+sub, "status": "success", "report": str(report), "error": ""})
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"\nFAILED sub-{sub}: {msg}")
            run_rows.append({"subject": "sub-"+sub, "status": "failed", "report": "", "error": msg})
    summary = out / "runner_summary.csv"
    pd.DataFrame(run_rows).to_csv(summary, index=False)
    print(f"\nRunner summary: {summary}")
    print(f"Successful participants: {sum(r['status']=='success' for r in run_rows)}/{len(run_rows)}")

if __name__ == "__main__":
    main()
