"""Stage 0: human inclusion decision based on the existing posterior-channel report.

This stage deliberately does NOT try to decide automatically whether alpha is
modulated. It opens the participant's most recent HTML report so the researcher
can inspect the established PO3/PO4/POz analysis, then records an explicit yes/no
in JSON. The pipeline reads that JSON before any all-channel preprocessing.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime
from pathlib import Path

from pipeline_config import POSTERIOR_SCREEN_CHANNELS, qc_dir, resolve_project_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True, help="BIDS subject label without 'sub-'.")
    p.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    p.add_argument("--project-root", default=None)
    p.add_argument(
        "--decision",
        choices=["include", "exclude"],
        default=None,
        help="Optional explicit decision. If omitted, you will be asked interactively.",
    )
    return p.parse_args()


def find_reports(project_root: Path, subject: str) -> list[Path]:
    """Find HTML reports in the two report locations used in this repository."""
    candidates = []
    for folder in (
        project_root / "derivatives" / "reports" / f"sub-{subject}",
        project_root / "derivatives" / "reports",
    ):
        if folder.exists():
            candidates.extend(folder.glob(f"*sub-{subject}*.html"))
            candidates.extend(folder.glob("*.html") if folder.name == f"sub-{subject}" else [])
    return sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)


def main() -> None:
    args = parse_args()
    subject = args.subject.removeprefix("sub-")
    root = resolve_project_root(args.platform, args.project_root)
    reports = find_reports(root, subject)

    print("\n" + "=" * 72)
    print(f"SUBJECT SCREENING: sub-{subject}")
    print("Inspect alpha modulation in: " + ", ".join(POSTERIOR_SCREEN_CHANNELS))
    print("This is a HUMAN inclusion decision; no automated threshold is used.")
    print("=" * 72)

    report_path = reports[0] if reports else None
    if report_path:
        print(f"Opening most recent report:\n  {report_path}")
        webbrowser.open(report_path.as_uri())
    else:
        print("WARNING: no existing HTML participant report was found automatically.")
        print("Open the posterior-channel report manually before making the decision.")

    decision = args.decision
    if decision is None:
        while True:
            answer = input(
                "\nDoes this subject show the posterior alpha modulation required "
                "for inclusion? [y = include / n = exclude]: "
            ).strip().lower()
            if answer in {"y", "yes"}:
                decision = "include"
                break
            if answer in {"n", "no"}:
                decision = "exclude"
                break
            print("Please answer y or n.")

    record = {
        "subject": f"sub-{subject}",
        "decision": decision,
        "criterion": "human inspection of alpha modulation in PO3, PO4, POz",
        "screen_channels": list(POSTERIOR_SCREEN_CHANNELS),
        "report_opened": str(report_path) if report_path else None,
        "timestamp_local": datetime.now().astimezone().isoformat(),
    }
    outfile = qc_dir(root, subject) / "inclusion_decision.json"
    outfile.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved inclusion decision: {decision.upper()}")
    print(f"Audit file: {outfile}")


if __name__ == "__main__":
    main()
