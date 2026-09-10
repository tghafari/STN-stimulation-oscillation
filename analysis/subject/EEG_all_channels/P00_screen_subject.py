"""Stage 0: human inclusion decision based on the existing posterior-channel PDF report.

For each participant, this stage looks for the report produced by the existing
posterior-channel analysis at:

    <project_root>/derivatives/reports/sub-<subject>/
        sub-<subject>_analysis_report.pdf

For example, sub-102 on Tara's Mac is expected at:

    /Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/
    STN-in-PD/derivatives/reports/sub-102/sub-102_analysis_report.pdf

The PDF is opened in the operating system's default PDF viewer so the researcher
can inspect the established PO3/PO4/POz alpha-modulation results. The inclusion
choice remains a HUMAN decision; this script does not impose an automatic alpha
threshold.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    """Return participant PDF reports, preferring the standard report filename.

    The first candidate is the exact filename produced by the existing report
    pipeline: ``sub-<subject>_analysis_report.pdf``. If that file is absent, the
    function searches the same participant report directory for other PDFs and
    returns the newest one first. This fallback makes the QC stage tolerant of
    older report naming conventions without accidentally searching another
    participant's folder.
    """
    report_folder = project_root / "derivatives" / "reports" / f"sub-{subject}"
    expected = report_folder / f"sub-{subject}_analysis_report.pdf"

    if expected.is_file():
        return [expected]

    if not report_folder.is_dir():
        return []

    candidates = list(report_folder.glob(f"sub-{subject}*.pdf"))
    if not candidates:
        candidates = list(report_folder.glob("*.pdf"))

    return sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)


def open_pdf(report_path: Path) -> None:
    """Open a local PDF with the platform's normal PDF viewer.

    On macOS this uses ``open`` (normally Preview or the user's chosen viewer).
    Linux/Bluebear uses ``xdg-open`` when available. Windows support is included
    for completeness when this function is reused interactively.
    """
    report_path = report_path.resolve()

    if sys.platform == "darwin":
        subprocess.run(["open", str(report_path)], check=False)
    elif sys.platform.startswith("win"):
        import os
        os.startfile(str(report_path))  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(report_path)], check=False)


def main() -> None:
    args = parse_args()
    subject = args.subject.removeprefix("sub-")
    root = resolve_project_root(args.platform, args.project_root)
    reports = find_reports(root, subject)

    print("\n" + "=" * 72)
    print(f"SUBJECT SCREENING: sub-{subject}")
    print("Inspect alpha modulation in: " + ", ".join(POSTERIOR_SCREEN_CHANNELS))
    print("This is a HUMAN inclusion decision; no automated threshold is used.")
    print(f"Project root: {root}")
    print("=" * 72)

    report_path = reports[0] if reports else None
    if report_path:
        print(f"Opening participant PDF report:\n  {report_path}")
        open_pdf(report_path)
    else:
        expected = (
            root
            / "derivatives"
            / "reports"
            / f"sub-{subject}"
            / f"sub-{subject}_analysis_report.pdf"
        )
        print("WARNING: no participant PDF report was found.")
        print("Expected the report here:")
        print(f"  {expected}")
        print("Check the project root above before making the inclusion decision.")

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
        "expected_report": str(
            root
            / "derivatives"
            / "reports"
            / f"sub-{subject}"
            / f"sub-{subject}_analysis_report.pdf"
        ),
        "timestamp_local": datetime.now().astimezone().isoformat(),
    }
    outfile = qc_dir(root, subject) / "inclusion_decision.json"
    outfile.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved inclusion decision: {decision.upper()}")
    print(f"Audit file: {outfile}")


if __name__ == "__main__":
    main()
