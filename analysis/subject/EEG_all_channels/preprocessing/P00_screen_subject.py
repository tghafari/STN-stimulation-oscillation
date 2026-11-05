"""Screen a participant for inclusion in the all-channel EEG workflow.

The posterior-channel pipeline must already have generated a participant
report. This stage opens that report and asks the researcher to judge
alpha modulation in PO3, PO4 and POz. There is no automatic threshold.

The human decision is saved as `inclusion_decision.json` in the
participant's all-channel QC directory. Later processing stages use
that decision to decide whether to continue.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from all_channel_report import (
    fmt_channels,
    participant_report,
    remove_old_posterior_channel_quality_section,
)
from pipeline_config import (
    POSTERIOR_SCREEN_CHANNELS,
    qc_dir,
    resolve_project_root,
)


def parse_args():
    """Read participant, dataset and optional explicit screening decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument(
        "--platform", choices=["mac", "bluebear"], default="mac"
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--decision", choices=["include", "exclude"], default=None
    )
    return parser.parse_args()


def open_pdf(path: Path):
    """Open the existing participant report using the operating system."""
    if sys.platform == "darwin":
        subprocess.run(["open", str(path.resolve())], check=False)
    elif sys.platform.startswith("win"):
        import os

        os.startfile(str(path.resolve()))
    else:
        subprocess.run(
            ["xdg-open", str(path.resolve())], check=False
        )


def main():
    """Record the researcher's posterior-alpha inclusion decision."""
    args = parse_args()
    subject = args.subject.removeprefix("sub-")
    root = resolve_project_root(args.platform, args.project_root)
    report = participant_report(root, subject)

    # Avoid duplicating the old posterior-pipeline channel-quality
    # section in the persistent participant PDF.
    remove_old_posterior_channel_quality_section(report)

    if report.pdf_fname.exists():
        print(f"Opening participant PDF:\n  {report.pdf_fname}")
        open_pdf(report.pdf_fname)
    else:
        print(
            f"WARNING: participant PDF not found: {report.pdf_fname}"
        )

    print(
        "\nInspect posterior alpha modulation in "
        + fmt_channels(POSTERIOR_SCREEN_CHANNELS)
    )
    decision = args.decision
    while decision is None:
        answer = input(
            "Posterior alpha modulation sufficient for inclusion? [y/n]: "
        ).strip().lower()
        if answer in {"y", "yes"}:
            decision = "include"
        elif answer in {"n", "no"}:
            decision = "exclude"

    record = {
        "subject": f"sub-{subject}",
        "decision": decision,
        "criterion": "human inspection of alpha modulation in PO3, PO4, POz",
        "timestamp_local": datetime.now().astimezone().isoformat(),
    }
    output = qc_dir(root, subject) / "inclusion_decision.json"
    output.write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Inclusion decision: {decision.upper()}")


if __name__ == "__main__":
    main()
