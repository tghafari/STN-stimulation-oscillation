"""Run the all-channel EEG preprocessing and analysis stages in order.

Prerequisite
------------
Complete the posterior-channel pipeline first, so raw BIDS EEG and the
participant report exist. P00 screening must approve a participant before
the later stages run.

Stages
------
0. Screen posterior alpha and record the inclusion decision.
1. PyPREP/manual channel QC, stimulation segmentation, cue epoching and
   optional EEG rereferencing.
2. Fit one shared ICA on the two stimulation conditions and apply it.
3. Manually reject epochs and inspect spectral quality.
4. Compute participant ERP results.
5. Compute participant TFR results.
6. Add a manuscript-style methods and analysis overview.

Use --subjects or --range, --from-stage to resume, and
--continue-on-error to process subsequent participants after a failure.
The default platform is Mac and the default reference is average EEG.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pipeline_config import qc_dir, resolve_project_root  # noqa: E402

PREPROCESSING = HERE / "preprocessing"
SENSOR = HERE / "sensor"

STAGES = [
    (
        0, "posterior-alpha inclusion QC",
        PREPROCESSING / "P00_screen_subject.py",
    ),
    (
        1, "LP100 + PyPREP/manual channel QC -> epochs -> reference",
        PREPROCESSING / "P01_epoch_pyprep_reref.py",
    ),
    (
        2, "ICA on concatenated condition epochs",
        PREPROCESSING / "P02_ica.py",
    ),
    (
        3, "manual epoch rejection + spectral QC",
        PREPROCESSING / "P03_manual_epoch_rejection.py",
    ),
    (4, "ERP analysis", SENSOR / "A01_ERP.py"),
    (5, "TFR analysis", SENSOR / "A02_TFR.py"),
    (
        6, "manuscript-style analysis overview",
        SENSOR / "A03_analysis_overview.py",
    ),
]


def parse_args():
    """Parse participants, platform, EEG reference and resume options."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subjects", nargs="+")
    group.add_argument("--range", nargs=2, type=int)

    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument(
        "--platform", choices=["mac", "bluebear"], default="mac"
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--rereference", choices=["avg", "none"], default="avg",
        help="Use average EEG reference (default), or none.",
    )
    parser.add_argument(
        "--from-stage", type=int, choices=range(7), default=0
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def subjects(args):
    """Expand explicit IDs or an inclusive numeric participant range."""
    if args.subjects:
        return [
            str(subject).removeprefix("sub-")
            for subject in args.subjects
        ]
    return [
        str(subject)
        for subject in range(args.range[0], args.range[1] + 1)
    ]


def status(root, subject):
    """Read the saved P00 inclusion decision, if available."""
    path = qc_dir(root, subject) / "inclusion_decision.json"
    if path.exists():
        return json.loads(path.read_text()).get("decision")
    return None


def run_stage(script, subject, args):
    """Execute one stage in a subprocess with consistent arguments.

    Each stage runs in a fresh interpreter to isolate interactive MNE
    windows and preserve the stage-specific command-line interface.
    """
    command = [
        sys.executable, str(script),
        "--subject", subject, "--platform", args.platform,
    ]
    if script.name != "A03_analysis_overview.py":
        command.extend([
            "--session", args.session,
            "--task", args.task,
            "--run", args.run,
        ])
    if script.name == "P01_epoch_pyprep_reref.py":
        command.extend(["--rereference", args.rereference])
    if args.project_root:
        command.extend(["--project-root", args.project_root])

    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(HERE) + (os.pathsep + existing if existing else "")
    )
    print("\nRUNNING: " + " ".join(command))
    subprocess.run(command, check=True, env=environment)


def main():
    """Process participants sequentially, respecting P00 screening."""
    args = parse_args()
    root = resolve_project_root(args.platform, args.project_root)
    print(f"Platform: {args.platform} (default is mac)")
    print(
        "Rereferencing:",
        "average" if args.rereference == "avg" else "none",
    )

    for subject in subjects(args):
        try:
            for number, label, script in STAGES:
                if number < args.from_stage:
                    continue
                if number > 0 and status(root, subject) != "include":
                    print(
                        f"Stopping sub-{subject}: inclusion status is "
                        f"{status(root, subject)!r}."
                    )
                    break
                print(f"\nStage {number}: {label}")
                run_stage(script, subject, args)
        except Exception as exc:
            print(
                f"ERROR sub-{subject}: "
                f"{type(exc).__name__}: {exc}"
            )
            if not args.continue_on_error:
                raise


if __name__ == "__main__":
    main()
