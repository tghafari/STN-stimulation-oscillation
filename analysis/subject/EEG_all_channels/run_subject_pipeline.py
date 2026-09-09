"""Semi-automatic runner for the all-channel EEG preprocessing pipeline.

Stages
------
0. Open the existing posterior-channel report and record include/exclude.
1. Define cue epochs, detect bad channels with PyPREP, inspect/edit them, rereference.
2. Fit ICA, inspect components manually, confirm exclusions, apply ICA.
3. Inspect all good EEG channels and manually reject bad trials.

The runner intentionally stops for human decisions. It does not hide interactive
steps and it never modifies the source scripts in memory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pipeline_config import qc_dir, resolve_project_root

HERE = Path(__file__).resolve().parent
STAGES = [
    (0, "screen", HERE / "P00_screen_subject.py"),
    (1, "epoch-pyprep-reref", HERE / "P01_epoch_pyprep_reref.py"),
    (2, "ica", HERE / "P02_ica.py"),
    (3, "manual-epochs", HERE / "P03_manual_epoch_rejection.py"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--subjects", nargs="+", help="Example: --subjects 115 116 117")
    group.add_argument("--range", nargs=2, type=int, metavar=("START", "END"))
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="SpAtt")
    p.add_argument("--run", default="01")
    p.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    p.add_argument("--project-root", default=None)
    p.add_argument("--from-stage", type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def subjects_from_args(args: argparse.Namespace) -> list[str]:
    if args.subjects:
        return [str(s).removeprefix("sub-") for s in args.subjects]
    start, end = args.range
    if end < start:
        raise ValueError("--range END must be >= START")
    return [str(x) for x in range(start, end + 1)]


def inclusion_status(root: Path, subject: str) -> str | None:
    path = qc_dir(root, subject) / "inclusion_decision.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("decision")


def run_stage(script: Path, subject: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(script),
        "--subject", subject,
        "--session", args.session,
        "--task", args.task,
        "--run", args.run,
        "--platform", args.platform,
    ]
    if args.project_root:
        cmd.extend(["--project-root", args.project_root])

    # Stage 0 does not accept session/task/run arguments.
    if script.name == "P00_screen_subject.py":
        cmd = [
            sys.executable,
            str(script),
            "--subject", subject,
            "--platform", args.platform,
        ]
        if args.project_root:
            cmd.extend(["--project-root", args.project_root])

    print("\nRUNNING:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    root = resolve_project_root(args.platform, args.project_root)
    subjects = subjects_from_args(args)

    for subject in subjects:
        print("\n" + "#" * 80)
        print(f"ALL-CHANNEL PIPELINE: sub-{subject}")
        print("#" * 80)
        try:
            for number, label, script in STAGES:
                if number < args.from_stage:
                    continue

                if number > 0:
                    status = inclusion_status(root, subject)
                    if status != "include":
                        print(
                            f"Stopping sub-{subject} before Stage {number}: inclusion status is {status!r}. "
                            "Only subjects explicitly marked 'include' continue."
                        )
                        break

                print(f"\nStage {number}: {label}")
                run_stage(script, subject, args)

            else:
                print(f"\nsub-{subject}: preprocessing completed through manual epoch rejection.")

        except Exception as exc:
            print(f"\nERROR for sub-{subject}: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                raise
            print("Continuing to next subject because --continue-on-error was supplied.")


if __name__ == "__main__":
    main()
