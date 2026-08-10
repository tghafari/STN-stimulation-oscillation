# -*- coding: utf-8 -*-
"""Run the five active single-subject EEG scripts in sequence.

Pipeline order
--------------
1. preprocessing/P01_first_look_BIDS_conversion.py
2. preprocessing/P02_segmenting_stim.py
3. preprocessing/P03_epoching_SpAtt.py
4. sensor/A01_ERP.py
5. sensor/A02_three_channel_TFR.py

The original scripts are executed from source, but their subject/path variables are
patched in memory. The original files are not rewritten. This keeps this runner
compatible with the analysis code you already test line by line.

Stimulation crop times are stored persistently in
``analysis/subject/stimulation_cropped_time.json``. If both stim and no-stim crop
ranges already exist for a subject, the raw browser is skipped and the runnerAz
prints the requested message. If they are missing, the BIDS raw data is opened,
the user enters 2 or 4 crop times for each condition, and the table is saved
before P02 is run.

P03 remains interactive. Its existing MNE epoch browser opens with PO3, PO4 and
POz only; bad epochs marked there are saved by P03 before ERP and TFR continue.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, Iterable, List

import mne
from mne_bids import BIDSPath, read_raw_bids

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CROP_TABLE_PATH = HERE / "stimulation_cropped_time.json"

SCRIPT_ORDER = [
    HERE / "preprocessing" / "P01_first_look_BIDS_conversion.py",
    HERE / "preprocessing" / "P02_segmenting_stim.py",
    HERE / "preprocessing" / "P03_epoching_SpAtt.py",
    HERE / "sensor" / "A01_ERP.py",
    HERE / "sensor" / "A02_three_channel_TFR.py",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P01 -> P02 -> P03 -> ERP -> TFR for one or more subjects."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--subjects",
        nargs="+",
        help="Subject numbers without sub-, e.g. --subjects 115 116 117",
    )
    group.add_argument(
        "--range",
        nargs=2,
        metavar=("START", "END"),
        type=int,
        help="Inclusive numeric range, e.g. --range 115 123",
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="Root of the STN-in-PD data project (contains data/BIDS and derivatives).",
    )
    parser.add_argument(
        "--session", default="01", help="BIDS session label without ses- (default: 01)."
    )
    parser.add_argument(
        "--task", default="SpAtt", help="BIDS task label (default: SpAtt)."
    )
    parser.add_argument(
        "--run", default="01", help="BIDS run label without run- (default: 01)."
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next subject if one subject fails. Default: stop immediately.",
    )
    return parser.parse_args()


def _subjects_from_args(args: argparse.Namespace) -> List[str]:
    if args.subjects:
        subjects = [str(s).removeprefix("sub-") for s in args.subjects]
    else:
        start, end = args.range
        if end < start:
            raise ValueError("--range END must be >= START")
        subjects = [str(s) for s in range(start, end + 1)]
    return subjects


def _load_crop_table() -> Dict[str, Dict[str, List[float]]]:
    if not CROP_TABLE_PATH.exists():
        return {}
    with CROP_TABLE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_crop_table(table: Dict[str, Dict[str, List[float]]]) -> None:
    with CROP_TABLE_PATH.open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, sort_keys=True)
        f.write("\n")


def _valid_crop_times(values: object) -> bool:
    if not isinstance(values, list) or len(values) not in (2, 4):
        return False
    try:
        vals = [float(v) for v in values]
    except (TypeError, ValueError):
        return False
    return all(vals[i] < vals[i + 1] for i in range(0, len(vals), 2))


def _parse_crop_input(text: str) -> List[float]:
    parts = [p for p in re.split(r"[\s,;]+", text.strip()) if p]
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError("Crop times must be numbers.") from exc
    if not _valid_crop_times(values):
        raise ValueError(
            "Enter either 2 numbers (start end) or 4 numbers "
            "(start1 end1 start2 end2), with each start < end."
        )
    return values


def _ask_crop_times(label: str) -> List[float]:
    while True:
        raw = input(
            f"Enter {label} crop times in seconds (2 or 4 numbers): "
        ).strip()
        try:
            return _parse_crop_input(raw)
        except ValueError as exc:
            print(f"Invalid input: {exc}")


def _get_or_collect_crop_times(
    subject: str,
    project_root: Path,
    session: str,
    task: str,
    run: str,
) -> Dict[str, List[float]]:
    """Return crop times; open raw data only when the subject has none saved."""
    table = _load_crop_table()
    key = f"sub-{subject}"
    existing = table.get(key, {})

    if _valid_crop_times(existing.get("no-stim")) and _valid_crop_times(existing.get("stim")):
        print("this subject has cropped times for stim on and stim off")
        return {
            "no-stim": [float(v) for v in existing["no-stim"]],
            "stim": [float(v) for v in existing["stim"]],
        }

    bids_root = project_root / "data" / "BIDS"
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        root=bids_root,
        datatype="eeg",
        suffix="eeg",
    )

    print(
        f"No complete stimulation crop table found for {key}.\n"
        "Opening the BIDS raw data. Inspect the recording, then close the browser."
    )
    raw = read_raw_bids(
        bids_path=bids_path,
        verbose=True,
        extra_params={"preload": True},
    )
    raw.plot(block=True, title=f"{key}: identify stimulation ON/OFF crop times")

    no_stim = _ask_crop_times("NO-STIM")
    stim = _ask_crop_times("STIM")
    table[key] = {"no-stim": no_stim, "stim": stim}
    _save_crop_table(table)
    print(f"Saved crop times to {CROP_TABLE_PATH}")
    return table[key]


def _replace_assignment(source: str, name: str, value: str) -> str:
    """Replace a simple top-level string assignment in one of the legacy scripts."""
    pattern = rf"(?m)^(?P<indent>\s*){re.escape(name)}\s*=\s*(?:r)?['\"].*?['\"]\s*(?:#.*)?$"
    replacement = f"{name} = {value!r}"
    updated, n = re.subn(pattern, replacement, source, count=1)
    if n == 0:
        print(f"Note: variable {name!r} was not found in this script; leaving it unchanged.")
    return updated


def _patch_script_source(
    source: str,
    script_path: Path,
    subject: str,
    project_root: Path,
    session: str,
    task: str,
    run: str,
    crop_times: Dict[str, List[float]] | None,
) -> str:
    """Patch only runtime configuration; analysis logic stays in the original script."""
    values = {
        "subject": subject,
        "session": session,
        "task": task,
        "run": run,
        "project_root": str(project_root),
        "GITHUB_ROOT": str(REPO_ROOT),
    }
    for name, value in values.items():
        source = _replace_assignment(source, name, value)

    if script_path.name == "P02_segmenting_stim.py":
        if crop_times is None:
            raise RuntimeError("P02 requires resolved stimulation crop times.")

        # P02 currently opens raw unconditionally. The runner has already opened it
        # when input was needed, so suppress that duplicate browser.
        source = re.sub(
            r"(?m)^\s*raw\.plot\([^\n]*\)\s*(?:#.*)?$",
            "print('Crop times resolved by run_subject_pipeline.py; continuing segmentation.')",
            source,
            count=1,
        )

        # Override/extend the dictionary in memory, including subjects not yet listed
        # in P02 itself. No source file is edited.
        injection = (
            "\n# injected by run_subject_pipeline.py\n"
            f"stimulation_cropped_time['sub-{subject}_no-stim'] = {crop_times['no-stim']!r}\n"
            f"stimulation_cropped_time['sub-{subject}_stim'] = {crop_times['stim']!r}\n\n"
        )
        marker = "def make_segment"
        if marker not in source:
            raise RuntimeError("Could not locate make_segment() in P02_segmenting_stim.py")
        source = source.replace(marker, injection + marker, 1)

    return source


def _run_script(
    script_path: Path,
    subject: str,
    project_root: Path,
    session: str,
    task: str,
    run: str,
    crop_times: Dict[str, List[float]] | None = None,
) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Required analysis script not found: {script_path}")

    source = script_path.read_text(encoding="utf-8")
    source = _patch_script_source(
        source,
        script_path,
        subject,
        project_root,
        session,
        task,
        run,
        crop_times,
    )
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(script_path),
        "__package__": None,
    }
    exec(compile(source, str(script_path), "exec"), globals_dict, globals_dict)


def _run_subject(subject: str, args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).expanduser().resolve()
    key = f"sub-{subject}"
    print("\n" + "=" * 78)
    print(f"STARTING {key}")
    print("=" * 78)

    # P01 must run first because missing crop times are read from the BIDS file
    # produced/updated there.
    p01 = SCRIPT_ORDER[0]
    print(f"\n[{key}] 1/5 BIDS conversion: {p01.name}")
    _run_script(p01, subject, project_root, args.session, args.task, args.run)

    crop_times = _get_or_collect_crop_times(
        subject, project_root, args.session, args.task, args.run
    )

    print(f"\n[{key}] 2/5 stimulation segmentation: {SCRIPT_ORDER[1].name}")
    _run_script(
        SCRIPT_ORDER[1],
        subject,
        project_root,
        args.session,
        args.task,
        args.run,
        crop_times=crop_times,
    )

    print(
        f"\n[{key}] 3/5 epoching: {SCRIPT_ORDER[2].name}\n"
        "The epoch browser is interactive. Reject bad trials using PO3, PO4 and POz, "
        "then close the browser to save the cleaned epochs and continue."
    )
    _run_script(
        SCRIPT_ORDER[2], subject, project_root, args.session, args.task, args.run
    )

    print(f"\n[{key}] 4/5 ERP: {SCRIPT_ORDER[3].name}")
    _run_script(
        SCRIPT_ORDER[3], subject, project_root, args.session, args.task, args.run
    )

    print(f"\n[{key}] 5/5 TFR: {SCRIPT_ORDER[4].name}")
    _run_script(
        SCRIPT_ORDER[4], subject, project_root, args.session, args.task, args.run
    )

    print(f"\nFINISHED {key}")


def main() -> None:
    args = _parse_args()
    subjects = _subjects_from_args(args)

    print("Subjects to run: " + ", ".join(f"sub-{s}" for s in subjects))
    print(f"Project root: {Path(args.project_root).expanduser().resolve()}")
    print(f"Repository root: {REPO_ROOT}")
    print(f"Crop-time table: {CROP_TABLE_PATH}")

    failures = []
    for subject in subjects:
        try:
            _run_subject(subject, args)
        except KeyboardInterrupt:
            print("\nPipeline stopped by user.")
            raise
        except Exception as exc:
            failures.append((subject, str(exc)))
            print(f"\nFAILED sub-{subject}: {exc}", file=sys.stderr)
            traceback.print_exc()
            if not args.continue_on_error:
                raise

    if failures:
        print("\nCompleted with failures:")
        for subject, error in failures:
            print(f"  sub-{subject}: {error}")
        sys.exit(1)


    print("\nAll requested subjects completed successfully.")


if __name__ == "__main__":
    main()
