# -*- coding: utf-8 -*-
"""
==============================================
Run the five active single-subject EEG scripts in sequence.

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
==============================================
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

# -----------------------------------------------------------------------------
# Platform-specific data locations
# -----------------------------------------------------------------------------

BLUEBEAR_PROJECT_ROOT = Path(
    "/rds/projects/j/jenseno-avtemporal-attention/"
    "Projects/subcortical-structures/STN-in-PD"
)

BLUEBEAR_DATA_ROOT = BLUEBEAR_PROJECT_ROOT / "data" / "data-organised"
BLUEBEAR_BIDS_ROOT = BLUEBEAR_PROJECT_ROOT / "data" / "BIDS"

MAC_PROJECT_ROOT = Path(
    "/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/"
    "BEAR_outage/STN-in-PD"
)

MAC_DATA_ROOT = MAC_PROJECT_ROOT / "data" / "data-organised"
MAC_BIDS_ROOT = MAC_PROJECT_ROOT / "data" / "BIDS"


SCRIPT_ORDER = [
    HERE / "preprocessing" / "P01_first_look_BIDS_conversion.py",
    HERE / "preprocessing" / "P02_segmenting_stim.py",
    HERE / "preprocessing" / "P03_epoching_SpAtt.py",
    HERE / "sensor" / "A01_ERP.py",
    HERE / "sensor" / "A02_three_channel_TFR.py",
]

def _choose_platform():
    """Ask whether the pipeline is running on Bluebear or Mac."""

    while True:
        print("\nWhere are you running the analysis?")
        print("  1 = Bluebear")
        print("  2 = Mac")

        answer = input("Choose 1 or 2: ").strip().lower()

        if answer in {"1", "bluebear", "bear"}:
            return {
                "platform": "bluebear",
                "project_root": BLUEBEAR_PROJECT_ROOT,
                "data_root": BLUEBEAR_DATA_ROOT,
                "bids_root": BLUEBEAR_BIDS_ROOT,
            }

        if answer in {"2", "mac"}:
            return {
                "platform": "mac",
                "project_root": MAC_PROJECT_ROOT,
                "data_root": MAC_DATA_ROOT,
                "bids_root": MAC_BIDS_ROOT,
            }

        print("Please enter 1 for Bluebear or 2 for Mac.")

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


def _replace_assignment(
    source: str,
    name: str,
    value,
) -> str:

    pattern = (
        rf"(?m)^"
        rf"(?P<indent>\s*)"
        rf"{re.escape(name)}"
        rf"\s*=\s*"
        rf"(?:"
        rf"(?:r)?['\"].*?['\"]"
        rf"|None"
        rf")"
        rf"\s*(?:#.*)?$"
    )

    replacement = (
        f"{name} = {value!r}"
    )

    updated, n = re.subn(
        pattern,
        replacement,
        source,
        count=1,
    )

    if n == 0:
        print(
            f"WARNING: could not patch {name!r} "
            f"in {source[:100]!r}..."
        )

    return updated

def _patch_script_source(
    source: str,
    script_path: Path,
    subject: str,
    project_root: Path,
    data_root: Path,
    bids_root: Path,
    session: str,
    task: str,
    run: str,
    brainvision_basename: str | None,
    crop_times=None,
) -> str:

    values = {
        "subject": subject,
        "session": session,
        "task": task,
        "run": run,
        "project_root": str(project_root),
        "data_root": str(data_root),
        "bids_root": str(bids_root),
        "GITHUB_ROOT": str(REPO_ROOT),
        "brainVision_basename": brainvision_basename,
    }

    for name, value in values.items():
        if value is None:
            continue

        source = _replace_assignment(
            source,
            name,
            value,
        )

    return source

def _find_brainvision_basename(
    subject: str,
    data_root: Path,
    session: str,
) -> str:
    """
    Find the BrainVision filename basename for a subject.

    For ordinary subjects, exactly one .vhdr file is expected.

    For special subjects with multiple recordings:
        sub-110: *_blocks1-2.vhdr and *_blocks3-8.vhdr
        sub-111: *_stimright.vhdr, *_nostimright.vhdr,
                 and *_nostimleft.vhdr

    In those cases, the common basename before the special suffix
    is returned.
    """

    eeg_folder = (
        data_root
        / f"sub-{subject}"
        / f"ses-{session}"
        / "eeg"
    )

    if not eeg_folder.exists():
        raise FileNotFoundError(
            f"EEG folder does not exist for sub-{subject}:\n"
            f"{eeg_folder}"
        )

    vhdr_files = sorted(eeg_folder.glob("*.vhdr"))

    if not vhdr_files:
        raise FileNotFoundError(
            f"No BrainVision .vhdr files found for sub-{subject}:\n"
            f"{eeg_folder}"
        )

    # ----------------------------------------------------------
    # Special subject 110
    # ----------------------------------------------------------
    if subject == "110":

        expected_suffixes = [
            "_blocks1-2.vhdr",
            "_blocks3-8.vhdr",
        ]

        matches = []

        for suffix in expected_suffixes:

            candidates = [
                f for f in vhdr_files
                if f.name.endswith(suffix)
            ]

            if len(candidates) != 1:
                names = "\n".join(
                    f"  {f.name}" for f in candidates
                )

                raise RuntimeError(
                    f"Could not uniquely identify the "
                    f"{suffix} recording for sub-110.\n"
                    f"Candidates:\n{names}"
                )

            matches.append(candidates[0])

        basenames = [
            f.name[:-len(suffix)]
            for f, suffix in zip(
                matches,
                expected_suffixes,
            )
        ]

        if len(set(basenames)) != 1:
            raise RuntimeError(
                "The two sub-110 BrainVision files do not "
                "share a common basename:\n"
                + "\n".join(
                    f"  {f.name}"
                    for f in matches
                )
            )

        basename = basenames[0]

        print(
            f"Found BrainVision files for sub-110:"
        )
        for f in matches:
            print(f"  {f.name}")

        print(
            f"Using BrainVision basename for sub-110: "
            f"{basename}"
        )

        return basename

    # ----------------------------------------------------------
    # Special subject 111
    # ----------------------------------------------------------
    if subject == "111":

        expected_suffixes = [
            "_stimright.vhdr",
            "_nostimright.vhdr",
            "_nostimleft.vhdr",
        ]

        matches = []

        for suffix in expected_suffixes:

            candidates = [
                f for f in vhdr_files
                if f.name.endswith(suffix)
            ]

            if len(candidates) != 1:
                names = "\n".join(
                    f"  {f.name}" for f in candidates
                )

                raise RuntimeError(
                    f"Could not uniquely identify the "
                    f"{suffix} recording for sub-111.\n"
                    f"Candidates:\n{names}"
                )

            matches.append(candidates[0])

        basenames = [
            f.name[:-len(suffix)]
            for f, suffix in zip(
                matches,
                expected_suffixes,
            )
        ]

        if len(set(basenames)) != 1:
            raise RuntimeError(
                "The sub-111 BrainVision files do not "
                "share a common basename:\n"
                + "\n".join(
                    f"  {f.name}"
                    for f in matches
                )
            )

        basename = basenames[0]

        print(
            f"Found BrainVision files for sub-111:"
        )
        for f in matches:
            print(f"  {f.name}")

        print(
            f"Using BrainVision basename for sub-111: "
            f"{basename}"
        )

        return basename

    # ----------------------------------------------------------
    # Ordinary subjects
    # ----------------------------------------------------------

    if len(vhdr_files) != 1:

        names = "\n".join(
            f"  {f.name}"
            for f in vhdr_files
        )

        raise RuntimeError(
            f"Expected exactly one BrainVision .vhdr file "
            f"for sub-{subject}, but found {len(vhdr_files)}:\n"
            f"{names}"
        )

    basename = vhdr_files[0].stem

    print(
        f"Found BrainVision file for sub-{subject}: "
        f"{vhdr_files[0].name}"
    )

    return basename

def _run_script(
    script_path: Path,
    subject: str,
    project_root: Path,
    data_root: Path,
    bids_root: Path,
    session: str,
    task: str,
    run: str,
    brainvision_basename: str | None = None,
    crop_times=None,
) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Required analysis script not found: {script_path}")

    source = script_path.read_text(encoding="utf-8")
    source = _patch_script_source(
        source=source,
        script_path=script_path,
        subject=subject,
        project_root=project_root,
        data_root=data_root,
        bids_root=bids_root,
        session=session,
        task=task,
        run=run,
        brainvision_basename=brainvision_basename,
        crop_times=crop_times,
    )
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(script_path),
        "__package__": None,
    }

    if script_path.name == "P01_first_look_BIDS_conversion.py":
        print(
            "DEBUG: brainVision_basename in patched P01 source:"
        )

        match = re.search(
            r"(?m)^\s*brainVision_basename\s*=\s*.*$",
            source,
        )

        print(
            match.group(0)
            if match
            else "NOT FOUND"
        )
    exec(compile(source, str(script_path), "exec"), globals_dict, globals_dict)

def _run_subject(subject: str, args: argparse.Namespace) -> None:
    """Run the appropriate subject pipeline for the selected platform."""

    project_root = args.project_root
    data_root = args.data_root
    bids_root = args.bids_root

    key = f"sub-{subject}"

    print("\n" + "=" * 78)
    print(f"STARTING {key}")
    print("=" * 78)

    # ------------------------------------------------------------------
    # BLUEBEAR
    # ------------------------------------------------------------------
    # Preprocessing is intentionally skipped on Bluebear because P01-P03
    # require interactive inspection/cleaning that is performed on the Mac.
    #
    # Bluebear therefore assumes cleaned epochs already exist.
    # ------------------------------------------------------------------

    if args.platform == "bluebear":

        print("\n" + "!" * 78)
        print("BLUEBEAR MODE")
        print("!" * 78)

        print(
            "\nPreprocessing will NOT be run on Bluebear.\n\n"
            "The following steps are being skipped:\n"
            "  P01 - BIDS conversion\n"
            "  P02 - stimulation segmentation\n"
            "  P03 - epoching and manual cleaning\n\n"
            "Bluebear assumes that preprocessing has already been completed\n"
            "and that cleaned epoch files are available in the BIDS derivatives.\n"
        )

        # --------------------------------------------------------------
        # Check that the required cleaned epochs actually exist
        # --------------------------------------------------------------

        base = (
            f"sub-{subject}"
            f"_ses-{args.session}"
            f"_task-{args.task}"
            f"_run-{args.run}"
            "_eeg"
        )

        subject_deriv_folder = (
            Path(bids_root)
            / "derivatives"
            / f"sub-{subject}"
        )

        required_epochs = [
            subject_deriv_folder / f"{base}_no-stim_epo-cue.fif",
            subject_deriv_folder / f"{base}_stim_epo-cue.fif",
        ]

        missing_epochs = [
            fname
            for fname in required_epochs
            if not fname.exists()
        ]

        if missing_epochs:

            missing_text = "\n".join(
                f"  {fname}"
                for fname in missing_epochs
            )

            raise FileNotFoundError(
                "\nBluebear cannot start the post-preprocessing analysis "
                f"for {key}.\n\n"
                "The following cleaned epoch files are missing:\n"
                f"{missing_text}\n\n"
                "Run preprocessing for this participant on the Mac first, "
                "then make sure the resulting derivatives are available "
                "on Bluebear."
            )

        print(
            f"Cleaned epochs found for {key}.\n"
            "Starting post-preprocessing analyses."
        )

        # --------------------------------------------------------------
        # ERP
        # --------------------------------------------------------------

        print(
            f"\n[{key}] BLUEBEAR 1/2 ERP: "
            f"{SCRIPT_ORDER[3].name}"
        )

        _run_script(
            SCRIPT_ORDER[3],
            subject,
            project_root,
            data_root,
            bids_root,
            args.session,
            args.task,
            args.run,
        )

        # --------------------------------------------------------------
        # TFR
        # --------------------------------------------------------------

        print(
            f"\n[{key}] BLUEBEAR 2/2 TFR: "
            f"{SCRIPT_ORDER[4].name}"
        )

        _run_script(
            SCRIPT_ORDER[4],
            subject,
            project_root,
            data_root,
            bids_root,
            args.session,
            args.task,
            args.run,
        )

        print(
            f"\nFINISHED {key} "
            "(Bluebear post-preprocessing analysis)"
        )

        return

    # ------------------------------------------------------------------
    # MAC
    # ------------------------------------------------------------------
    # The complete preprocessing + sensor-level pipeline runs here.
    # ------------------------------------------------------------------

    print(
        "\nMAC MODE\n"
        "Running the complete preprocessing and sensor-level pipeline."
    )

    # --------------------------------------------------------------
    # P01
    # --------------------------------------------------------------

    p01 = SCRIPT_ORDER[0]

    brainvision_basename = _find_brainvision_basename(
        subject,
        data_root,
        args.session,
    )
    print(
        f"\n[{key}] 1/5 BIDS conversion: "
        f"{p01.name}"
    )

    print(
        f"BrainVision basename: {brainvision_basename}"
    )
    _run_script(
        p01,
        subject,
        project_root,
        data_root,
        bids_root,
        args.session,
        args.task,
        args.run,
        brainvision_basename=brainvision_basename,
    )

    # --------------------------------------------------------------
    # Stimulation crop times
    # --------------------------------------------------------------

    crop_times = _get_or_collect_crop_times(
        subject,
        project_root,
        args.session,
        args.task,
        args.run,
    )

    # --------------------------------------------------------------
    # P02
    # --------------------------------------------------------------

    print(
        f"\n[{key}] 2/5 stimulation segmentation: "
        f"{SCRIPT_ORDER[1].name}"
    )

    _run_script(
        SCRIPT_ORDER[1],
        subject,
        project_root,
        data_root,
        bids_root,
        args.session,
        args.task,
        args.run,
        crop_times=crop_times,
    )

    # --------------------------------------------------------------
    # P03
    # --------------------------------------------------------------

    print(
        f"\n[{key}] 3/5 epoching: {SCRIPT_ORDER[2].name}\n"
        "The epoch browser is interactive.\n"
        "Reject bad trials using the available posterior channels,\n"
        "then close the browser to save the cleaned epochs and continue."
    )

    _run_script(
        SCRIPT_ORDER[2],
        subject,
        project_root,
        data_root,
        bids_root,
        args.session,
        args.task,
        args.run,
    )

    # --------------------------------------------------------------
    # ERP
    # --------------------------------------------------------------

    print(
        f"\n[{key}] 4/5 ERP: "
        f"{SCRIPT_ORDER[3].name}"
    )

    _run_script(
        SCRIPT_ORDER[3],
        subject,
        project_root,
        data_root,
        bids_root,
        args.session,
        args.task,
        args.run,
    )

    # --------------------------------------------------------------
    # TFR
    # --------------------------------------------------------------

    print(
        f"\n[{key}] 5/5 TFR: "
        f"{SCRIPT_ORDER[4].name}"
    )

    _run_script(
        SCRIPT_ORDER[4],
        subject,
        project_root,
        data_root,
        bids_root,
        args.session,
        args.task,
        args.run,
    )

    print(
        f"\nFINISHED {key} "
        "(complete Mac pipeline)"
    )


def main() -> None:
    args = _parse_args()
    paths = _choose_platform()
    args.platform = paths["platform"]
    args.project_root = paths["project_root"]
    args.data_root = paths["data_root"]
    args.bids_root = paths["bids_root"]

    subjects = _subjects_from_args(args)

    print("\n" + "=" * 78)
    print("PIPELINE CONFIGURATION")
    print("=" * 78)

    print("Subjects to run: " + ", ".join(f"sub-{s}" for s in subjects))
    print(f"Platform:       {args.platform}")
    print(f"Project root: {Path(args.project_root).expanduser().resolve()}")
    print(f"Data root:      {args.data_root}")
    print(f"BIDS root:      {args.bids_root}")
    print(f"Repository root: {REPO_ROOT}")
    print(f"Crop-time table: {CROP_TABLE_PATH}")

    if args.platform == "bluebear":
        print("\n" + "!" * 78)
        print("WARNING: BLUEBEAR POST-PREPROCESSING MODE")
        print("!" * 78)
        print(
            "\nP01, P02 and P03 will NOT be run.\n"
            "No BIDS conversion, stimulation segmentation, channel cleaning,\n"
            "or manual epoch rejection will occur on Bluebear.\n\n"
            "The runner will start from the cleaned epoch files and run:\n"
            "  A01 - ERP\n"
            "  A02 - TFR\n\n"
            "Use the Mac version of this runner for preprocessing."
        )
        print("!" * 78 + "\n")
    else:
        print(
            "\nMac selected: the complete P01 -> P02 -> P03 -> ERP -> TFR "
            "pipeline will run.\n"
        )

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
