#!/usr/bin/env python
"""Open raw BIDS EEG and manually replace stimulation crop times (no analysis).

The shared file is analysis/subject/stimulation_cropped_time.json, used by
both the posterior and all-channel EEG preprocessing pipelines.

Usage:
 python review_raw_stimulation_crops.py --subjects 102 104 105 107 110 116 120 122 123

For each participant, inspect the raw EEG browser, close it, enter one or two
intervals per condition, and explicitly confirm before saving. A timestamped
backup is made before the first change. Existing unrelated subjects are kept.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import mne
from mne_bids import read_raw_bids

HERE = Path(__file__).resolve().parent
SUBJECT_DIR = HERE.parent
CONFIG_DIR = SUBJECT_DIR / "EEG_all_channels"
sys.path.insert(0, str(CONFIG_DIR))
from pipeline_config import base_bids_path, crop_table_path, resolve_project_root


def parse_times(value):
    parts = [p for p in re.split(r"[\s,;]+", value.strip()) if p]
    if len(parts) not in (2, 4):
        raise ValueError("Enter 2 numbers (start end) or 4 (start1 end1 start2 end2).")
    try:
        vals = [float(x) for x in parts]
    except ValueError as exc:
        raise ValueError("All time points must be numeric.") from exc
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("Times must be finite.")
    if any(a >= b for a, b in zip(vals[::2], vals[1::2])):
        raise ValueError("Every interval must have start < end.")
    if len(vals) == 4 and vals[1] >= vals[2]:
        raise ValueError("Intervals must be chronological and must not overlap.")
    return vals


def ask_times(label, duration):
    while True:
        answer = input(f"{label} intervals in seconds (2 or 4 values): ").strip()
        try:
            vals = parse_times(answer)
            if vals[0] < 0 or vals[-1] > duration + 1e-3:
                raise ValueError(f"Intervals must be within 0 to {duration:.3f} seconds.")
            return vals
        except ValueError as exc:
            print(f"Invalid crop times: {exc}")


def ranges(values):
    return list(zip(values[::2], values[1::2]))


def validate_conditions(off, stim):
    for a, b in ranges(off):
        for c, d in ranges(stim):
            if max(a, c) < min(b, d):
                raise ValueError(
                    f"STIM and NO-STIM intervals overlap: [{a}, {b}] and [{c}, {d}]."
                )


def load_table(path):
    if not path.exists():
        raise FileNotFoundError(f"Shared crop table not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        table = json.load(f)
    if not isinstance(table, dict):
        raise ValueError("Shared crop table must be a JSON object.")
    return table


def atomic_write(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()

    root = resolve_project_root(args.platform, args.project_root)
    table_path = crop_table_path()
    print(f"\nShared crop JSON: {table_path}")
    print("Only crop time definitions are edited. No preprocessing or analysis is run.")
    print("All entered times are seconds from the start of the BIDS raw recording.")
    backup_path = None

    for raw_subject in args.subjects:
        subject = str(raw_subject).removeprefix("sub-")
        key = f"sub-{subject}"
        print(f"\n{'=' * 65}\n{key}")
        try:
            table = load_table(table_path)
            old = table.get(key)
            print(f"Previously saved intervals: {json.dumps(old, ensure_ascii=False)}")
            bids_path = base_bids_path(root, subject, args.session, args.task, args.run)
            raw = read_raw_bids(
                bids_path=bids_path, verbose=False, extra_params={"preload": False}
            )
            duration = (raw.n_times - 1) / float(raw.info["sfreq"])
            print(f"Raw BIDS recording duration: {duration:.3f} seconds")
            print("Inspect block markers and EEG in the browser, then CLOSE the browser.")
            raw.plot(
                block=True, title=f"{key} — define STIM / NO-STIM crop times",
                duration=30.0, n_channels=min(30, len(raw.ch_names)),
            )
            print("Enter boundaries based on the raw BIDS timeline.")
            off = ask_times("NO-STIM", duration)
            stim = ask_times("STIM", duration)
            validate_conditions(off, stim)
            print(f"\n{key} proposed replacement:")
            print(f"  NO-STIM: {off}\n  STIM:    {stim}")
            choice = input("Overwrite this subject in the shared JSON? [y/N]: ").strip().lower()
            if choice not in ("y", "yes"):
                print(f"{key}: no changes saved.")
                continue
            if backup_path is None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = table_path.with_name(table_path.stem + f"_backup_{stamp}.json")
                shutil.copy2(table_path, backup_path)
                print(f"Original JSON backed up at: {backup_path}")
            # Reload immediately before writing to preserve any other subjects' entries.
            table = load_table(table_path)
            table[key] = {"no-stim": off, "stim": stim}
            atomic_write(table_path, table)
            print(f"{key}: updated {table_path}")
        except KeyboardInterrupt:
            print("\nStopped by user. Previously confirmed subjects remain saved.")
            break
        except Exception as exc:
            print(f"{key}: not saved — {type(exc).__name__}: {exc}")
            answer = input("Continue to the next subject? [Y/n]: ").strip().lower()
            if answer in ("n", "no"):
                break

    print(f"\nFinished. Shared JSON: {table_path}")
    if backup_path is not None:
        print(f"Original backup: {backup_path}")
    print("NOTE: Existing cleaned epoch files are not modified or regenerated by this script.")


if __name__ == "__main__":
    main()
