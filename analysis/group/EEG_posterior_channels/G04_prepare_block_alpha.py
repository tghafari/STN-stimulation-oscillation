#!/usr/bin/env python3
"""Reconstruct and audit posterior EEG trial-to-stimulation-block assignments.

This is a reusable NON-STATISTICAL preparation step. It preserves the
chronological task block, four stimulation periods, stimulation laterality,
cue direction, absolute posterior alpha power and the source of the labels.

Stimulation labels come from the wiki-derived JSON and are PROPOSED, not
independently confirmed. Original BIDS block-onset/end events determine the
actual boundaries. The validated BIDS-to-cleaned-Epochs mapping must match
every retained cue epoch uniquely or processing stops.

Output: trial_alpha.csv, block_summary.csv and mapping_audit.csv.
These outputs can be reused for ON/OFF, left/right and cue-side analyses.

Run from repository root:
python analysis/group/EEG_posterior_channels/G04_prepare_block_alpha.py \\
    --subjects 101 102 103
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parents[1]
SUBJECT_DIR = ANALYSIS / "subject"
for folder in (
    SUBJECT_DIR / "EEG_all_channels",
    SUBJECT_DIR / "EEG_STN_laterality",
):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from pipeline_config import resolve_project_root, stage_path
from A00_review_stimulation_blocks import events_path, get_blocks
from A02_validate_bids_epoch_mapping import map_condition_epochs

# Edit these constants for subsequent analyses.
ATTENTION_WINDOW = (0.3, 0.8)
ALPHA_PEAK_RANGE = 1.0
FREQS = np.arange(2.0, 32.0, 0.5)
ROI8 = ("PO3", "POz", "PO4", "O1", "Oz", "O2", "PO7", "PO8")
SEQUENCE_FILE = SUBJECT_DIR / "stimulation_sequence_reference.json"
CROP_FILE = SUBJECT_DIR / "stimulation_cropped_time.json"


def parse_args():
    """Read subjects and file locations; Mac is the default platform."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--platform", choices=("mac", "bluebear"), default="mac")
    parser.add_argument("--project-root")
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--peaks-csv", type=Path)
    return parser.parse_args()


def crop_ranges(values):
    """Convert two/four crop endpoints into one/two time intervals."""
    numbers = [float(value) for value in values]
    if len(numbers) not in (2, 4):
        raise ValueError(f"Expected 2 or 4 crop endpoints, got {numbers}")
    return list(zip(numbers[::2], numbers[1::2]))


def load_epochs(root, subject, condition, args):
    """Read cleaned cue epochs, retaining only valid cue directions."""
    path = stage_path(
        root, subject, args.session, args.task, args.run,
        condition, "clean", "epo"
    )
    epochs = mne.read_epochs(path, preload=True, verbose=False)
    cues = [
        name for name in ("cue_onset_left", "cue_onset_right")
        if name in epochs.event_id
    ]
    if not cues:
        raise ValueError(f"sub-{subject}: no cue events in {path}")
    return epochs[cues]


def absolute_alpha(epochs, channels, peak, n_jobs):
    """Average linear multitaper power over channels, Hz and attention time.

    Return one value per trial; NO baseline correction or log transform.
    """
    tfr = epochs.copy().pick(channels).compute_tfr(
        method="multitaper",
        freqs=FREQS,
        n_cycles=FREQS / 2.0,
        time_bandwidth=2.0,
        use_fft=True,
        zero_mean=True,
        return_itc=False,
        average=False,
        decim=2,
        n_jobs=n_jobs,
        verbose=False,
    )
    frequency_mask = (
        (tfr.freqs >= peak - ALPHA_PEAK_RANGE - 1e-9)
        & (tfr.freqs <= peak + ALPHA_PEAK_RANGE + 1e-9)
    )
    time_mask = (
        (tfr.times >= ATTENTION_WINDOW[0])
        & (tfr.times <= ATTENTION_WINDOW[1])
    )
    if not frequency_mask.any() or not time_mask.any():
        raise ValueError("Requested frequency or attention window is empty")
    return tfr.data[:, :, frequency_mask, :][
        :, :, :, time_mask
    ].mean(axis=(1, 2, 3))


def prepare_subject(root, subject, peak, sequence, crops, args):
    """Build validated trial rows and audit records for one participant."""
    labels = sequence.get(f"sub-{subject}")
    if not labels or len(labels) != 4:
        raise ValueError(f"sub-{subject}: no four-period sequence in JSON")
    if labels == ["exclude"] * 4 or "exclude" in labels:
        raise ValueError(f"sub-{subject}: excluded in stimulation sequence")
    if sorted(labels) != sorted(("off", "off", "stim_left", "stim_right")):
        raise ValueError(f"sub-{subject}: invalid proposed labels {labels}")

    bids = pd.read_csv(
        events_path(root, subject, args.session, args.task, args.run),
        sep="\t",
    )
    blocks = get_blocks(bids)
    if len(blocks) != 8:
        raise ValueError(
            f"sub-{subject}: expected eight BIDS task blocks, got {len(blocks)}"
        )
    intervals = [
        (float(block["onset_s"]), float(block["offset_s"]))
        for block in blocks
    ]
    key = f"sub-{subject}"
    if key not in crops:
        raise ValueError(f"sub-{subject}: missing crop-time record")

    epochs = {
        condition: load_epochs(root, subject, condition, args)
        for condition in ("no-stim", "stim")
    }
    channels = [
        channel for channel in ROI8
        if all(
            channel in ep.ch_names and channel not in ep.info["bads"]
            for ep in epochs.values()
        )
    ]
    if not channels:
        raise ValueError(f"sub-{subject}: no shared good posterior channels")

    trial_rows = []
    audits = []
    for condition, ep in epochs.items():
        mapping = map_condition_epochs(
            ep, bids, crop_ranges(crops[key][condition]),
            intervals, condition,
        )
        values = absolute_alpha(ep, channels, peak, args.n_jobs)
        indices = mapping["matched_task_block_indices"]
        on_label = condition == "stim"
        for index, (task_index, power) in enumerate(zip(indices, values)):
            period = int(task_index) // 2
            stimulation = labels[period]
            if (stimulation != "off") != on_label:
                raise ValueError(
                    f"sub-{subject}: wiki condition disagrees with "
                    f"cleaned epochs in period {period + 1}"
                )
            event_code = int(ep.events[index, 2])
            cue = next(
                name for name, code in ep.event_id.items()
                if code == event_code
            )
            trial_rows.append({
                "subject": subject,
                "trial_index_in_condition": index,
                "source_condition": condition,
                "task_block": int(task_index) + 1,
                "stimulation_period": period + 1,
                "stimulation_side": stimulation,
                "dbs_on": stimulation != "off",
                "cue_direction": "left" if cue.endswith("left") else "right",
                "bids_cue_onset_s": mapping["matched_bids_onsets_s"][index],
                "block_start_s": intervals[task_index][0],
                "block_end_s": intervals[task_index][1],
                "alpha_peak_hz": peak,
                "alpha_power": float(power),
                "n_posterior_channels": len(channels),
                "channels": ";".join(channels),
                "label_source": "wiki_proposed_json",
                "label_verified": False,
            })
        audits.append({
            "subject": subject,
            "source_condition": condition,
            "clean_epochs": len(ep),
            "mapped_epochs": len(indices),
            "sample_offset": mapping["sample_offset"],
            "unretained_bids_cues": mapping["n_unretained_bids_cues"],
            "label_source": "wiki_proposed_json",
            "label_verified": False,
        })
    if len(trial_rows) != sum(map(len, epochs.values())):
        raise ValueError(f"sub-{subject}: incomplete trial mapping")
    return trial_rows, audits


def main():
    """Prepare reusable tables; fail rather than silently mislabel trials."""
    args = parse_args()
    root = resolve_project_root(args.platform, args.project_root)
    peaks_path = args.peaks_csv or (
        root / "derivatives" / "reports" / "group"
        / "EEG_posterior_channels_alpha_changes_stimOn_stimOff"
        / "individual_alpha_peaks.csv"
    )
    peaks = pd.read_csv(peaks_path, dtype={"subject": str})
    peaks["subject"] = peaks["subject"].str.removeprefix("sub-")
    if peaks["subject"].duplicated().any():
        raise ValueError("Duplicate subject in alpha-peak CSV")
    peaks = peaks.set_index("subject")
    sequence = json.loads(SEQUENCE_FILE.read_text(encoding="utf-8"))
    crops = json.loads(CROP_FILE.read_text(encoding="utf-8"))

    trials = []
    audits = []
    for value in args.subjects:
        subject = value.removeprefix("sub-")
        if subject not in peaks.index:
            raise ValueError(f"sub-{subject}: missing individual alpha peak")
        peak = float(peaks.loc[subject, "peak_hz"])
        if not np.isfinite(peak):
            raise ValueError(f"sub-{subject}: invalid alpha peak")
        trial_rows, audit_rows = prepare_subject(
            root, subject, peak, sequence, crops, args
        )
        trials.extend(trial_rows)
        audits.extend(audit_rows)
        print(f"sub-{subject}: mapped {len(trial_rows)} cleaned trials")

    out = (
        root / "derivatives" / "reports" / "group"
        / "EEG_posterior_channels_block_mapping"
    )
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(trials)
    frame.to_csv(out / "trial_alpha.csv", index=False)
    summary = frame.groupby(
        ["subject", "stimulation_period", "task_block",
         "stimulation_side", "dbs_on"],
        as_index=False,
    ).agg(
        n_trials=("alpha_power", "size"),
        mean_absolute_alpha=("alpha_power", "mean"),
        block_start_s=("block_start_s", "first"),
        block_end_s=("block_end_s", "first"),
    )
    summary.to_csv(out / "block_summary.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "mapping_audit.csv", index=False)
    print(f"Saved reusable mapping and power tables: {out}")
    print("WARNING: stimulation laterality comes from PROPOSED wiki labels.")


if __name__ == "__main__":
    main()
