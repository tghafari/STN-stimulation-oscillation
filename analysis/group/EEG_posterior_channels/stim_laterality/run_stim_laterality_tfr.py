#!/usr/bin/env python3
"""Participant-wise posterior TFR reports split by stimulated STN side.

Generate two PDFs, one for RIGHT STN ON and one for LEFT STN ON.
Each participant has one page with five cue-locked multitaper TFRs:

1. Left posterior ROI: ON minus OFF (each condition baseline corrected).
2. Right posterior ROI: ON minus OFF (each condition baseline corrected).
3. Left posterior ROI: unbaselined ON minus unbaselined OFF.
4. Right posterior ROI: unbaselined ON minus unbaselined OFF.
5. ON only: right posterior ROI minus left posterior ROI (unbaselined).

The ON epochs in each report belong exclusively to the confirmed
stim_right or stim_left stimulation period. OFF reference epochs come
from BOTH verified OFF periods for the same participant.

Trial allocation relies on B03's validated trial_alpha.csv, NOT inferred
epoch positions or raw time heuristics. The table's
trial_index_in_condition is an index into the cue-filtered cleaned
epoch object. A full mapping is required for each condition.

Lateral posterior channels: LEFT = PO3, PO7, O1;
RIGHT = PO4, PO8, O2. Midline POz and Oz cannot be assigned to a
hemisphere and are excluded from left/right sensor contrasts.
All eight channels remain the parent posterior ROI.

Baseline-corrected contrasts are differences of percent baseline
changes: 100*(ON/ON_baseline-1) - 100*(OFF/OFF_baseline-1).
Unbaselined contrasts subtract linear absolute power without correction.
Condition power is trial averaged first, then sensor averaged.

Run from repository root:
python analysis/group/EEG_posterior_channels/stim_laterality/run_stim_laterality_tfr.py \
    --subjects 101 102 103 105 106 107 108 112 113 114 115 116 118 120 121 123

Requirements: MNE, NumPy, pandas, Matplotlib, mne-bids.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parents[2]
CONFIG_DIR = ANALYSIS / "subject" / "EEG_all_channels"
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

from pipeline_config import resolve_project_root, stage_path

LEFT = ("PO3", "PO7", "O1")
RIGHT = ("PO4", "PO8", "O2")
MIDLINE = ("POz", "Oz")
FREQS = np.arange(2.0, 32.0, 0.5)
BASELINE = (-0.3, -0.1)
DISPLAY_TIME = (-0.5, 1.5)
SIDES = {"right": "stim_right", "left": "stim_left"}


def parse_args():
    """Parse dataset paths and computational settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--platform", choices=("mac", "bluebear"),
                        default="mac")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--trial-csv", type=Path, default=None,
                        help="B03's validated trial_alpha.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    args.subjects = [str(s).removeprefix("sub-") for s in args.subjects]
    if len(set(args.subjects)) != len(args.subjects):
        parser.error("Duplicate subject identifiers.")
    return args


def load_cue_epochs(root, subject, condition, args):
    """Read exactly the cue-filtered epochs indexed by B03."""
    path = stage_path(
        root, subject, args.session, args.task, args.run,
        condition, "clean", "epo",
    )
    epochs = mne.read_epochs(path, preload=True, verbose=False)
    cues = [
        key for key in ("cue_onset_left", "cue_onset_right")
        if key in epochs.event_id
    ]
    if not cues:
        raise ValueError("No cue events in " + str(path))
    return epochs[cues]


def verified_rows(frame, subject, epochs):
    """Verify that every cue epoch maps to exactly one confirmed row.

    An incomplete or outdated B03 table must never silently assign
    stimulation laterality to the wrong cleaned EEG epoch.
    """
    rows = frame.loc[frame["subject"] == subject].copy()
    if rows.empty:
        raise ValueError("No validated B03 rows for sub-" + subject)
    if not rows["label_verified"].astype(str).str.lower().eq("true").all():
        raise ValueError("Unconfirmed stimulation labels in B03 input")
    for condition in ("no-stim", "stim"):
        subset = rows.loc[rows["source_condition"] == condition]
        indices = subset["trial_index_in_condition"].to_numpy(dtype=int)
        expected = np.arange(len(epochs[condition]))
        if len(indices) != len(expected) or not np.array_equal(
            np.sort(indices), expected
        ):
            raise ValueError(
                f"sub-{subject}: B03 mapping does not cover every "
                f"{condition} cue epoch exactly once; rerun B03."
            )
        expected_on = condition == "stim"
        if not (
            subset["dbs_on"].astype(str).str.lower()
            == str(expected_on).lower()
        ).all():
            raise ValueError("Contradictory ON/OFF labels in B03 input")
    labels = rows.groupby("stimulation_period")["stimulation_side"].unique()
    if len(labels) != 4 or any(len(value) != 1 for value in labels):
        raise ValueError("Expected four consistently labelled periods")
    all_labels = [value[0] for value in labels.values]
    if sorted(all_labels) != sorted(
        ("off", "off", "stim_left", "stim_right")
    ):
        raise ValueError("Four stimulation periods have invalid labels")
    return rows


def choose_condition_epochs(epochs, rows, side):
    """Select side-specific ON epochs and both OFF stimulation periods."""
    off_rows = rows.loc[rows["stimulation_side"] == "off"]
    on_rows = rows.loc[rows["stimulation_side"] == side]
    if on_rows.empty or off_rows.empty:
        raise ValueError("No retained trials for " + side + " or OFF")
    off_indices = np.sort(
        off_rows["trial_index_in_condition"].to_numpy(dtype=int)
    )
    on_indices = np.sort(
        on_rows["trial_index_in_condition"].to_numpy(dtype=int)
    )
    return epochs["stim"][on_indices], epochs["no-stim"][off_indices]


def common_channels(on, off):
    """Use channels present and marked good in both compared conditions."""
    return {
        "left": [
            ch for ch in LEFT
            if ch in on.ch_names and ch in off.ch_names
            and ch not in on.info["bads"] and ch not in off.info["bads"]
        ],
        "right": [
            ch for ch in RIGHT
            if ch in on.ch_names and ch in off.ch_names
            and ch not in on.info["bads"] and ch not in off.info["bads"]
        ],
    }


def condition_tfr(epochs, channels, n_jobs):
    """Return linear multitaper power averaged over retained trials."""
    tfr = epochs.copy().pick(channels).compute_tfr(
        method="multitaper",
        freqs=FREQS,
        n_cycles=FREQS / 2.0,
        time_bandwidth=2.0,
        use_fft=True,
        zero_mean=True,
        return_itc=False,
        average=True,
        decim=2,
        n_jobs=n_jobs,
        verbose=False,
    )
    return tfr


def percent_from_baseline(power, times):
    """Compute percentage change per frequency and sensor."""
    mask = (times >= BASELINE[0]) & (times <= BASELINE[1])
    if not mask.any():
        raise ValueError("No samples in baseline interval")
    mean = power[:, :, mask].mean(axis=-1, keepdims=True)
    if (mean <= 0).any():
        raise ValueError("Baseline power must be positive")
    return (power / mean - 1.0) * 100.0


def calculate_maps(on, off, channels, n_jobs):
    """Compute four left/right maps from the two trial conditions."""
    maps = {}
    times = None
    for hemi, picks in channels.items():
        if not picks:
            raise ValueError(f"No shared good {hemi} posterior sensors")
        on_tfr = condition_tfr(on, picks, n_jobs)
        off_tfr = condition_tfr(off, picks, n_jobs)
        if not np.array_equal(on_tfr.times, off_tfr.times):
            raise ValueError("Stim ON/OFF TFR time grids differ")
        if times is None:
            times = on_tfr.times
        elif not np.array_equal(times, on_tfr.times):
            raise ValueError("Left/right TFR time grids differ")

        on_raw = on_tfr.data.mean(axis=0)
        off_raw = off_tfr.data.mean(axis=0)
        on_pct = percent_from_baseline(on_tfr.data, times).mean(axis=0)
        off_pct = percent_from_baseline(off_tfr.data, times).mean(axis=0)
        maps[(hemi, "baselined")] = on_pct - off_pct
        maps[(hemi, "raw_diff")] = (on_raw - off_raw) * 1e12
        maps[(hemi, "on_raw")] = on_raw * 1e12

    maps[("right_minus_left", "on_raw")] = (
        maps[("right", "on_raw")] - maps[("left", "on_raw")]
    )
    return maps, times


def add_tfr(ax, times, values, title, limit, unit):
    """Show a zero-centred diverging TFR with a consistent colour scale."""
    shown = (times >= DISPLAY_TIME[0]) & (times <= DISPLAY_TIME[1])
    image = ax.pcolormesh(
        times[shown], FREQS, values[:, shown],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
        shading="auto",
        rasterized=True,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set(
        title=title,
        xlabel="Time from cue onset (s)",
        ylabel="Frequency (Hz)",
    )
    ax.set_ylim(2, 31.5)
    return image


def symmetric_limit(maps, keys, times):
    """Use the 99th percentile of absolute displayed contrasts."""
    shown = (times >= DISPLAY_TIME[0]) & (times <= DISPLAY_TIME[1])
    values = np.concatenate(
        [np.abs(maps[key][:, shown]).ravel() for key in keys]
    )
    return max(float(np.nanpercentile(values, 99)), 1e-12)


def plot_page(subject, side, maps, times, channels, n_on, n_off):
    """Arrange two baselined, two raw and one ON lateralisation TFR."""
    fig = plt.figure(figsize=(14.5, 13), layout="constrained")
    grid = fig.add_gridspec(
        3, 2, height_ratios=(1, 1, 1),
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, :]),
    ]
    baseline_keys = [
        ("left", "baselined"), ("right", "baselined")
    ]
    raw_keys = [
        ("left", "raw_diff"), ("right", "raw_diff"),
        ("right_minus_left", "on_raw"),
    ]
    pct_limit = symmetric_limit(maps, baseline_keys, times)
    raw_limit = symmetric_limit(maps, raw_keys, times)
    specifications = [
        (baseline_keys[0], "Left sensors: ON - OFF (baseline %)",
         pct_limit, "Percentage points"),
        (baseline_keys[1], "Right sensors: ON - OFF (baseline %)",
         pct_limit, "Percentage points"),
        (raw_keys[0], "Left sensors: ON - OFF (absolute)",
         raw_limit, "Power difference (microvolt squared)"),
        (raw_keys[1], "Right sensors: ON - OFF (absolute)",
         raw_limit, "Power difference (microvolt squared)"),
        (raw_keys[2], "ON only: right sensors - left sensors",
         raw_limit, "Power difference (microvolt squared)"),
    ]

    for axis, (key, label, limit, unit) in zip(axes, specifications):
        image = add_tfr(axis, times, maps[key], label, limit, unit)
        fig.colorbar(image, ax=axis, label=unit, shrink=0.85, pad=0.01)

    fig.suptitle(
        f"sub-{subject} | {side.upper()} STN stimulation | "
        f"ON n={n_on}; OFF n={n_off}",
        fontsize=17,
    )
    fig.text(
        0.5, 0.005,
        f"Left: {', '.join(channels['left'])}  |  "
        f"Right: {', '.join(channels['right'])}  |  "
        "Midline POz/Oz excluded. "
        "Baseline -0.3 to -0.1 s; cue directions pooled.",
        ha="center", fontsize=9,
    )
    return fig


def main():
    """Generate both reports; write one page per successful participant."""
    args = parse_args()
    root = resolve_project_root(args.platform, args.project_root)
    trial_csv = args.trial_csv or (
        root / "derivatives" / "reports" / "group"
        / "EEG_posterior_channels_block_mapping" / "trial_alpha.csv"
    )
    output = args.output_dir or (
        root / "derivatives" / "reports" / "group"
        / "EEG_posterior_channels_stim_laterality"
    )
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(trial_csv, dtype={"subject": str})
    required = {
        "subject", "source_condition", "trial_index_in_condition",
        "label_verified", "dbs_on", "stimulation_period",
        "stimulation_side",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Missing B03 trial_alpha columns: {required - set(frame.columns)}"
        )
    frame["subject"] = frame["subject"].str.removeprefix("sub-")
    pdfs = {
        side: PdfPages(output / f"posterior_TFR_{side}_STN_ON.pdf")
        for side in SIDES
    }
    page_count = {side: 0 for side in SIDES}
    try:
        for subject in args.subjects:
            try:
                epochs = {
                    condition: load_cue_epochs(
                        root, subject, condition, args
                    )
                    for condition in ("no-stim", "stim")
                }
                rows = verified_rows(frame, subject, epochs)
                for side, stimulation in SIDES.items():
                    on, off = choose_condition_epochs(
                        epochs, rows, stimulation
                    )
                    channels = common_channels(on, off)
                    maps, times = calculate_maps(
                        on, off, channels, args.n_jobs
                    )
                    fig = plot_page(
                        subject, side, maps, times, channels,
                        len(on), len(off),
                    )
                    pdfs[side].savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    page_count[side] += 1
                print(f"sub-{subject}: completed LEFT and RIGHT STN PDFs")
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                print(f"sub-{subject}: skipped, {exc}")
    finally:
        for pdf in pdfs.values():
            pdf.close()
    for side, count in page_count.items():
        print(f"{side.upper()} STN PDF: {count} participant pages")


if __name__ == "__main__":
    main()
