#!/usr/bin/env python3
"""Plot unbaselined TFR contrasts and baselined alpha curves by STN side.

Three PDF reports are produced, with one page per eligible participant:

RIGHT STN ON versus DBS OFF, and LEFT STN ON versus DBS OFF:
    1. ON: baseline-corrected alpha time courses on left versus right sensors.
    2. OFF: baseline-corrected alpha time courses on left versus right sensors.
    3. Left sensors: ON minus OFF absolute-power TFR.
    4. Right sensors: ON minus OFF absolute-power TFR.
    5. ON only: right minus left sensor absolute-power TFR.

RIGHT STN ON versus LEFT STN ON:
    1. Left sensors: right-STN ON minus left-STN ON TFR.
    2. Right sensors: right-STN ON minus left-STN ON TFR.
    3. Right sensors: baseline-corrected right/left STN alpha curves.
    4. Left sensors: baseline-corrected right/left STN alpha curves.

TFR contrasts are unbaselined linear power differences in microvolt squared.
Alpha curves use condition-specific percent baseline correction from
-0.3 to -0.1 s AFTER averaging power over trials.

Use B03's validated trial_alpha.csv to select the exact cleaned epochs
associated with the two confirmed ON stimulation periods and both OFF
periods. Reject missing, duplicate or inconsistent trial mappings rather
than guessing a laterality assignment. Both cue directions are pooled.

Sensor groups: LEFT (PO3, PO7, O1); RIGHT (PO4, PO8, O2).
POz and Oz belong to the parent eight-sensor posterior ROI but are
midline, so they are excluded from lateralised sensor comparisons.
Use only good channels shared by all three stimulation conditions.

Alpha frequency is each participant's precomputed OFF-derived peak
(peak_hz column) from individual_alpha_peaks.csv, with a +/-1 Hz band.
No alpha peak is estimated from the current stimulation contrasts.

Run from the repository root:
    python analysis/group/EEG_posterior_channels/stim_laterality/\
run_stim_laterality_tfr.py --subjects 101 102 103

The script outputs only PDF reports. No TFR baseline correction,
interpolation, inferential tests or statistical exclusions are performed.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
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

LEFT_SENSORS = ("PO3", "PO7", "O1")
RIGHT_SENSORS = ("PO4", "PO8", "O2")
FREQUENCIES = np.arange(2.0, 32.0, 0.5)
PLOT_TIME = (-0.5, 1.5)
ALPHA_HALF_WIDTH_HZ = 1.0
ALPHA_BASELINE = (-0.3, -0.1)
STIM_LABELS = {"right": "stim_right", "left": "stim_left"}
POWER_SCALE = 1e12  # V squared to microvolt squared.


def parse_args():
    """Read participant IDs, BIDS settings and optional input/output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument(
        "--platform", choices=("mac", "bluebear"), default="mac"
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--trial-csv", type=Path)
    parser.add_argument("--peaks-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    args.subjects = [
        str(subject).removeprefix("sub-") for subject in args.subjects
    ]
    if len(args.subjects) != len(set(args.subjects)):
        parser.error("Participant identifiers must not be duplicated.")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive.")
    return args


def locate_peaks_csv(root, explicit_path):
    """Find the existing subject alpha-peak CSV without re-estimating peaks."""
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Alpha peaks CSV not found: {path}")
        return path

    group = root / "derivatives" / "reports" / "group"
    candidates = (
        group / "individual_alpha_peaks.csv",
        group / "EEG_posterior_channels_alpha_changes_stimOn_stimOff"
        / "individual_alpha_peaks.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No individual_alpha_peaks.csv was found. "
        "Provide its exact location with --peaks-csv."
    )


def load_alpha_peaks(path):
    """Read unique, finite participant peaks from the existing CSV."""
    frame = pd.read_csv(path, dtype={"subject": str})
    if not {"subject", "peak_hz"}.issubset(frame.columns):
        raise ValueError("Alpha CSV needs subject and peak_hz columns.")
    frame["subject"] = frame["subject"].str.removeprefix("sub-")
    if frame["subject"].duplicated().any():
        raise ValueError("Duplicate participant in alpha-peak CSV.")
    frame["peak_hz"] = pd.to_numeric(frame["peak_hz"], errors="coerce")
    return frame.set_index("subject")["peak_hz"].to_dict()


def load_cue_epochs(root, subject, condition, args):
    """Reproduce B03's cue filtering to preserve its saved epoch indices."""
    path = stage_path(
        root, subject, args.session, args.task, args.run,
        condition, "clean", "epo",
    )
    epochs = mne.read_epochs(path, preload=True, verbose=False)
    cues = [
        cue for cue in ("cue_onset_left", "cue_onset_right")
        if cue in epochs.event_id
    ]
    if not cues:
        raise ValueError(f"sub-{subject}: no cue events in {path}")
    return epochs[cues]


def verified_rows(trials, subject, epochs):
    """Require a complete, uniquely indexed mapping for both epoch files."""
    rows = trials.loc[trials["subject"] == subject].copy()
    if rows.empty:
        raise ValueError("No B03 trial mapping.")
    if not rows["label_verified"].astype(str).str.lower().eq("true").all():
        raise ValueError("Stimulation sequence is not fully confirmed.")

    for condition in ("no-stim", "stim"):
        subset = rows.loc[rows["source_condition"] == condition]
        indices = subset["trial_index_in_condition"].to_numpy(dtype=int)
        expected = np.arange(len(epochs[condition]))
        if not np.array_equal(np.sort(indices), expected):
            raise ValueError(
                f"Incomplete or duplicated {condition} mapping; rerun B03."
            )
        flags = subset["dbs_on"].astype(str).str.lower()
        if not flags.eq(str(condition == "stim").lower()).all():
            raise ValueError("Inconsistent DBS ON/OFF labels.")

    grouped = rows.groupby("stimulation_period")["stimulation_side"].unique()
    if len(grouped) != 4 or any(len(values) != 1 for values in grouped):
        raise ValueError("Expected four internally consistent periods.")
    labels = [values[0] for values in grouped.values]
    if sorted(labels) != sorted(
        ("off", "off", "stim_left", "stim_right")
    ):
        raise ValueError("Missing a confirmed stimulation period.")
    return rows


def extract_conditions(epochs, rows):
    """Separate both OFF periods and each single-side ON period."""
    conditions = {}
    for label, source in (
        ("off", "no-stim"),
        ("stim_left", "stim"),
        ("stim_right", "stim"),
    ):
        subset = rows.loc[rows["stimulation_side"] == label]
        indices = np.sort(
            subset["trial_index_in_condition"].to_numpy(dtype=int)
        )
        if not len(indices):
            raise ValueError(f"No retained trials for {label}.")
        conditions[label] = epochs[source][indices]
    return conditions


def shared_sensors(conditions):
    """Exclude channels absent or bad in any of the three conditions."""
    common = {}
    for hemisphere, candidates in (
        ("left", LEFT_SENSORS),
        ("right", RIGHT_SENSORS),
    ):
        common[hemisphere] = [
            channel for channel in candidates
            if all(
                channel in epochs.ch_names
                and channel not in epochs.info["bads"]
                for epochs in conditions.values()
            )
        ]
        if not common[hemisphere]:
            raise ValueError(f"No common good {hemisphere} sensors.")
    return common


def multitaper_power(epochs, channels, n_jobs):
    """Average linear multitaper power over trials; never baseline."""
    tfr = epochs.copy().pick(channels).compute_tfr(
        method="multitaper",
        freqs=FREQUENCIES,
        n_cycles=FREQUENCIES / 2.0,
        time_bandwidth=2.0,
        use_fft=True,
        zero_mean=True,
        return_itc=False,
        average=True,
        decim=2,
        n_jobs=n_jobs,
        verbose=False,
    )
    return tfr.times, tfr.data * POWER_SCALE


def participant_power(conditions, channels, n_jobs):
    """Compute time-matched left/right sensor-average TFRs.

    Every condition is calculated once for all selected lateral sensors.
    Sensor averaging occurs after trial-averaged TFR computation.
    """
    picks = channels["left"] + channels["right"]
    maps = {}
    times = None
    for condition, epochs in conditions.items():
        condition_times, data = multitaper_power(epochs, picks, n_jobs)
        if times is None:
            times = condition_times
        elif not np.array_equal(times, condition_times):
            raise ValueError("TFR time axes differ between conditions.")
        for hemi, selected in channels.items():
            indices = [picks.index(channel) for channel in selected]
            maps[(condition, hemi)] = data[indices].mean(axis=0)
    return times, maps


def alpha_curve(power, peak_hz, times):
    """Baseline trial-averaged power per Hz, then average alpha Hz.

    Power is already averaged over trials and sensors for this condition.
    Percent baseline uses the same condition at -0.3 to -0.1 seconds.
    TFR difference panels remain entirely unbaselined.
    """
    frequencies = (
        (FREQUENCIES >= peak_hz - ALPHA_HALF_WIDTH_HZ - 1e-9)
        & (FREQUENCIES <= peak_hz + ALPHA_HALF_WIDTH_HZ + 1e-9)
    )
    if not np.any(frequencies):
        raise ValueError(f"No frequency bins at peak {peak_hz:g} Hz.")
    baseline = (
        (times >= ALPHA_BASELINE[0])
        & (times <= ALPHA_BASELINE[1])
    )
    if not baseline.any():
        raise ValueError("No samples in the alpha baseline interval.")
    reference = power[:, baseline].mean(axis=1, keepdims=True)
    if not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError("Invalid condition-specific baseline power.")
    corrected = 100.0 * (power / reference - 1.0)
    return corrected[frequencies].mean(axis=0)


def add_curve(ax, times, curves, peak_hz, title):
    """Show condition-specific baseline-corrected alpha curves."""
    visible = (times >= PLOT_TIME[0]) & (times <= PLOT_TIME[1])
    for label, values, colour in curves:
        ax.plot(
            times[visible], values[visible],
            label=label, color=colour, linewidth=1.8,
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvspan(0.3, 0.8, alpha=0.08, color="gray")
    ax.set(
        title=title,
        xlabel="Time after cue (s)",
        ylabel="Alpha change from baseline (%)",
        xlim=PLOT_TIME,
    )
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(alpha=0.16)
    ax.text(
        0.02, 0.94, f"Peak {peak_hz:g} Hz +/-1 Hz",
        transform=ax.transAxes, va="top", fontsize=9,
    )


def colour_limit(maps, times):
    """Choose a shared symmetric scale for a set of absolute contrasts."""
    visible = (times >= PLOT_TIME[0]) & (times <= PLOT_TIME[1])
    values = np.concatenate([
        np.abs(matrix[:, visible]).ravel() for matrix in maps
    ])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite TFR values.")
    return max(float(np.nanpercentile(values, 99)), 1e-12)


def add_tfr(ax, times, values, title, limit):
    """Plot an unbaselined difference TFR with a zero-centred colourmap."""
    visible = (times >= PLOT_TIME[0]) & (times <= PLOT_TIME[1])
    image = ax.pcolormesh(
        times[visible], FREQUENCIES, values[:, visible],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        shading="auto",
        rasterized=True,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set(
        title=title,
        xlabel="Time after cue (s)",
        ylabel="Frequency (Hz)",
        ylim=(2.0, 31.5),
    )
    return image


def plot_on_off_page(subject, side, times, maps, channels, peak, counts):
    """Two sensor-comparison curves and three unbaselined TFR panels."""
    on_label = STIM_LABELS[side]
    left_diff = maps[(on_label, "left")] - maps[("off", "left")]
    right_diff = maps[(on_label, "right")] - maps[("off", "right")]
    on_right_left = maps[(on_label, "right")] - maps[(on_label, "left")]
    limit = colour_limit(
        [left_diff, right_diff, on_right_left], times
    )

    fig = plt.figure(figsize=(14, 13), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(0.8, 1, 1.05))
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, :]),
    ]
    add_curve(
        axes[0], times, [
            ("Left sensors",
             alpha_curve(maps[(on_label, "left")], peak, times),
             "tab:blue"),
            ("Right sensors",
             alpha_curve(maps[(on_label, "right")], peak, times),
             "tab:orange"),
        ], peak, "DBS ON: left versus right sensors",
    )
    add_curve(
        axes[1], times, [
            ("Left sensors", alpha_curve(maps[("off", "left")], peak, times),
             "tab:blue"),
            ("Right sensors", alpha_curve(maps[("off", "right")], peak, times),
             "tab:orange"),
        ], peak, "DBS OFF: left versus right sensors",
    )
    for ax, data, title in zip(
        axes[2:],
        (left_diff, right_diff, on_right_left),
        (
            "Left sensors: ON - OFF",
            "Right sensors: ON - OFF",
            "ON only: right sensors - left sensors",
        ),
    ):
        image = add_tfr(ax, times, data, title, limit)
        fig.colorbar(
            image, ax=ax, label="Power difference (microvolt squared)",
            shrink=0.85, pad=0.01,
        )
    fig.suptitle(
        f"sub-{subject} | {side.upper()} STN ON vs OFF | "
        f"ON {counts[on_label]}, OFF {counts['off']} trials | "
        f"peak {peak:g} Hz",
        fontsize=16,
    )
    fig.text(
        0.5, 0.004,
        f"Left: {', '.join(channels['left'])} | "
        f"Right: {', '.join(channels['right'])} | "
        "Alpha curves: baseline-corrected. TFRs: unbaselined.",
        ha="center", fontsize=9,
    )
    return fig


def plot_stim_comparison_page(subject, times, maps, channels, peak, counts):
    """Two right-minus-left STN TFRs and two alpha-condition curves."""
    left_diff = (
        maps[("stim_right", "left")] - maps[("stim_left", "left")]
    )
    right_diff = (
        maps[("stim_right", "right")] - maps[("stim_left", "right")]
    )
    limit = colour_limit([left_diff, right_diff], times)
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), layout="constrained",
    )
    for ax, values, title in zip(
        axes[0],
        (left_diff, right_diff),
        (
            "Left sensors: right STN ON - left STN ON",
            "Right sensors: right STN ON - left STN ON",
        ),
    ):
        image = add_tfr(ax, times, values, title, limit)
        fig.colorbar(
            image, ax=ax, label="Power difference (microvolt squared)",
            shrink=0.85, pad=0.01,
        )

    for ax, hemisphere, title in (
        (
            axes[1, 0], "right",
            "Right sensors: right STN ON vs left STN ON",
        ),
        (
            axes[1, 1], "left",
            "Left sensors: right STN ON vs left STN ON",
        ),
    ):
        add_curve(
            ax, times, [
                (
                    "Right STN ON",
                    alpha_curve(maps[("stim_right", hemisphere)], peak, times),
                    "tab:orange",
                ),
                (
                    "Left STN ON",
                    alpha_curve(maps[("stim_left", hemisphere)], peak, times),
                    "tab:blue",
                ),
            ], peak, title,
        )
    fig.suptitle(
        f"sub-{subject} | RIGHT vs LEFT STN ON | "
        f"right {counts['stim_right']}, left {counts['stim_left']} trials | "
        f"peak {peak:g} Hz",
        fontsize=16,
    )
    fig.text(
        0.5, 0.003,
        f"Left: {', '.join(channels['left'])} | "
        f"Right: {', '.join(channels['right'])} | "
        "Alpha curves: baseline-corrected. TFRs: unbaselined.",
        ha="center", fontsize=9,
    )
    return fig


def methods_page(report_key):
    """Create a dedicated readable methodology page for each PDF."""
    shared = [
        ("Trial selection",
         "Read the validated B03 trial_alpha.csv and cleaned cue epochs. "
         "Check that every cleaned epoch has exactly one validated BIDS "
         "assignment. Use confirmed stimulation labels to select periods. "
         "Both cue directions are pooled; no trial is reassigned."),
        ("Sensor selection",
         "Left sensors: PO3, PO7, O1. Right sensors: PO4, PO8, O2. "
         "Midline POz and Oz are excluded. Only sensors present and "
         "marked good in all three conditions are retained. "
         "The actual sensors and trial counts appear on each subject page."),
        ("Power and individual alpha frequency",
         "Compute multitaper power separately for each condition using "
         "2-31.5 Hz in 0.5-Hz steps, n_cycles=frequency/2, "
         "time_bandwidth=2 and decim=2. Average power across trials "
         "FIRST and then across retained sensors. Read each participant's "
         "OFF-derived peak from individual_alpha_peaks.csv; use peak "
         "+/-1 Hz for the alpha curves."),
        ("Alpha baseline correction (curves only)",
         "For each condition and sensor group independently, calculate "
         "the mean trial-averaged power from -0.3 to -0.1 seconds at "
         "each frequency. Convert the trial-averaged power to percent "
         "change: 100 * (power / baseline - 1). Only then average "
         "frequencies within the peak +/-1 Hz band to obtain the curve. "
         "No baseline correction is applied to any TFR panel."),
        ("Plot conventions",
         "Cue onset is time zero; plots cover -0.5 to 1.5 seconds. "
         "The shaded 0.3-0.8-second interval on alpha curves marks "
         "attention. Alpha curves use percent baseline change; "
         "TFR contrasts use linear power differences in microvolt squared. "
         "All figures are descriptive, without statistical testing."),
    ]
    if report_key == "comparison":
        details = (
            "Panels 1 and 2 are unbaselined right-STN-ON minus "
            "left-STN-ON TFRs on left and right sensors, respectively. "
            "Panel 3 shows baseline-corrected alpha curves for right "
            "sensors under right-STN ON and left-STN ON. Panel 4 "
            "shows the same comparison for left sensors. Both ON "
            "periods are analysed separately; OFF is not used in "
            "these comparisons."
        )
        title = "RIGHT versus LEFT STN ON: analysis methods"
    else:
        side = report_key.upper()
        details = (
            f"This report uses only {side} STN ON trials for its ON "
            "condition and pools both DBS OFF periods as its OFF "
            "reference. Panel 1 shows baseline-corrected alpha curves "
            "for left versus right sensors during ON. Panel 2 shows "
            "the same sensor comparison during OFF. Panels 3 and 4 "
            "are unbaselined ON minus OFF TFRs on left and right "
            "sensors, respectively. Panel 5 is unbaselined right "
            "minus left sensor power during ON only."
        )
        title = f"{side} STN ON versus OFF: analysis methods"
    sections = [("What each participant page contains", details)] + shared
    fig, ax = plt.subplots(figsize=(11.7, 10.5))
    ax.axis("off")
    ax.text(
        0.055, 0.965, title, fontsize=16, weight="bold",
        va="top", transform=ax.transAxes,
    )
    y = 0.90
    for heading, paragraph in sections:
        ax.text(
            0.055, y, heading, fontsize=11.5, weight="bold",
            va="top", transform=ax.transAxes,
        )
        y -= 0.031
        wrapped = textwrap.fill(paragraph, width=103)
        ax.text(
            0.055, y, wrapped, fontsize=9.4, linespacing=1.4,
            va="top", transform=ax.transAxes,
        )
        y -= 0.019 * (wrapped.count("
") + 1) + 0.041
    return fig


def main():
    """Create three PDF reports from verified participant-specific epochs."""
    args = parse_args()
    root = resolve_project_root(args.platform, args.project_root)
    group = root / "derivatives" / "reports" / "group"
    trial_path = args.trial_csv or (
        group / "EEG_posterior_channels_block_mapping" / "trial_alpha.csv"
    )
    peak_path = locate_peaks_csv(root, args.peaks_csv)
    output = args.output_dir or (
        group / "EEG_posterior_channels_stim_laterality"
    )
    output.mkdir(parents=True, exist_ok=True)

    trials = pd.read_csv(trial_path, dtype={"subject": str})
    required = {
        "subject", "source_condition", "trial_index_in_condition",
        "label_verified", "dbs_on", "stimulation_period",
        "stimulation_side",
    }
    if not required.issubset(trials.columns):
        raise ValueError(
            f"Missing trial_alpha columns: {required - set(trials.columns)}"
        )
    trials["subject"] = trials["subject"].str.removeprefix("sub-")
    peaks = load_alpha_peaks(peak_path)

    names = {
        "right": "posterior_TFR_right_STN_ON.pdf",
        "left": "posterior_TFR_left_STN_ON.pdf",
        "comparison": "posterior_TFR_right_vs_left_STN_ON.pdf",
    }
    pdfs = {
        key: PdfPages(output / filename)
        for key, filename in names.items()
    }
    pages = {key: 0 for key in names}
    try:
        for report_key, pdf in pdfs.items():
            page = methods_page(report_key)
            pdf.savefig(page)
            plt.close(page)
        for subject in args.subjects:
            try:
                peak = peaks.get(subject)
                if peak is None or not np.isfinite(peak):
                    raise ValueError(
                        "Missing/invalid individual OFF-derived alpha peak."
                    )
                epochs = {
                    condition: load_cue_epochs(
                        root, subject, condition, args
                    )
                    for condition in ("no-stim", "stim")
                }
                rows = verified_rows(trials, subject, epochs)
                conditions = extract_conditions(epochs, rows)
                channels = shared_sensors(conditions)
                times, maps = participant_power(
                    conditions, channels, args.n_jobs
                )
                counts = {
                    name: len(value) for name, value in conditions.items()
                }
                for side in STIM_LABELS:
                    fig = plot_on_off_page(
                        subject, side, times, maps, channels, peak, counts
                    )
                    pdfs[side].savefig(fig)
                    plt.close(fig)
                    pages[side] += 1

                fig = plot_stim_comparison_page(
                    subject, times, maps, channels, peak, counts
                )
                pdfs["comparison"].savefig(fig)
                plt.close(fig)
                pages["comparison"] += 1
                print(f"sub-{subject}: saved all three report pages.")
            except (
                FileNotFoundError, ValueError, RuntimeError, KeyError
            ) as exc:
                print(f"sub-{subject}: skipped - {exc}")
    finally:
        for pdf in pdfs.values():
            pdf.close()
    for name, count in pages.items():
        print(f"{names[name]}: {count} participant pages")
    print(f"Report directory: {output}")


if __name__ == "__main__":
    main()
