#!/usr/bin/env python3
"""Participant posterior TFR QC: stimulation OFF/ON and individual alpha peak.

Run from repository root:
    python analysis/group/EEG_posterior_channels/G03_individual_alpha_report.py \
        --subjects 101 102 103

For each participant, estimate multitaper power averaged over trials FIRST,
then apply percent baseline correction separately to ON and OFF. Average the
available eight posterior sensors. Choose a single alpha peak from unbaselined
OFF power (7-14 Hz; 0.3-0.8 s), and plot peak +/- 1 Hz for both conditions.
Each participant gets one PDF page: two TFRs and one alpha time course.
No inferential statistics or across-participant averaging are performed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parents[1]
for folder in (ANALYSIS_DIR / "subject" / "EEG_all_channels",):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from pipeline_config import resolve_project_root, stage_path

ROI8 = ("PO3", "POz", "PO4", "O1", "Oz", "O2", "PO7", "PO8")
FREQS = np.arange(2.0, 32.0, 0.5)
BASELINE = (-0.3, -0.1)
PEAK_SEARCH = (7.0, 14.0)
PEAK_TIME = (0.3, 0.8)
PLOT_WINDOW = (-0.5, 1.5)


def parse_args():
    """Parse participant identifiers and dataset configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument("--platform", choices=("mac", "bluebear"), default="mac")
    parser.add_argument("--project-root")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--alpha-peak-half-width", type=float, default=1.0)
    args = parser.parse_args()
    if not np.isfinite(args.alpha_peak_half_width) or args.alpha_peak_half_width <= 0:
        parser.error("--alpha-peak-half-width must be positive.")
    return args


def load_epochs(root, subject, args):
    """Load cleaned cue epochs, pooling left and right cue events."""
    result = {}
    for condition in ("no-stim", "stim"):
        path = stage_path(
            root, subject, args.session, args.task, args.run,
            condition, "clean", "epo"
        )
        epochs = mne.read_epochs(path, preload=True, verbose=False)
        cues = [
            cue for cue in ("cue_onset_right", "cue_onset_left")
            if cue in epochs.event_id
        ]
        result[condition] = epochs[cues] if cues else epochs
    return result


def retained_channels(epochs):
    """Use posterior sensors present and not marked bad in BOTH conditions."""
    off = epochs["no-stim"]
    on = epochs["stim"]
    return [
        ch for ch in ROI8
        if ch in off.ch_names and ch in on.ch_names
        and ch not in off.info["bads"] and ch not in on.info["bads"]
    ]


def averaged_tfr(epochs, channels, n_jobs):
    """Compute trial-averaged (not evoked) multitaper power."""
    return epochs.copy().pick(channels).compute_tfr(
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


def alpha_peak(off_tfr):
    """Select peak from raw OFF power, avoiding ON-dependent peak selection."""
    frequencies = off_tfr.freqs
    frequency_mask = (
        (frequencies >= PEAK_SEARCH[0])
        & (frequencies <= PEAK_SEARCH[1])
    )
    time_mask = (
        (off_tfr.times >= PEAK_TIME[0])
        & (off_tfr.times <= PEAK_TIME[1])
    )
    spectrum = off_tfr.data[:, :, time_mask].mean(axis=(0, 2))
    candidates = np.flatnonzero(frequency_mask)
    return float(frequencies[candidates[np.argmax(spectrum[candidates])]])


def roi_percent_tfr(raw_tfr):
    """Baseline trial-averaged power, then average retained ROI sensors."""
    corrected = raw_tfr.copy()
    corrected.apply_baseline(BASELINE, mode="percent")
    return corrected.data.mean(axis=0) * 100.0


def plot_participant(subject, raw, channels, counts, half_width):
    """Create one neatly arranged page with two TFRs and one alpha curve."""
    peak = alpha_peak(raw["no-stim"])
    band = (
        (FREQS >= peak - half_width - 1e-9)
        & (FREQS <= peak + half_width + 1e-9)
    )
    if not np.any(band):
        raise ValueError(f"No frequency bins selected for sub-{subject}.")

    off = roi_percent_tfr(raw["no-stim"])
    on = roi_percent_tfr(raw["stim"])
    times = raw["no-stim"].times
    if not np.array_equal(times, raw["stim"].times):
        raise ValueError(f"ON/OFF time grids differ for sub-{subject}.")

    time_mask = (times >= PLOT_WINDOW[0]) & (times <= PLOT_WINDOW[1])
    values = np.concatenate([off[:, time_mask].ravel(), on[:, time_mask].ravel()])
    limit = max(float(np.nanpercentile(np.abs(values), 98)), 1.0)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 0.85))
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    curve_ax = fig.add_subplot(grid[1, :])

    for ax, data, label in zip(
        axes, (off, on), ("DBS OFF", "DBS ON")
    ):
        image = ax.pcolormesh(
            times[time_mask], FREQS, data[:, time_mask],
            cmap="RdBu_r", norm=norm, shading="auto"
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
        ax.axhspan(peak - half_width, peak + half_width,
                   facecolor="none", edgecolor="black", linewidth=1)
        ax.set(title=label, xlabel="Time from cue (s)", ylabel="Frequency (Hz)")
        ax.set_ylim(2, 31.5)

    fig.colorbar(image, ax=axes, shrink=0.78, label="Power change from baseline (%)")

    for data, color, label in (
        (off, "tab:blue", "DBS OFF"),
        (on, "tab:orange", "DBS ON"),
    ):
        curve = data[band].mean(axis=0)
        curve_ax.plot(times[time_mask], curve[time_mask],
                      color=color, linewidth=2.0, label=label)
    curve_ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
    curve_ax.axhline(0, color="gray", linestyle=":", linewidth=0.9)
    curve_ax.set(
        title=f"Individual alpha peak: {peak:.1f} Hz +/- {half_width:g} Hz",
        xlabel="Time from cue (s)",
        ylabel="Alpha change from baseline (%)",
        xlim=PLOT_WINDOW,
    )
    curve_ax.legend(loc="best", frameon=False)
    curve_ax.grid(alpha=0.18)

    fig.suptitle(
        f"sub-{subject} | posterior ROI ({len(channels)}/8 channels) | "
        f"OFF {counts['no-stim']} trials, ON {counts['stim']} trials",
        fontsize=15,
    )
    fig.text(
        0.5, 0.012,
        "Power averaged across trials, then percent baseline corrected "
        "(-0.3 to -0.1 s); ROI averaged last. "
        "Alpha peak selected from unbaselined DBS OFF (0.3-0.8 s).",
        ha="center", fontsize=9, color="dimgray",
    )
    return fig, peak


def main():
    """Generate a multi-page PDF and a companion peak-frequency CSV."""
    args = parse_args()
    root = resolve_project_root(args.platform, args.project_root)
    output_dir = (
        root / "derivatives" / "reports" / "group"
        / "EEG_posterior_channels_alpha_changes_stimOn_stimOff"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = [subject.removeprefix("sub-") for subject in args.subjects]
    report_path = output_dir / ("posterior_alpha_ON_OFF_" + "_".join(subjects) + ".pdf")
    summary = ["subject,peak_hz,channels,n_off,n_on,status"]

    with PdfPages(report_path) as pdf:
        for subject in subjects:
            try:
                epochs = load_epochs(root, subject, args)
                channels = retained_channels(epochs)
                if not channels:
                    raise ValueError("No common retained posterior channels.")
                raw = {
                    condition: averaged_tfr(epochs[condition], channels, args.n_jobs)
                    for condition in ("no-stim", "stim")
                }
                counts = {key: len(value) for key, value in epochs.items()}
                fig, peak = plot_participant(
                    subject, raw, channels, counts, args.alpha_peak_half_width
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                summary.append(
                    f"{subject},{peak:.2f},{len(channels)},"
                    f"{counts['no-stim']},{counts['stim']},ok"
                )
                print(f"sub-{subject}: peak {peak:.1f} Hz; {len(channels)} channels")
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                print(f"Skipping sub-{subject}: {exc}")
                summary.append(f"{subject},,,,,skipped")
    (output_dir / "individual_alpha_peaks.csv").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
