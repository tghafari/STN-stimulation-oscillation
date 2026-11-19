#!/usr/bin/env python3
"""
===================================================================
Percent-baseline-corrected DBS OFF peak-aligned alpha curves by sensor.

Load final cleaned OFF cue epochs and individual OFF-derived alpha peak
frequencies from individual_alpha_peaks.csv. For each participant, compute
trial-averaged multitaper TFR power and apply percent baseline correction
(-0.3 to -0.1 s) separately for each sensor and frequency, before averaging
the peak +/-2 Hz band. Then grand-average participants per sensor equally.

The CSV peaks are NOT re-estimated here; only the alpha-power time courses
and their 0.3-0.8 and 0.1-0.7 s window means are baseline corrected.
No stimulation ON data or across-sensor ROI averaging is used.

Outputs: derivatives/reports/group/EEG_all_channels_OFF_peak_alpha/
====================================================================
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent.parent
for module_dir in (ANALYSIS_DIR / "subject" / "EEG_all_channels",):
    sys.path.insert(0, str(module_dir))
from pipeline_config import resolve_project_root, stage_path

FREQS = np.arange(2.0, 32.0, 0.5)
TIME_RANGE = (0.0, 1.2)
WINDOWS = {"alpha_0p3_0p8": (0.3, 0.8), "alpha_0p1_0p7": (0.1, 0.7)}
PEAK_HALF_WIDTH = 2.0
BASELINE = (-0.3, -0.1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--peaks-csv", type=Path, default=None)
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    args.subjects = [s.removeprefix("sub-") for s in args.subjects]
    if len(set(args.subjects)) != len(args.subjects):
        parser.error("Duplicate subjects")
    return args


def load_peaks(path, subjects):
    frame = pd.read_csv(path, dtype={"subject": str})
    if not {"subject", "peak_hz"}.issubset(frame.columns):
        raise ValueError("Peaks CSV must have subject and peak_hz columns")
    frame["subject"] = frame["subject"].str.removeprefix("sub-")
    if frame.subject.duplicated().any():
        raise ValueError("Duplicate subjects in peaks CSV")
    if "status" in frame.columns:
        frame = frame.loc[frame.status.astype(str).str.lower() == "ok"]
    peaks = dict(zip(frame.subject, pd.to_numeric(frame.peak_hz, errors="coerce")))
    for sub in subjects:
        if sub not in peaks or not np.isfinite(peaks[sub]) or not 8 <= peaks[sub] <= 12:
            raise ValueError(f"Missing or invalid individual OFF peak: sub-{sub}")
    return peaks


def participant_curves(epochs, peak, n_jobs):
    """Return good-channel percent-baseline alpha curves, averaging trials first."""
    eeg = epochs.copy().pick("eeg")
    good = [ch for ch in eeg.ch_names if ch not in eeg.info["bads"]]
    if not good:
        return {}, None
    tfr = eeg.pick(good).compute_tfr(
        method="multitaper", freqs=FREQS, n_cycles=FREQS / 2,
        time_bandwidth=2.0, use_fft=True, zero_mean=True,
        return_itc=False, average=True, decim=2, n_jobs=n_jobs,
        verbose=False,
    )
    freq_mask = ((tfr.freqs >= peak - PEAK_HALF_WIDTH - 1e-9)
                 & (tfr.freqs <= peak + PEAK_HALF_WIDTH + 1e-9))
    time_mask = ((tfr.times >= TIME_RANGE[0])
                 & (tfr.times <= TIME_RANGE[1]))
    if not freq_mask.any() or not time_mask.any():
        raise ValueError("Requested alpha frequencies or times absent")
    # Baseline each participant/sensor/frequency BEFORE frequency averaging.
    # MNE percent mode returns fractions, so multiply by 100 for % units.
    tfr.apply_baseline(BASELINE, mode="percent")
    # Shape: sensors x frequency x time -> sensors x time.
    alpha = 100.0 * tfr.data[:, freq_mask, :][:, :, time_mask].mean(axis=1)
    if not np.isfinite(alpha).all():
        raise ValueError("Nonfinite baseline-corrected alpha power")
    return dict(zip(tfr.ch_names, alpha)), tfr.times[time_mask]


def add_curve_pages(pdf, curves, times, title):
    """Paginate sensors and show grand mean with between-subject SEM."""
    names = sorted(curves)
    for start in range(0, len(names), 12):
        subset = names[start:start + 12]
        fig, axes = plt.subplots(4, 3, figsize=(13, 12), sharex=True)
        for ax, ch in zip(axes.flat, subset):
            data = np.stack(curves[ch])
            mean = data.mean(axis=0)
            sem = data.std(axis=0, ddof=1) / np.sqrt(len(data)) if len(data) > 1 else np.zeros_like(mean)
            ax.plot(times, mean, color="tab:blue")
            ax.fill_between(times, mean - sem, mean + sem,
                            color="tab:blue", alpha=0.2)
            ax.axvspan(0.3, 0.8, color="tab:orange", alpha=0.12)
            ax.axvspan(0.1, 0.7, color="tab:green", alpha=0.08)
            ax.set_title(f"{ch} (n={len(data)})", fontsize=10)
            ax.set_ylabel("Power change from baseline (%)")
            ax.set_xlim(TIME_RANGE)
        for ax in axes.flat[len(subset):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("Time from cue (s)")
        fig.suptitle(title + " | DBS OFF | individual peak ±2 Hz | percent baseline -0.3 to -0.1 s")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig)
        plt.close(fig)


def main():
    args = parse_args()
    root = Path(resolve_project_root(args.platform, args.project_root))
    peaks_csv = args.peaks_csv or (
        root / "derivatives" / "reports" / "group" / "individual_alpha_peaks.csv"
    )
    peaks = load_peaks(peaks_csv, args.subjects)
    output = root / "derivatives" / "reports" / "group" / "EEG_all_channels_OFF_peak_alpha"
    output.mkdir(parents=True, exist_ok=True)

    curves = {}
    membership = {}
    times = None
    for subject in args.subjects:
        path = stage_path(root, subject, args.session, args.task, args.run,
                          "no-stim", "clean", "epo")
        epochs = mne.read_epochs(path, preload=True, verbose=False)
        cue_ids = [key for key in ("cue_onset_right", "cue_onset_left")
                   if key in epochs.event_id]
        if not cue_ids:
            raise ValueError(f"sub-{subject}: no cue-onset event IDs")
        epochs = epochs[cue_ids]
        data, participant_times = participant_curves(
            epochs, float(peaks[subject]), args.n_jobs
        )
        if participant_times is None:
            print(f"sub-{subject}: no good EEG channels; skipped")
            continue
        if times is None:
            times = participant_times
        elif not np.allclose(times, participant_times, atol=1e-9):
            raise ValueError(f"sub-{subject}: incompatible TFR time grid")
        for ch, values in data.items():
            curves.setdefault(ch, []).append(values)
            membership.setdefault(ch, []).append(subject)
        print(f"sub-{subject}: peak={peaks[subject]:g} Hz; "
              f"trials={len(epochs)}; sensors={len(data)}")

    if not curves:
        raise ValueError("No usable EEG sensor data")
    rows = []
    for ch, arrays in curves.items():
        mean_curve = np.mean(np.stack(arrays), axis=0)
        row = {"sensor": ch, "n_subjects": len(arrays),
               "subjects": ";".join("sub-" + x for x in membership[ch])}
        for name, (start, end) in WINDOWS.items():
            mask = (times >= start) & (times <= end)
            if not mask.any():
                raise ValueError(f"No samples in {name}")
            row[name] = float(mean_curve[mask].mean())
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        "alpha_0p3_0p8", ascending=False
    ).reset_index(drop=True)
    table.to_csv(output / "alpha_power_per_sensor.csv", index=False)

    fig, ax = plt.subplots(figsize=(max(12, len(table) * 0.24), 6))
    positions = np.arange(len(table))
    width = 0.42
    ax.bar(positions - width / 2, table["alpha_0p3_0p8"], width,
           label="0.3–0.8 s")
    ax.bar(positions + width / 2, table["alpha_0p1_0p7"], width,
           label="0.1–0.7 s")
    ax.set_xticks(positions, table.sensor, rotation=75)
    ax.set_ylabel("Mean alpha power change from baseline (%)")
    ax.set_title("DBS OFF | peak ±2 Hz | percent-baseline alpha ranking")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "sensor_alpha_power_ranking.png", dpi=200)
    with PdfPages(output / "off_peak_alpha_sensor_report.pdf") as pdf:
        pdf.savefig(fig)
        add_curve_pages(pdf, curves, times, "Group mean ± SEM across participants")
    plt.close(fig)
    print("\nTop sensors by 0.3–0.8 s percent-baseline alpha change:")
    print(table[["sensor", "n_subjects", "alpha_0p3_0p8",
                 "alpha_0p1_0p7"]].head(15).to_string(index=False))
    print(f"Saved report and CSV to {output}")


if __name__ == "__main__":
    main()
