#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G03: participant-level 8-posterior-sensor TFR quality-control report.

This is a QC/visual-inspection analysis. There is NO averaging across participants.
Each requested participant receives a separate PDF report.

Sensors:
    PO3, POz, PO4, O1, Oz, O2, PO7, PO8

For every participant the report contains:
    1. Stim OFF TFRs: available 8 posterior sensors + their within-participant mean
    2. Stim ON TFRs: available 8 posterior sensors + their within-participant mean
    3. Normalized difference: (Stim ON - Stim OFF) / (Stim ON + Stim OFF)
       for every available posterior sensor + their within-participant mean

Only channels available and good in BOTH conditions are used. Missing channels are
reported and are not interpolated. The posterior mean is an arithmetic sensor mean
within that participant only.

TFR settings match the group grand-average analysis:
    2-31.5 Hz in 0.5-Hz steps
    multitaper
    n_cycles = frequency / 2
    time-bandwidth = 2
    decimation = 2
    FFT = True
    ITC = False
    average trials = True

Stim ON/OFF displays use percent baseline correction (-0.3 to -0.1 s).
The normalized difference is calculated from original unbaselined power and receives
no baseline correction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parents[1]
SUBJECT_ALL_DIR = ANALYSIS_DIR / "subject" / "EEG_all_channels"
UTILS_DIR = ANALYSIS_DIR / "utils"
for path in (SUBJECT_ALL_DIR, UTILS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pipeline_config import CONDITIONS, resolve_project_root, stage_path  # noqa: E402
from pdf_report import ParticipantPDF  # noqa: E402

POSTERIOR = ("PO3", "POz", "PO4", "O1", "Oz", "O2", "PO7", "PO8")
BASELINE = (-0.3, -0.1)
FREQS = np.arange(2.0, 32.0, 0.5)
N_CYCLES = FREQS / 2.0
TIME_BANDWIDTH = 2.0
DECIM = 2
PLOT_TMIN = -0.5
PLOT_TMAX = 1.5
ROBUST_PERCENTILE = 98.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True,
                        help="One or more subjects, e.g. --subjects 115 116 118")
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def load_epochs(root, subject, args):
    out = {}
    for condition in CONDITIONS:
        fname = stage_path(root, subject, args.session, args.task, args.run,
                           condition, "clean", "epo")
        if not fname.exists():
            raise FileNotFoundError(
                f"Missing final cleaned epochs for sub-{subject} {condition}: {fname}"
            )
        epochs = mne.read_epochs(fname, preload=True)
        cue_events = [name for name in ("cue_onset_right", "cue_onset_left")
                      if name in epochs.event_id]
        out[condition] = epochs[cue_events] if cue_events else epochs
    return out


def common_posterior(epochs):
    stim = epochs["stim"]
    nostim = epochs["no-stim"]
    stim_eeg = stim.copy().pick("eeg").ch_names
    return [ch for ch in POSTERIOR
            if ch in stim_eeg
            and ch in nostim.ch_names
            and ch not in stim.info["bads"]
            and ch not in nostim.info["bads"]]


def compute_tfr(epochs, picks, n_jobs):
    return epochs.copy().pick(picks).compute_tfr(
        method="multitaper",
        freqs=FREQS,
        n_cycles=N_CYCLES,
        time_bandwidth=TIME_BANDWIDTH,
        use_fft=True,
        zero_mean=True,
        return_itc=False,
        average=True,
        decim=DECIM,
        n_jobs=n_jobs,
    )


def normalized_difference(stim_raw, nostim_raw):
    ratio = stim_raw.copy()
    ratio.data = ((stim_raw.data - nostim_raw.data) /
                  (stim_raw.data + nostim_raw.data + np.finfo(float).eps))
    return ratio


def sensor_mean(tfr, channels):
    roi = tfr.copy().pick(channels)
    roi.data = roi.data.mean(axis=0, keepdims=True)
    roi.info = mne.pick_info(roi.info, [0], copy=True)
    roi.info["chs"][0]["ch_name"] = "Posterior_8_mean"
    roi.info["ch_names"][0] = "Posterior_8_mean"
    return roi


def robust_vlim(tfr):
    ti = (tfr.times >= PLOT_TMIN) & (tfr.times <= PLOT_TMAX)
    fi = (tfr.freqs >= 2.0) & (tfr.freqs <= 31.5)
    x = np.asarray(tfr.data)[:, fi][:, :, ti]
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return (None, None)
    vmax = float(np.percentile(np.abs(finite), ROBUST_PERCENTILE))
    if not np.isfinite(vmax) or vmax <= 0:
        return (None, None)
    return (-vmax, vmax)


def plot_channels(tfr, available, title, vlim):
    """Fixed 2x4 panel so the same electrode always occupies the same panel."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    axes = axes.ravel()
    for ax, ch in zip(axes, POSTERIOR):
        if ch not in available:
            ax.axis("off")
            ax.set_title(f"{ch}: unavailable")
            continue
        kwargs = dict(
            picks=[ch], tmin=PLOT_TMIN, tmax=PLOT_TMAX,
            fmin=2.0, fmax=31.5, baseline=None, mode=None,
            axes=ax, show=False, colorbar=True, cmap="RdBu_r",
        )
        if None not in vlim:
            kwargs["vlim"] = vlim
        tfr.plot(**kwargs)
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_title(ch)
    fig.suptitle(title, fontsize=15)
    return fig


def plot_mean(roi, title, vlim):
    kwargs = dict(
        picks=["Posterior_8_mean"], tmin=PLOT_TMIN, tmax=PLOT_TMAX,
        fmin=2.0, fmax=31.5, baseline=None, mode=None,
        show=False, colorbar=True, cmap="RdBu_r",
    )
    if None not in vlim:
        kwargs["vlim"] = vlim
    fig = roi.plot(**kwargs)
    fig = fig[0] if isinstance(fig, list) else fig
    fig.axes[0].axvline(0, color="k", linestyle="--", linewidth=0.8)
    fig.axes[0].set_title(title)
    return fig


def add_result(report, fig_dir, subject, result_tfr, available, title, section, caption):
    # One scale per result, calculated from all eight sensor data. The same scale is
    # reused for the individual-channel figure and its within-participant ROI mean.
    vlim = robust_vlim(result_tfr)
    roi = sensor_mean(result_tfr, available)
    scale = (f"Robust symmetric scale: {vlim[0]:.4g} to {vlim[1]:.4g}."
             if None not in vlim else "Automatic color scale.")
    report.add_figure(
        plot_channels(result_tfr, available, f"sub-{subject}: {title}", vlim),
        str(fig_dir / f"sub-{subject}_{section.replace(' ', '_')}_eight_sensors.png"),
        f"sub-{subject}: {title} - eight posterior sensors",
        caption + " " + scale +
        " Missing/rejected posterior sensors are shown as unavailable and are not interpolated.",
        section,
    )
    report.add_figure(
        plot_mean(roi, f"sub-{subject}: {title} - posterior mean", vlim),
        str(fig_dir / f"sub-{subject}_{section.replace(' ', '_')}_posterior_mean.png"),
        f"sub-{subject}: {title} - posterior sensor mean",
        caption + " " + scale +
        f" Arithmetic within-participant mean across available sensors: {', '.join(available)}.",
        section,
    )


def build_subject_report(root, subject, args):
    epochs = load_epochs(root, subject, args)
    available = common_posterior(epochs)
    missing = [ch for ch in POSTERIOR if ch not in available]
    if not available:
        raise RuntimeError(f"sub-{subject}: none of the requested posterior sensors are available.")

    report_root = (root / "derivatives" / "reports" / "QC" /
                   "eight_posterior_TFR" / f"sub-{subject}")
    fig_dir = report_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report = ParticipantPDF(str(report_root), f"{subject}_eight_posterior_TFR_QC")

    report.add_text(
        "Posterior sensor availability",
        f"Requested: {', '.join(POSTERIOR)}\n"
        f"Available in both conditions: {', '.join(available)}\n"
        f"Unavailable/rejected: {', '.join(missing) if missing else 'none'}\n"
        f"Stim OFF cue epochs: {len(epochs['no-stim'])}\n"
        f"Stim ON cue epochs: {len(epochs['stim'])}",
        "QC summary",
    )
    report.add_text(
        "QC analysis details",
        "This report is participant-level quality control only; there is no averaging across participants.\n"
        "Input: final cleaned cue-locked all-channel epochs; cue-left/right trials combined within stimulation condition.\n"
        "TFR: multitaper, 2-31.5 Hz in 0.5-Hz steps, n_cycles=frequency/2, time-bandwidth=2, FFT=True, zero_mean=True, ITC=False, trial-average=True, decimation=2.\n"
        "Stim OFF and Stim ON displays: percent baseline -0.3 to -0.1 s.\n"
        "Normalized difference: (Stim ON - Stim OFF)/(Stim ON + Stim OFF), computed from original unbaselined power with no baseline correction.\n"
        "Posterior mean: arithmetic mean of the requested posterior sensors that are available in BOTH conditions for this participant. No missing channel is interpolated.\n"
        "The same robust color scale is used for the eight-sensor panel and posterior-mean plot within each result.",
        "QC summary",
    )

    raw = {
        condition: compute_tfr(epochs[condition], available, args.n_jobs)
        for condition in CONDITIONS
    }
    display_nostim = raw["no-stim"].copy().apply_baseline(BASELINE, mode="percent")
    display_stim = raw["stim"].copy().apply_baseline(BASELINE, mode="percent")
    ratio = normalized_difference(raw["stim"], raw["no-stim"])

    add_result(
        report, fig_dir, subject, display_nostim, available,
        "Stim OFF", "1 Stim OFF",
        "Percent power change relative to the -0.3 to -0.1 s baseline.",
    )
    add_result(
        report, fig_dir, subject, display_stim, available,
        "Stim ON", "2 Stim ON",
        "Percent power change relative to the -0.3 to -0.1 s baseline.",
    )
    add_result(
        report, fig_dir, subject, ratio, available,
        "(Stim ON - Stim OFF) / (Stim ON + Stim OFF)",
        "3 Normalized difference",
        "Normalized difference calculated from original unbaselined power; no baseline correction.",
    )

    print(f"sub-{subject}: QC report complete: {report.pdf_fname}")


def main():
    args = parse_args()
    root = resolve_project_root(args.platform, args.project_root)
    subjects = [s.removeprefix("sub-") for s in args.subjects]
    for subject in subjects:
        build_subject_report(root, subject, args)


if __name__ == "__main__":
    main()
