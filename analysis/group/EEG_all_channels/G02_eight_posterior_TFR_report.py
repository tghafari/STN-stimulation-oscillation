#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G02: separate 8-posterior-sensor group TFR diagnostic report.

Posterior sensors: PO3, POz, PO4, O1, Oz, O2, PO7, PO8.

For each requested subject, final cleaned cue-locked epochs from the all-channel
preprocessing pipeline are used. Subject TFRs are computed first and then grand
averaged, so subjects contribute equally. For an individual sensor, only subjects
with that sensor available in BOTH stimulation conditions contribute.

The separate PDF contains three sections, in this order:
  1. stimulation OFF (no-stim): eight individual posterior sensors + posterior mean
  2. stimulation ON (stim): eight individual posterior sensors + posterior mean
  3. (stim ON - stim OFF) / (stim ON + stim OFF): eight sensors + posterior mean

The posterior mean is calculated WITHIN each subject first using that subject's
available members of the predefined 8-sensor ROI; subject ROI TFRs are then averaged.
Thus a participant with a missing posterior sensor can still contribute to the ROI,
without interpolating the missing channel.

TFR parameters match the all-channel group/posterior grand-average analysis:
2-31.5 Hz in 0.5-Hz steps; multitaper; n_cycles=f/2; time-bandwidth=2;
decim=2; FFT=True; ITC=False; trial-average=True. Stim/no-stim displays use
percent baseline correction from -0.3 to -0.1 s. The normalized difference is
calculated from original unbaselined power and receives no baseline correction.
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
                        help="Subjects, e.g. --subjects 115 116 118")
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def load_clean_epochs(root, subject, args):
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


def available_posterior(epoch_pair):
    stim = epoch_pair["stim"]
    nostim = epoch_pair["no-stim"]
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


def grand_channel_tfr(subject_tfrs, subjects_by_channel, channel, condition):
    tfrs = [subject_tfrs[s][condition].copy().pick([channel])
            for s in subjects_by_channel[channel]]
    if not tfrs:
        return None
    grand = tfrs[0].copy()
    grand.data = np.mean([x.data for x in tfrs], axis=0)
    grand.nave = len(tfrs)
    return grand


def make_contrast(stim, nostim):
    ratio = stim.copy()
    ratio.data = ((stim.data - nostim.data) /
                  (stim.data + nostim.data + np.finfo(float).eps))
    return ratio


def robust_vlim(tfrs):
    displayed = []
    for tfr in tfrs:
        if tfr is None:
            continue
        ti = (tfr.times >= PLOT_TMIN) & (tfr.times <= PLOT_TMAX)
        fi = (tfr.freqs >= 2.0) & (tfr.freqs <= 31.5)
        x = np.asarray(tfr.data)[:, fi][:, :, ti]
        x = x[np.isfinite(x)]
        if x.size:
            displayed.append(x)
    if not displayed:
        return (None, None)
    values = np.concatenate(displayed)
    vmax = float(np.percentile(np.abs(values), ROBUST_PERCENTILE))
    if not np.isfinite(vmax) or vmax <= 0:
        return (None, None)
    return (-vmax, vmax)


def plot_eight_channels(tfr_by_channel, title, vlim, subject_counts):
    """Eight TFRs in a fixed posterior anatomical-style 2 x 4 arrangement."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    axes = axes.ravel()
    for ax, ch in zip(axes, POSTERIOR):
        tfr = tfr_by_channel.get(ch)
        if tfr is None:
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
        ax.set_title(f"{ch} (n={subject_counts[ch]})")
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    fig.suptitle(title, fontsize=15)
    return fig


def subject_posterior_mean(subject_tfrs, subjects, result):
    """Average available ROI channels within subject, then average subjects."""
    roi_tfrs = []
    roi_subjects = []
    roi_channels = {}
    for subject in subjects:
        available = [ch for ch in POSTERIOR
                     if ch in subject_tfrs[subject]["stim"].ch_names
                     and ch in subject_tfrs[subject]["no-stim"].ch_names]
        if not available:
            continue
        stim = subject_tfrs[subject]["stim"].copy().pick(available)
        nostim = subject_tfrs[subject]["no-stim"].copy().pick(available)
        if result == "no-stim":
            x = nostim.copy().apply_baseline(BASELINE, mode="percent")
        elif result == "stim":
            x = stim.copy().apply_baseline(BASELINE, mode="percent")
        elif result == "ratio":
            x = make_contrast(stim, nostim)
        else:
            raise ValueError(result)
        x.data = x.data.mean(axis=0, keepdims=True)
        x.info = mne.pick_info(x.info, [0], copy=True)
        x.info["chs"][0]["ch_name"] = "Posterior_8_mean"
        x.info["ch_names"][0] = "Posterior_8_mean"
        roi_tfrs.append(x)
        roi_subjects.append(subject)
        roi_channels[subject] = available
    if not roi_tfrs:
        return None, [], {}
    grand = roi_tfrs[0].copy()
    grand.data = np.mean([x.data for x in roi_tfrs], axis=0)
    grand.nave = len(roi_tfrs)
    return grand, roi_subjects, roi_channels


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
    fig.axes[0].set_title(title)
    fig.axes[0].axvline(0, color="k", linestyle="--", linewidth=0.8)
    return fig


def add_result(report, fig_dir, subject_tfrs, subjects, subjects_by_channel,
               result, section, title, caption):
    channel_results = {}
    for ch in POSTERIOR:
        if not subjects_by_channel[ch]:
            channel_results[ch] = None
            continue
        if result in {"stim", "no-stim"}:
            raw = grand_channel_tfr(subject_tfrs, subjects_by_channel, ch, result)
            channel_results[ch] = raw.copy().apply_baseline(BASELINE, mode="percent")
        elif result == "ratio":
            stim = grand_channel_tfr(subject_tfrs, subjects_by_channel, ch, "stim")
            nostim = grand_channel_tfr(subject_tfrs, subjects_by_channel, ch, "no-stim")
            channel_results[ch] = make_contrast(stim, nostim)
        else:
            raise ValueError(result)

    roi, roi_subjects, roi_channels = subject_posterior_mean(
        subject_tfrs, subjects, result
    )
    scale_inputs = [x for x in channel_results.values() if x is not None]
    if roi is not None:
        scale_inputs.append(roi)
    vlim = robust_vlim(scale_inputs)
    scale_text = (f"Shared robust symmetric scale: {vlim[0]:.4g} to {vlim[1]:.4g}. "
                  if None not in vlim else "Automatic color scale. ")
    counts = {ch: len(subjects_by_channel[ch]) for ch in POSTERIOR}

    report.add_figure(
        plot_eight_channels(channel_results, title, vlim, counts),
        str(fig_dir / f"{result}_eight_posterior_channels.png"),
        f"{title}: eight posterior sensors",
        caption + " " + scale_text +
        "Each sensor includes only participants retaining that sensor in both conditions.",
        section,
    )

    if roi is not None:
        report.add_figure(
            plot_mean(roi, f"{title}: mean of 8-sensor posterior ROI", vlim),
            str(fig_dir / f"{result}_eight_posterior_mean.png"),
            f"{title}: posterior 8-sensor mean",
            caption + " " + scale_text +
            f"Posterior sensors were averaged within participant first, then across participants (n={len(roi_subjects)}).",
            section,
        )
        roi_text = []
        for subject in roi_subjects:
            roi_text.append(f"sub-{subject}: {', '.join(roi_channels[subject])}")
        report.add_text(
            "Sensors entering each participant's posterior mean",
            "\n".join(roi_text),
            section,
        )


def main():
    args = parse_args()
    subjects = [s.removeprefix("sub-") for s in args.subjects]
    root = resolve_project_root(args.platform, args.project_root)

    report_root = (root / "derivatives" / "reports" / "group" /
                   "EEG_eight_posterior_TFR_diagnostic")
    fig_dir = report_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_id = "eight_posterior_TFR_" + "_".join(subjects)
    report = ParticipantPDF(str(report_root), report_id)

    epochs = {s: load_clean_epochs(root, s, args) for s in subjects}
    good = {s: available_posterior(epochs[s]) for s in subjects}
    subjects_by_channel = {
        ch: [s for s in subjects if ch in good[s]] for ch in POSTERIOR
    }

    availability = []
    for ch in POSTERIOR:
        included = subjects_by_channel[ch]
        availability.append(
            f"{ch} (n={len(included)}): " +
            (", ".join(f"sub-{s}" for s in included) if included else "none")
        )
    report.add_text(
        "Subjects contributing to each posterior sensor",
        "\n".join(availability),
        "Data availability",
    )

    subject_tfrs = {}
    for subject in subjects:
        picks = good[subject]
        if not picks:
            raise RuntimeError(
                f"sub-{subject} has none of the requested posterior sensors available."
            )
        subject_tfrs[subject] = {}
        for condition in CONDITIONS:
            print(f"Computing sub-{subject} {condition}: {', '.join(picks)}")
            subject_tfrs[subject][condition] = compute_tfr(
                epochs[subject][condition], picks, args.n_jobs
            )

    report.add_text(
        "Analysis details",
        "Cue-locked final cleaned epochs only; attention-left and attention-right cue trials are combined within stimulation condition.\n"
        "Posterior ROI: PO3, POz, PO4, O1, Oz, O2, PO7, PO8.\n"
        "TFR: multitaper; 2-31.5 Hz in 0.5-Hz steps; n_cycles=frequency/2; time-bandwidth=2; FFT=True; zero_mean=True; ITC=False; trial-average=True; decimation=2.\n"
        "Stim OFF and Stim ON: percent baseline correction from -0.3 to -0.1 s.\n"
        "Normalized difference: (stim ON - stim OFF) / (stim ON + stim OFF), calculated from original unbaselined power; no baseline correction is applied to the ratio.\n"
        "For individual sensors, each subject is averaged first and only subjects retaining that sensor in both conditions contribute to its grand average.\n"
        "For the 8-sensor mean, available ROI sensors are averaged within each subject first; subject ROI TFRs are then averaged so participants contribute equally. Missing sensors are not interpolated.\n"
        f"Plots show {PLOT_TMIN:g} to {PLOT_TMAX:g} s and use one robust symmetric +/-{ROBUST_PERCENTILE:g}th-percentile scale shared by the eight sensor panels and ROI mean within each result.",
        "Analysis",
    )

    add_result(
        report, fig_dir, subject_tfrs, subjects, subjects_by_channel,
        "no-stim", "1. Stim OFF", "Stim OFF (no stimulation)",
        "Percent change from the -0.3 to -0.1 s baseline.",
    )
    add_result(
        report, fig_dir, subject_tfrs, subjects, subjects_by_channel,
        "stim", "2. Stim ON", "Stim ON (stimulation)",
        "Percent change from the -0.3 to -0.1 s baseline.",
    )
    add_result(
        report, fig_dir, subject_tfrs, subjects, subjects_by_channel,
        "ratio", "3. Normalized difference",
        "(Stim ON - Stim OFF) / (Stim ON + Stim OFF)",
        "Calculated from original unbaselined power; no baseline correction.",
    )

    print(f"Eight-posterior-sensor diagnostic complete: {report.pdf_fname}")


if __name__ == "__main__":
    main()
