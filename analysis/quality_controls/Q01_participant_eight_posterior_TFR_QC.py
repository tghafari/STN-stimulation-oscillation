#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q01: Participant-level posterior TFR and alpha-power quality control.

Steps
-----
1. Load each participant's final cleaned cue epochs for stim ON and OFF.
2. Identify the posterior sensors available in both conditions.
3. Calculate trial-averaged multitaper TFRs separately by condition.
4. Plot stim OFF, stim ON, their raw-power difference and normalized
   difference, for both the three- and eight-channel posterior ROIs.
5. Split participant TFRs into readable grids (up to four rows, two columns).
6. Compute single-trial 10-12 Hz alpha power, baseline-correct each trial,
   and calculate the within-participant mean and SEM across trials.
7. Display alpha time courses in matching participant grids, separately
   for the three- and eight-channel ROIs.
8. Add the figures and methods to the same cumulative PDF report.

Prerequisites: complete the posterior pipeline and the all-channel
subject pipeline for every participant. Use final cleaned cue epochs.

The script never averages across participants. The two TFR contrasts are
computed from original, unbaselined power; descriptive stim ON/OFF TFRs
use percent baseline (-0.3 to -0.1 seconds). Alpha curves use trial-wise
percent baseline and show mean +/- SEM across trials.

Run from the repository root:
    python analysis/quality_controls/EEG_all_channels/Q01_participant_eight_posterior_TFR_QC.py \
        --subjects 101 102 103 104 --platform mac
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

for folder in (
    ANALYSIS_DIR / "subject" / "EEG_all_channels",
    ANALYSIS_DIR / "utils",
):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from pipeline_config import CONDITIONS, resolve_project_root, stage_path
from pdf_report import ParticipantPDF

ROI3 = ("PO3", "POz", "PO4")
ROI8 = ("PO3", "POz", "PO4", "O1", "Oz", "O2", "PO7", "PO8")
BASELINE = (-0.3, -0.1)
FREQS = np.arange(2.0, 32.0, 0.5)
N_CYCLES = FREQS / 2.0
TIME_BANDWIDTH = 2.0
DECIM = 2
TMIN, TMAX = -0.5, 1.5
ALPHA_WINDOW = (-0.3, 1.3)
ROBUST = 98.0
SUBJECTS_PER_FIGURE = 8


def parse_args():
    """Read participant IDs, dataset paths, and parallelism."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument("--platform", choices=("mac", "bluebear"), default="mac")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def load(root, subject, args):
    """Load both stimulation conditions and combine cue directions."""
    epochs = {}
    for condition in CONDITIONS:
        filename = stage_path(
            root, subject, args.session, args.task, args.run, condition, "clean", "epo"
        )
        if not filename.exists():
            raise FileNotFoundError(
                f"Missing final epochs for sub-{subject}, {condition}: " f"{filename}"
            )
        ep = mne.read_epochs(filename, preload=True)
        cues = [
            cue for cue in ("cue_onset_right", "cue_onset_left") if cue in ep.event_id
        ]
        epochs[condition] = ep[cues] if cues else ep
    return epochs


def available(epochs):
    """Return posterior channels retained in BOTH conditions."""
    stim = epochs["stim"]
    no_stim = epochs["no-stim"]
    eeg_channels = stim.copy().pick("eeg").ch_names
    return [
        ch
        for ch in ROI8
        if ch in eeg_channels
        and ch in no_stim.ch_names
        and ch not in stim.info["bads"]
        and ch not in no_stim.info["bads"]
    ]


def compute(epochs, channels, n_jobs):
    """Compute trial-averaged multitaper power, without baseline."""
    return (
        epochs.copy()
        .pick(channels)
        .compute_tfr(
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
    )


def ratio(stim, no_stim):
    """Normalized power contrast on original unbaselined power."""
    result = stim.copy()
    result.data = (stim.data - no_stim.data) / (
        stim.data + no_stim.data + np.finfo(float).eps
    )
    return result


def mean_roi(tfr, roi):
    """Average available channels within one participant."""
    channels = [ch for ch in roi if ch in tfr.ch_names]
    if not channels:
        return None, []
    result = tfr.copy().pick(channels)
    result.data = result.data.mean(axis=0, keepdims=True)
    return result, channels


def common_lim(subject_rois):
    """Robust symmetric color limits, shared across all figure pages."""
    values = []
    for tfr in subject_rois.values():
        if tfr is None:
            continue
        times = (tfr.times >= TMIN) & (tfr.times <= TMAX)
        freqs = (tfr.freqs >= 2) & (tfr.freqs <= 31.5)
        data = np.asarray(tfr.data)[:, freqs][:, :, times]
        finite = data[np.isfinite(data)]
        if finite.size:
            values.append(finite)
    if not values:
        return None, None
    maximum = float(np.percentile(np.abs(np.concatenate(values)), ROBUST))
    if not np.isfinite(maximum) or maximum <= 0:
        return None, None
    return -maximum, maximum


def split_into_pages(subjects, subjects_per_page=SUBJECTS_PER_FIGURE):
    """Yield consecutive participant groups for readable figures."""
    for index in range(0, len(subjects), subjects_per_page):
        yield subjects[index : index + subjects_per_page]


def plot_grid(subjects, rois, title, limits):
    """Display up to eight participant TFRs in a 4-row, 2-column grid."""
    n_rows = int(np.ceil(len(subjects) / 2))
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(12, 3.3 * n_rows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()
    last_image = None

    for index, ax in enumerate(axes):
        if index >= len(subjects):
            ax.axis("off")
            continue

        subject = subjects[index]
        tfr = rois.get(subject)
        if tfr is None:
            ax.axis("off")
            ax.set_title(f"sub-{subject}: unavailable")
            continue

        time_mask = (tfr.times >= TMIN) & (tfr.times <= TMAX)
        freq_mask = (tfr.freqs >= 2) & (tfr.freqs <= 31.5)
        data = tfr.data[0][freq_mask][:, time_mask]

        last_image = ax.pcolormesh(
            tfr.times[time_mask],
            tfr.freqs[freq_mask],
            data,
            shading="auto",
            cmap="RdBu_r",
            vmin=limits[0],
            vmax=limits[1],
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"sub-{subject}")
        ax.set_xlabel("Time from cue onset (s)")
        ax.set_ylabel("Frequency (Hz)")

    if last_image is not None:
        fig.colorbar(last_image, ax=list(axes), shrink=0.8, label="TFR power")
    fig.suptitle(title, fontsize=15)
    return fig


def alpha_curves(epochs_pair, channels, n_jobs):
    """Return trial-level 10-12 Hz alpha mean and SEM per condition.

    Each trial is baseline-normalized independently before averaging
    across frequency and channels. SEM is across trials, NOT subjects.
    """
    curves = {}
    for condition in ("no-stim", "stim"):
        tfr = (
            epochs_pair[condition]
            .copy()
            .pick(channels)
            .compute_tfr(
                method="multitaper",
                freqs=FREQS,
                n_cycles=N_CYCLES,
                time_bandwidth=TIME_BANDWIDTH,
                use_fft=True,
                zero_mean=True,
                return_itc=False,
                average=False,
                decim=DECIM,
                n_jobs=n_jobs,
            )
        )
        tfr.apply_baseline(BASELINE, mode="percent")
        freq_mask = (tfr.freqs >= 10) & (tfr.freqs <= 12)
        time_mask = (tfr.times >= ALPHA_WINDOW[0]) & (tfr.times <= ALPHA_WINDOW[1])
        trials = tfr.data[:, :, freq_mask, :].mean(axis=(1, 2))
        trials = trials[:, time_mask]
        count = len(trials)
        mean = trials.mean(axis=0)
        sem = (
            trials.std(axis=0, ddof=1) / np.sqrt(count)
            if count > 1
            else np.zeros_like(mean)
        )
        curves[condition] = (tfr.times[time_mask], mean, sem, count)
    return curves


def plot_alpha_grid(subjects, curves, title):
    """Plot alpha ON/OFF mean +/- SEM in the same 4 x 2 TFR layout."""
    n_rows = int(np.ceil(len(subjects) / 2))
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(12, 3.3 * n_rows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    for index, ax in enumerate(axes):
        if index >= len(subjects):
            ax.axis("off")
            continue
        subject = subjects[index]
        if subject not in curves:
            ax.axis("off")
            ax.set_title(f"sub-{subject}: unavailable")
            continue

        for condition, color, label in (
            ("no-stim", "tab:blue", "Stim OFF"),
            ("stim", "tab:orange", "Stim ON"),
        ):
            times, mean, sem, count = curves[subject][condition]
            ax.plot(times, mean, color=color, label=f"{label} (n={count})")
            ax.fill_between(times, mean - sem, mean + sem, color=color, alpha=0.2)

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.7)
        ax.set_xlim(*ALPHA_WINDOW)
        ax.set_title(f"sub-{subject}")
        ax.set_xlabel("Time from cue onset (s)")
        ax.set_ylabel("Alpha power change (%)")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=15)
    return fig


def main():
    """Create one cumulative PDF with paginated TFR and alpha QC."""
    args = parse_args()
    subjects = [s.removeprefix("sub-") for s in args.subjects]
    root = resolve_project_root(args.platform, args.project_root)
    output_dir = root / "derivatives" / "reports" / "QC" / "posterior_TFR_all_subjects"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    report = ParticipantPDF(
        str(output_dir),
        "posterior_TFR_QC_" + "_".join(subjects),
    )

    epochs = {s: load(root, s, args) for s in subjects}
    good = {s: available(epochs[s]) for s in subjects}

    availability = "\n".join(
        f"sub-{s}: available={', '.join(good[s]) or 'none'}; "
        f"missing={', '.join(ch for ch in ROI8 if ch not in good[s]) or 'none'}"
        for s in subjects
    )
    report.add_text(
        "Subjects and channel availability",
        "Subjects: " + ", ".join(f"sub-{s}" for s in subjects) + "\n\n" + availability,
        "QC overview",
    )
    report.add_text(
        "Analysis details",
        "Each panel represents one participant; no group averaging. "
        "TFRs: multitaper 2-31.5 Hz, 0.5-Hz steps, n_cycles=f/2, "
        "time-bandwidth=2, decim=2. Stim ON/OFF descriptive power "
        "uses percent baseline -0.3 to -0.1 s; difference and "
        "normalized difference use unbaselined power. "
        "Figures are paginated into grids of at most 4 rows x 2 columns. "
        "Alpha curves: 10-12 Hz, -0.3 to 1.3 s, condition-wise "
        "trial-level percent baseline, mean +/- SEM across trials.",
        "QC overview",
    )

    raw_tfrs = {}
    for subject in subjects:
        if not good[subject]:
            continue
        print(f"Computing TFRs for sub-{subject}")
        raw_tfrs[subject] = {
            condition: compute(epochs[subject][condition], good[subject], args.n_jobs)
            for condition in CONDITIONS
        }

    sections = (
        ("Stim OFF", "no-stim"),
        ("Stim ON", "stim"),
        ("Stim ON - Stim OFF", "difference"),
        (
            "(Stim ON - Stim OFF) / (Stim ON + Stim OFF)",
            "ratio",
        ),
    )

    for title, key in sections:
        roi_results = {"mean3": {}, "mean8": {}}
        used = {"mean3": {}, "mean8": {}}

        for subject in subjects:
            if subject not in raw_tfrs:
                for roi_name in roi_results:
                    roi_results[roi_name][subject] = None
                continue

            stim = raw_tfrs[subject]["stim"]
            no_stim = raw_tfrs[subject]["no-stim"]

            if key in ("stim", "no-stim"):
                tfr = raw_tfrs[subject][key].copy()
                tfr.apply_baseline(BASELINE, mode="percent")
                caption = "Condition-wise percent baseline -0.3 to -0.1 s."
            elif key == "difference":
                tfr = stim.copy()
                tfr.data = stim.data - no_stim.data
                caption = "Difference from unbaselined power."
            else:
                tfr = ratio(stim, no_stim)
                caption = "Normalized difference from unbaselined power."

            for roi_name, channels in (("mean3", ROI3), ("mean8", ROI8)):
                result, picked = mean_roi(tfr, channels)
                roi_results[roi_name][subject] = result
                used[roi_name][subject] = picked

        report.add_text(
            "Section details",
            f"{title}. {caption} Each panel is one participant. "
            "Colour limits are shared across every figure in this section.",
            title,
        )

        for roi_name in ("mean3", "mean8"):
            limits = common_lim(roi_results[roi_name])
            for figure_number, page_subjects in enumerate(
                split_into_pages(subjects), start=1
            ):
                figure = plot_grid(
                    page_subjects,
                    roi_results[roi_name],
                    f"{title}: {roi_name}",
                    limits,
                )
                filename = (
                    figures_dir / f"{key}_all_subjects_{roi_name}_{figure_number}.png"
                )
                report.add_figure(
                    figure,
                    str(filename),
                    f"{title}: {roi_name}",
                    "Participants: "
                    + ", ".join(f"sub-{s}" for s in page_subjects)
                    + ". "
                    + caption,
                    title,
                )
                plt.close(figure)

        channel_details = []
        for roi_name in ("mean3", "mean8"):
            channel_details.append(f"{roi_name} ROI:")
            for subject in subjects:
                names = used[roi_name].get(subject, [])
                channel_details.append(
                    f"sub-{subject}: {', '.join(names) or 'unavailable'}"
                )
        report.add_text(
            "Channels used per participant",
            "\n".join(channel_details),
            title,
        )

    alpha_section = "Alpha power QC: 10-12 Hz"
    report.add_text(
        "Alpha power QC: stimulation ON vs OFF",
        "Single-trial 10-12 Hz power, -0.3 to 1.3 s relative to cue. "
        "Percent baseline -0.3 to -0.1 s applied independently to "
        "each trial in each condition. Mean +/- SEM across trials "
        "within each participant, not across participants.",
        alpha_section,
    )

    for roi_name, channels in (("mean3", ROI3), ("mean8", ROI8)):
        alpha_results = {}
        for subject in subjects:
            picks = [ch for ch in channels if ch in good[subject]]
            if not picks:
                continue
            print(f"Alpha QC sub-{subject}: {roi_name}")
            alpha_results[subject] = alpha_curves(epochs[subject], picks, args.n_jobs)

        for figure_number, page_subjects in enumerate(
            split_into_pages(subjects), start=1
        ):
            figure = plot_alpha_grid(
                page_subjects,
                alpha_results,
                f"Alpha power (10-12 Hz): {roi_name}, mean +/- SEM",
            )
            filename = figures_dir / f"alpha_10-12Hz_{roi_name}_{figure_number}.png"
            report.add_figure(
                figure,
                str(filename),
                f"Alpha power 10-12 Hz: {roi_name}",
                "Stim ON versus OFF; trial-level percent baseline "
                "-0.3 to -0.1 s; shaded +/- SEM across trials. "
                "Participants: " + ", ".join(f"sub-{s}" for s in page_subjects),
                alpha_section,
            )
            plt.close(figure)

    print(f"QC report complete: {report.pdf_fname}")


if __name__ == "__main__":
    main()
