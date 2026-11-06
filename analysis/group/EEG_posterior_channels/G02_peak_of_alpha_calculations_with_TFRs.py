#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Participant posterior TFR quality control and individual alpha peaks.

For each participant, this script reads final cleaned cue epochs for DBS ON
and OFF and selects posterior EEG sensors retained in both conditions.

The individual alpha peak is estimated from DBS OFF **unbaselined** multitaper
power: average over retained posterior sensors and 0.3-0.8 seconds after cue,
restrict to 8-12 Hz, and select the frequency bin with maximum power
(0.5-Hz resolution). This is a maximum-power estimate, not an
aperiodic-corrected or independently validated oscillatory peak.
The same participant-specific peak +/- --alpha-peak-half-width Hz is used
for ON and OFF alpha curves.

Outputs also include participant-level three- and eight-sensor TFR maps
for ON, OFF, raw ON-OFF, and the unbaselined normalized contrast;
fixed 10-12 Hz single-trial alpha mean +/- SEM curves; and an alpha
consistency diagnostic. ON/OFF descriptive TFRs and peak-centred curves
use percent baseline (-0.3 to -0.1 seconds); raw contrasts do not.
No across-participant averaging or statistical inference is performed.

Prerequisite: final cleaned stim/no-stim cue epochs from the subject EEG
pipelines. Run from the repository root, for example:

    python analysis/group/EEG_posterior_channels/G02_peak_of_alpha_calculations_with_TFRs.py --subjects 115 116 118 --platform mac

Outputs:
    <PROJECT_ROOT>/derivatives/reports/QC/posterior_TFR_all_subjects/
        Participant PDF and figures/
    <PROJECT_ROOT>/derivatives/reports/group/
        EEG_posterior_channels_alpha_changes_stimOn_stimOff/
        individual_alpha_peaks.csv

The CSV supplies participant alpha peaks for the stimulation-laterality
runner. Run the full intended participant set before that analysis because
each run replaces the CSV with the participants selected for that run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import TwoSlopeNorm

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
PEAK_SEARCH = (8.0, 12.0)
PEAK_TIME = (0.3, 0.8)
PLOT_WINDOW = (-0.5, 1.5)


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
    parser.add_argument("--alpha-peak-half-width", type=float, default=1.0)
    args = parser.parse_args()
    if not np.isfinite(args.alpha_peak_half_width) or args.alpha_peak_half_width <= 0:
        parser.error('--alpha-peak-half-width must be positive.')
    return args


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

def alpha_from_tfr(tfr):
    """Extract 10-12 Hz alpha from a baseline-corrected ROI TFR."""

    freq_mask = (
        (tfr.freqs >= 10)
        & (tfr.freqs <= 12)
    )

    time_mask = (
        (tfr.times >= ALPHA_WINDOW[0])
        & (tfr.times <= ALPHA_WINDOW[1])
    )

    # Average across channels and alpha frequencies
    alpha = tfr.data[:, freq_mask, :].mean(axis=(0, 1))

    return tfr.times[time_mask], alpha[time_mask] * 100

def check_alpha_consistency(tfr, alpha_times, alpha_values):
    """Compare alpha extracted from TFR with the QC alpha curve."""

    tfr_times, tfr_alpha = alpha_from_tfr(tfr)

    if not np.allclose(tfr_times, alpha_times):
        raise ValueError("Time points do not match.")

    difference = tfr_alpha - alpha_values

    return {
        "max_absolute_difference": np.max(np.abs(difference)),
        "mean_absolute_difference": np.mean(np.abs(difference)),
        "tfr_alpha_mean": np.mean(tfr_alpha),
        "qc_alpha_mean": np.mean(alpha_values),
    }

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
    """Compute baseline-corrected alpha after averaging trials."""

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

        # Baseline-correct each trial independently
        tfr.apply_baseline(BASELINE, mode="percent")

        # Select alpha frequencies
        freq_mask = (
            (tfr.freqs >= 10)
            & (tfr.freqs <= 12)
        )

        # Select time window
        time_mask = (
            (tfr.times >= ALPHA_WINDOW[0])
            & (tfr.times <= ALPHA_WINDOW[1])
        )

        # Average across channels and alpha frequencies
        # Shape: (n_trials, n_times)
        trials = tfr.data[:, :, freq_mask, :].mean(axis=(1, 2))
        trials = trials[:, time_mask]

        # Convert fractional change to percentage
        trials = trials * 100

        # Calculate mean and SEM across trials
        count = len(trials)
        mean = trials.mean(axis=0)

        sem = (
            trials.std(axis=0, ddof=1) / np.sqrt(count)
            if count > 1
            else np.zeros_like(mean)
        )

        curves[condition] = (
            tfr.times[time_mask],
            mean,
            sem,
            count,
        )

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


def alpha_peak(off_tfr):
    """Identify each participant's alpha peak from DBS OFF power.

    The input is an unbaselined multitaper TFR, already averaged
    across trials using compute_tfr(average=True).
    The input shape is:
        (n_channels, n_frequencies, n_times)

    Steps
    -----
    1. Select the time interval 0.3-0.8 seconds after cue onset.

    2. Average absolute power across the selected time points
       and all retained posterior channels, independently for
       each frequency.

    3. Restrict the resulting power spectrum to 8-12 Hz.

    4. Identify the frequency with the highest mean power.
       Frequencies are sampled in 0.5 Hz steps.

    5. Return this frequency as the participant's alpha peak.

    The peak is identified using DBS OFF only, avoiding selection
    based on the stimulation effect.

    The same peak +/- 1 Hz frequency band is subsequently used
    for DBS ON and DBS OFF alpha time courses.

    Important
    ---------
    No baseline correction is applied during peak identification.
    The algorithm selects the maximum raw power within 8-12 Hz.
    It does not remove the aperiodic 1/f background or require
    the selected maximum to be a genuine local oscillatory peak.
    """
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
    """Generate participant QC, OFF-derived alpha peaks, and peak CSV."""
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

    # Reuse the already computed trial-averaged TFRs for individual peaks.
    peak_dir = (root / "derivatives" / "reports" / "group"
                / "EEG_posterior_channels_alpha_changes_stimOn_stimOff")
    peak_dir.mkdir(parents=True, exist_ok=True)
    peak_rows = ["subject,peak_hz,channels,n_off,n_on,status"]
    report.add_text(
        "OFF-derived individual alpha peaks",
        "Select the 8-12 Hz peak from unbaselined DBS OFF power "
        "averaged across available posterior sensors and 0.3-0.8 s. "
        "Peak-centred ON/OFF curves use condition-wise percent baseline "
        "-0.3 to -0.1 s. CSV is also used by stimulation-laterality analysis.",
        "Individual alpha peaks",
    )
    for subject in subjects:
        if subject not in raw_tfrs:
            peak_rows.append(f"{subject},,,,,skipped")
            continue
        counts = {c: len(epochs[subject][c]) for c in CONDITIONS}
        figure, peak = plot_participant(
            subject, raw_tfrs[subject], good[subject], counts,
            args.alpha_peak_half_width,
        )
        report.add_figure(
            figure,
            str(figures_dir / f"individual_peak_sub-{subject}.png"),
            f"sub-{subject}: alpha peak {peak:.1f} Hz",
            "Individual OFF-derived peak, condition-wise baseline-corrected curves.",
            "Individual alpha peaks",
        )
        plt.close(figure)
        peak_rows.append(
            f"{subject},{peak:.2f},{len(good[subject])},"
            f"{counts['no-stim']},{counts['stim']},ok"
        )
    (peak_dir / "individual_alpha_peaks.csv").write_text(
        "\n".join(peak_rows) + "\n", encoding="utf-8"
    )

    consistency_results = []

    for subject in subjects:
        if subject not in raw_tfrs:
            continue

        for roi_name, roi_channels in (
            ("mean3", ROI3),
            ("mean8", ROI8),
        ):
            picks = [
                ch for ch in roi_channels
                if ch in good[subject]
            ]

            if not picks:
                continue

            # Calculate the existing trial-wise alpha curves
            qc_curves = alpha_curves(
                epochs[subject],
                picks,
                args.n_jobs,
            )

            for condition in ("no-stim", "stim"):

                # Use the same TFR as the descriptive TFR figures
                tfr = raw_tfrs[subject][condition].copy()

                # Baseline after trial averaging
                tfr.apply_baseline(BASELINE, mode="percent")

                # Average the same posterior channels
                roi_tfr, _ = mean_roi(tfr, picks)

                # Existing QC alpha curve
                times, alpha, sem, count = qc_curves[condition]

                result = check_alpha_consistency(
                    roi_tfr,
                    times,
                    alpha,
                )

                consistency_results.append(
                    f"sub-{subject}, {roi_name}, {condition}: "
                    f"max difference = "
                    f"{result['max_absolute_difference']:.6f}, "
                    f"mean difference = "
                    f"{result['mean_absolute_difference']:.6f}, "
                    f"TFR mean = "
                    f"{result['tfr_alpha_mean']:.6f}, "
                    f"QC mean = "
                    f"{result['qc_alpha_mean']:.6f}"
                )
    report.add_text(
    "Alpha consistency check",
    "\n".join(consistency_results),
    "Alpha QC",
    )

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
