#!/usr/bin/env python3
"""Test whether STN stimulation changes alpha magnitude in either direction.

This is a SEPARATE exploratory analysis from S01's signed temporal cluster
test. It reuses S02's validated B03 mapping and trial-level alpha extraction.

For each participant, pool retained unbaselined alpha power trials from
the two ON and two OFF periods. Compute (ON-OFF)/(ON+OFF) and take its
absolute value. No baseline correction is used in either sensitivity test.
The group statistic is mean(abs(participant effect)), which cannot be
tested against zero using a conventional one-sample t-test.

For each Monte Carlo permutation, independently shuffle the four WHOLE
period labels per participant, keeping exactly two ON and two OFF
periods. Recompute participant effects, take absolute values, and
average them. The upper-tail p-value compares the observed magnitude
with this randomisation null distribution.

IMPORTANT: This test requires exchangeability of stimulation-period
labels under the experimental design. It does not establish that
individual participants have nonzero effects or control for order
and carryover effects. Report alongside the signed two-sided S01/S02
analyses. No separate CSVs are written.

Run from repository root:
python analysis/statistics/S03_alpha_magnitude_permutation.py \\
    --subjects 101 102 103 105 106 107 108 112 113 114 115 116 118 120 121 123
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from S02_absolute_alpha_block_permutation import (
    validated_subject,
    run_permutation,
)
from pipeline_config import resolve_project_root, stage_path
import mne
from S04_fixed_frequency_tfr import FREQS, WINDOW, ROI8, power_epochs, normalise, plot_map


def arguments():
    """Parse participants, dataset paths and reproducible permutations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--platform", choices=("mac", "bluebear"), default="mac")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--peaks-csv", type=Path)
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.subjects = [
        str(subject).removeprefix("sub-") for subject in args.subjects
    ]
    if len(set(args.subjects)) != len(args.subjects):
        parser.error("Duplicate participant identifiers.")
    if args.n_permutations < 1:
        parser.error("Number of permutations must be positive.")
    return args



def fixed_frequency_analysis(root, args, trials):
    """Run whole-period magnitude permutations without individual peaks."""
    dataset = []
    for subject in args.subjects:
        rows = trials.loc[trials.subject == subject]
        if rows.empty or not rows.label_verified.astype(str).str.lower().eq("true").all():
            raise ValueError(f"sub-{subject}: unverified mapping")
        epochs = {}
        for condition in ("no-stim", "stim"):
            path = stage_path(root, subject, args.session, args.task, args.run,
                              condition, "clean", "epo")
            ep = mne.read_epochs(path, preload=True, verbose=False)
            cues = [key for key in ("cue_onset_left", "cue_onset_right")
                    if key in ep.event_id]
            epochs[condition] = ep[cues]
        channels = [ch for ch in ROI8 if all(
            ch in ep.ch_names and ch not in ep.info["bads"]
            for ep in epochs.values()
        )]
        if not channels:
            raise ValueError(f"sub-{subject}: no good posterior sensors")
        periods = [[] for _ in range(4)]
        labels = np.zeros(4, dtype=bool)
        times = None
        for condition, ep in epochs.items():
            subset = rows.loc[rows.source_condition == condition]
            indices = subset.trial_index_in_condition.to_numpy(dtype=int)
            if not np.array_equal(np.sort(indices), np.arange(len(ep))):
                raise ValueError(f"sub-{subject}: invalid trial indices")
            values, current_times = power_epochs(
                ep, channels, args.n_jobs, average=False
            )
            if times is None:
                times = current_times
            elif not np.array_equal(times, current_times):
                raise ValueError("Condition time grids differ")
            values = values.mean(axis=1)
            for row in subset.itertuples(index=False):
                index = int(row.stimulation_period) - 1
                if index not in range(4):
                    raise ValueError("Invalid stimulation period")
                periods[index].append(values[int(row.trial_index_in_condition)])
        for index in range(4):
            subset = rows.loc[rows.stimulation_period == index + 1]
            if subset.empty or subset.dbs_on.nunique() != 1:
                raise ValueError("Incomplete stimulation period")
            labels[index] = str(subset.dbs_on.iloc[0]).lower() == "true"
        if labels.sum() != 2 or any(not p for p in periods):
            raise ValueError("Expected two ON and two OFF periods")
        dataset.append(([np.stack(p) for p in periods], labels, times))

    times = dataset[0][2]
    if any(not np.array_equal(times, item[2]) for item in dataset):
        raise ValueError("Participant time axes differ")
    window = (times >= WINDOW[0]) & (times <= WINDOW[1])
    alpha_band = (FREQS >= 8.0) & (FREQS <= 12.0)

    def participant_map(periods, labels):
        on = np.concatenate([p for p, flag in zip(periods, labels)
                             if flag]).mean(axis=0)
        off = np.concatenate([p for p, flag in zip(periods, labels)
                              if not flag]).mean(axis=0)
        return normalise(on, off)

    maps = [participant_map(p, labels) for p, labels, _ in dataset]
    observed = float(np.mean([
        np.abs(value[alpha_band][:, window]).mean() for value in maps
    ]))
    rng = np.random.default_rng(args.seed)
    null = np.empty(args.n_permutations)
    for iteration in range(args.n_permutations):
        null[iteration] = np.mean([
            np.abs(participant_map(p, rng.permutation(labels))[alpha_band][:, window]).mean()
            for p, labels, _ in dataset
        ])
    p_value = (
        1 + np.count_nonzero(null >= observed)
    ) / (args.n_permutations + 1)
    return np.mean(maps, axis=0), times, observed, null, p_value


def main():
    """Compute and report the period-randomised magnitude statistic."""
    args = arguments()
    root = resolve_project_root(args.platform, args.project_root)
    group = root / "derivatives" / "reports" / "group"
    trial_path = args.input_csv or (
        group / "EEG_posterior_channels_block_mapping" / "trial_alpha.csv"
    )
    peaks_path = args.peaks_csv or group / "individual_alpha_peaks.csv"
    trials = pd.read_csv(trial_path, dtype={"subject": str})
    peaks_frame = pd.read_csv(peaks_path, dtype={"subject": str})
    required = {"subject", "peak_hz"}
    if not required.issubset(peaks_frame.columns):
        raise ValueError("Peak CSV must contain subject and peak_hz.")
    peaks_frame["subject"] = peaks_frame["subject"].str.removeprefix("sub-")
    if peaks_frame["subject"].duplicated().any():
        raise ValueError("Duplicate subject in alpha peak CSV.")
    peaks = dict(zip(
        peaks_frame["subject"],
        pd.to_numeric(peaks_frame["peak_hz"], errors="coerce"),
    ))
    trials["subject"] = trials["subject"].str.removeprefix("sub-")
    data = []
    for subject in args.subjects:
        if subject not in peaks:
            raise ValueError(f"sub-{subject}: no individual alpha peak.")
        raw_periods, labels, _, _ = validated_subject(
            trials, subject, root, args, peaks[subject]
        )
        data.append((raw_periods, labels))
        print(f"sub-{subject}: four validated stimulation periods")

    effects, observed, null, p_value = run_permutation(
        data, args.n_permutations, args.seed,
        normalized=True, statistic="absolute",
    )
    fixed_mean, fixed_times, fixed_observed, fixed_null, fixed_p = (
        fixed_frequency_analysis(root, args, trials)
    )
    output = group / "alpha_magnitude_permutation"
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / "alpha_magnitude_permutation.pdf"

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(null, bins=45, color="lightsteelblue", edgecolor="white")
        ax.axvline(
            observed, color="darkred", linewidth=2.5,
            label="Observed mean absolute ON - OFF",
        )
        ax.axvline(
            np.quantile(null, 0.95), color="black", linestyle="--",
            label="95th percentile of null",
        )
        ax.set(
            title=(
                f"Alpha effect magnitude, N={len(effects)}; "
                f"permutation p={p_value:.4g}"
            ),
            xlabel="Mean |(ON-OFF)/(ON+OFF)|, individual peak +/-1 Hz",
            ylabel="Number of permutations",
        )
        ax.legend(frameon=False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        lines = [
            "Statistical question: does stimulation alter alpha magnitude",
            "more than expected under whole-period label exchangeability?",
            "",
            "Alpha measure: participant-specific OFF peak +/-1 Hz,",
            "attention interval 0.3-0.8 s, unbaselined linear power.",
            "No baseline correction is performed.",
            "",
            "Each participant has four periods: two OFF, one left ON,",
            "one right ON. Pool trial values within the assigned ON",
            "and OFF conditions, calculate D_i = (ON-OFF)/(ON+OFF).",
            "Observed statistic = mean across participants of abs(D_i).",
            "",
            "For every permutation, shuffle four WHOLE period labels",
            "within each participant, preserving two ON and two OFF.",
            "Recalculate D_i and mean(abs(D_i)); never shuffle trials.",
            "Upper-tail Monte Carlo p = (1 + number of permuted",
            "statistics >= observed) / (1 + permutations).",
            "",
            f"Participants: {', '.join(args.subjects)}",
            f"Number of permutations: {args.n_permutations:,}",
            f"Random seed: {args.seed}",
            f"Observed mean absolute effect: {observed:.6g}",
            f"Upper-tail randomisation p-value: {p_value:.6g}",
            "",
            "This is a group magnitude test, not a one-sided t-test.",
            "It does not imply that all participants have an effect.",
            "Interpretation requires randomisation/exchangeability",
            "of period labels and consideration of time/order effects.",
        ]
        ax.text(
            0.06, 0.96, "\n".join(lines), va="top",
            fontsize=11, linespacing=1.35, transform=ax.transAxes,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(args.subjects, effects, color="steelblue")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(
            title="Signed participant effects (before taking magnitudes)",
            xlabel="Participant",
            ylabel="(ON-OFF)/(ON+OFF), individual peak +/-1 Hz",
        )
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        fig = plot_map(
            fixed_mean, fixed_times,
            "No peak selection: grand-average normalised ON-OFF TFR"
        )
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(fixed_null, bins=45, color="lightsteelblue")
        ax.axvline(fixed_observed, color="darkred", linewidth=2,
                   label="Observed mean absolute normalised effect")
        ax.axvline(np.quantile(fixed_null, 0.95), color="black",
                   linestyle="--", label="95th percentile of null")
        ax.set(
            title=f"Fixed-frequency magnitude test: p={fixed_p:.4g}",
            xlabel="Mean absolute normalised effect, fixed 8-12 Hz",
            ylabel="Permutations",
        )
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.axis("off")
        text = (
            "Fixed-frequency sensitivity (NO individual peak selection)\n\n"
            "Each participant's ON and OFF power is averaged over trials "
            "and good posterior sensors at every absolute frequency/time "
            "point. Compute (ON-OFF)/(ON+OFF) without baseline correction. "
            "The TFR grand average weights participants equally.\n\n"
            "Magnitude statistic: mean across participants of each "
            "participant's mean absolute normalised effect over "
            "fixed 8-12 Hz and 0.3-0.8 s. Shuffle four whole period labels "
            "within each participant, preserving two ON and two OFF, "
            "and recompute the statistic for every permutation.\n\n"
            f"Observed: {fixed_observed:.6g}\n"
            f"Upper-tail permutation p: {fixed_p:.6g}\n"
            f"Permutations: {args.n_permutations}\n\n"
            "This test uses fixed 8-12 Hz without individual peak selection."
        )
        ax.text(0.05, 0.95, text, va="top", transform=ax.transAxes,
                fontsize=10, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)
    print(f"Mean absolute effect: {observed:.6g}")
    print(f"Upper-tail permutation p-value: {p_value:.6g}")
    print(f"Report: {pdf_path}")


if __name__ == "__main__":
    main()
