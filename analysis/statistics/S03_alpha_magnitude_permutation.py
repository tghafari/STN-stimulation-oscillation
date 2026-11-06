#!/usr/bin/env python3
"""Test whether STN stimulation changes alpha magnitude in either direction.

This is a SEPARATE exploratory analysis from S01's signed temporal cluster
test. It reuses S02's validated B03 mapping and trial-level alpha extraction.

For each participant, pool retained trials from the two ON and two OFF
periods and calculate the ON-minus-OFF effect in percentage points.
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
from pipeline_config import resolve_project_root


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
        percent_periods, _, labels, _, _ = validated_subject(
            trials, subject, root, args, peaks[subject]
        )
        data.append((percent_periods, percent_periods, labels))
        print(f"sub-{subject}: four validated stimulation periods")

    effects, observed, null, p_value = run_permutation(
        data, args.n_permutations, args.seed,
        normalized=False, statistic="absolute",
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
            xlabel="Mean absolute participant ON - OFF (percentage points)",
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
            "attention interval 0.3-0.8 s, trial-level percent baseline",
            "correction (-0.3 to -0.1 s) as in S02.",
            "",
            "Each participant has four periods: two OFF, one left ON,",
            "one right ON. Pool trial values within the assigned ON",
            "and OFF conditions, calculate D_i = mean ON - mean OFF.",
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
            f"Observed mean absolute effect: {observed:.6g} pp",
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
            ylabel="ON - OFF (percentage points)",
        )
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"Mean absolute effect: {observed:.6g} pp")
    print(f"Upper-tail permutation p-value: {p_value:.6g}")
    print(f"Report: {pdf_path}")


if __name__ == "__main__":
    main()
