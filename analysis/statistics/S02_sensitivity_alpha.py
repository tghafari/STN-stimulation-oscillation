#!/usr/bin/env python3
"""
===================================================================
Optional sensitivity tests for S02's validated G04 alpha data.

Symmetric normalised effect: (mean ON - mean OFF) / (mean ON + mean OFF).
Both ON and OFF means in the denominator are recalculated for EVERY permutation. Whole
stimulation-period labels are shuffled within each participant, keeping
two ON and two OFF periods. Participants contribute equally.

Leave-one-out: rerun the original absolute-power permutation test after
omitting each participant. These are influence checks, not confirmatory
independent tests.

Usage:
python analysis/statistics/S02_sensitivity_alpha.py --subjects 101 102 103
===================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "subject" / "EEG_all_channels"))
from pipeline_config import resolve_project_root


def read_data(path, subjects):
    """Load verified G04 trials and require four complete periods."""
    frame = pd.read_csv(path, dtype={"subject": str})
    frame["subject"] = frame["subject"].str.removeprefix("sub-")
    required = {"subject", "stimulation_period", "dbs_on",
                "stimulation_side", "alpha_power", "label_verified"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {required - set(frame.columns)}")
    data = []
    for subject in subjects:
        subset = frame.loc[frame["subject"] == subject]
        if subset.empty or not subset["label_verified"].astype(
            str
        ).str.lower().eq("true").all():
            raise ValueError(f"sub-{subject}: missing or unverified data")
        periods = []
        labels = []
        sides = []
        for period in range(1, 5):
            rows = subset.loc[subset["stimulation_period"] == period]
            if rows.empty or rows["dbs_on"].nunique() != 1:
                raise ValueError(f"sub-{subject}: invalid period {period}")
            if rows["stimulation_side"].nunique() != 1:
                raise ValueError(f"sub-{subject}: mixed stimulation sides")
            values = rows["alpha_power"].to_numpy(dtype=float)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"sub-{subject}: invalid absolute power")
            side = str(rows["stimulation_side"].iloc[0]).lower()
            on = str(rows["dbs_on"].iloc[0]).lower() == "true"
            if on != (side != "off"):
                raise ValueError(f"sub-{subject}: condition mismatch")
            periods.append(values)
            labels.append(on)
            sides.append(side)
        if sorted(sides) != ["off", "off", "stim_left", "stim_right"]:
            raise ValueError(f"sub-{subject}: invalid period labels {sides}")
        data.append((periods, np.asarray(labels, dtype=bool)))
    return data


def participant_effect(periods, labels, percentage=False):
    """Compute trial-weighted ON-OFF, optionally divided by ON+OFF."""
    on = np.concatenate([x for x, flag in zip(periods, labels) if flag])
    off = np.concatenate([x for x, flag in zip(periods, labels) if not flag])
    difference = float(on.mean() - off.mean())
    if not percentage:
        return difference
    denominator = float(on.mean() + off.mean())
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("ON + OFF mean must be finite and positive")
    return difference / denominator


def run_test(data, iterations, seed, percentage=False):
    """Permute four whole-period labels; return observed, null and p."""
    observed_by_subject = np.asarray([
        participant_effect(p, l, percentage) for p, l in data
    ])
    observed = float(observed_by_subject.mean())
    rng = np.random.default_rng(seed)
    null = np.empty(iterations)
    for i in range(iterations):
        null[i] = np.mean([
            participant_effect(p, rng.permutation(l), percentage)
            for p, l in data
        ])
    p_value = (1 + np.count_nonzero(
        np.abs(null) >= abs(observed)
    )) / (iterations + 1)
    return observed_by_subject, observed, null, p_value


def report_text_page(title, sections):
    """Create a readable methods page with no clipped paragraphs."""
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    ax.text(0.055, 0.95, title, fontsize=17, weight="bold",
            transform=ax.transAxes, va="top")
    y = 0.87
    for heading, paragraph in sections:
        ax.text(0.055, y, heading, fontsize=12, weight="bold",
                transform=ax.transAxes, va="top")
        y -= 0.045
        lines = textwrap.fill(paragraph, width=94)
        ax.text(0.055, y, lines, fontsize=10, linespacing=1.45,
                transform=ax.transAxes, va="top")
        y -= 0.025 * (lines.count("\n") + 1) + 0.06
    return fig


def save_figure(fig, directory, stem):
    """Export publication-quality vector SVG and 600-dpi TIFF."""
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "tiff"):
        fig.savefig(
            directory / f"{stem}.{suffix}",
            dpi=600, bbox_inches="tight",
        )


def mark_two_sided_thresholds(ax, distribution):
    """Mark the two-sided alpha=0.05 absolute-null critical threshold.

    The p-value compares absolute permuted and observed statistics.
    Therefore both critical lines are at +/- the 95th percentile of
    absolute null statistics, rather than signed-tail quantiles.
    """
    cutoff = float(np.quantile(np.abs(distribution), 0.95))
    ax.axvline(
        -cutoff, color="#404040", linestyle="--", linewidth=1.5,
        label="Two-sided p = 0.05 threshold",
    )
    ax.axvline(cutoff, color="#404040", linestyle="--", linewidth=1.5)
    ax.legend(frameon=False)


def create_report(out, subjects, absolute, normalised, loo, seed):
    """Write a combined primary and sensitivity PDF from actual results."""
    absolute_effects = pd.read_csv(out / "participant_effects.csv")
    primary_null = pd.read_csv(out / "permutation_null.csv")[
        "permuted_group_effect"
    ].to_numpy(dtype=float)
    if set(absolute_effects["subject"].astype(str)) != set(subjects):
        raise ValueError(
            "Primary S02 participant list differs from sensitivity run. "
            "Rerun primary S02 with exactly the same participants."
        )
    absolute_observed = float(absolute_effects["on_minus_off"].mean())
    primary_p = (
        1 + np.count_nonzero(
            np.abs(primary_null) >= abs(absolute_observed)
        )
    ) / (len(primary_null) + 1)
    values, norm_observed, norm_null, norm_p = normalised
    report_path = out / "absolute_alpha_block_permutation.pdf"
    figures_dir = out / "figures"
    with PdfPages(report_path) as pdf:
        pages = [
            report_text_page("Absolute alpha: data and observed effect", [
                ("Input and alpha extraction",
                 "G04 trial_alpha.csv contains validated cue-to-block mapping, "
                 "confirmed stimulation labels and absolute multitaper alpha "
                 "power for each retained trial. G04 averages over available "
                 "posterior EEG channels, the individual OFF-derived alpha "
                 "peak +/- 1 Hz, and 0.3-0.8 seconds after cue onset. "
                 "No baseline correction is performed."),
                ("Stimulation structure",
                 "Eight task blocks are grouped into four stimulation periods "
                 "per participant: two OFF, one left ON, and one right ON. "
                 "Each stimulation period contains two consecutive task blocks."),
                ("Observed statistic",
                 "Pool trials within the two ON periods and within the two "
                 "OFF periods separately for each participant. Calculate "
                 "participant mean ON minus mean OFF. The observed group "
                 "statistic is the average of these participant differences, "
                 "giving every participant equal weight."),
            ]),
            report_text_page("Absolute alpha: permutation procedure", [
                ("One label permutation",
                 "For every participant independently, randomly reassign "
                 "the four whole stimulation-period labels while preserving "
                 "two ON and two OFF periods. All trial measurements remain "
                 "attached to their original period. Neither trials nor "
                 "participants are shuffled."),
                ("Recompute the group statistic",
                 "After relabelling, pool ON and OFF trials within each "
                 "participant and recalculate the difference. Average the "
                 "participant differences to obtain one permuted group value. "
                 f"Repeat {len(primary_null):,} times with the recorded seed "
                 f"({seed}) for the sensitivity analysis."),
                ("Two-sided permutation p-value",
                 "Count null values with absolute magnitude at least as "
                 "large as the absolute observed statistic. Calculate "
                 "p = (count + 1) / (number of permutations + 1)."),
                ("Assumptions",
                 "This procedure assumes stimulation-period labels are "
                 "exchangeable under the null according to the experimental "
                 "randomisation. It does not model order or carryover. "
                 "The individual OFF-selected alpha peaks remain fixed."),
            ]),
        ]
        for page in pages:
            pdf.savefig(page)
            plt.close(page)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(primary_null * 1e12, bins=50, color="lightsteelblue")
        ax.axvline(absolute_observed * 1e12, color="darkred", lw=2)
        ax.set(
            title=f"Absolute alpha: n={len(subjects)}, two-sided p={primary_p:.4g}",
            xlabel="Group ON - OFF (microvolt squared)",
            ylabel="Permutations",
        )
        mark_two_sided_thresholds(ax, primary_null * 1e12)
        fig.tight_layout()
        save_figure(fig, figures_dir, "absolute_alpha_permutation")
        pdf.savefig(fig)
        plt.close(fig)
        page = report_text_page("Sensitivity analysis: methods", [
            ("Symmetric normalised difference",
             "For each participant, calculate (mean ON - mean OFF) / "
             "(mean ON + mean OFF), with means pooled across retained "
             "trials in their two respective stimulation periods. "
             "This relative measure is dimensionless and ranges from "
             "-1 to +1 for nonnegative power."),
            ("Normalised permutation test",
             "Repeat the same whole-period ON/OFF label shuffling. "
             "For every permutation, recalculate both the numerator "
             "and the ON+OFF denominator from the newly assigned "
             "conditions. Average participant normalised differences "
             "and compare the observed magnitude against the null "
             "distribution using the two-sided Monte Carlo formula."),
            ("Leave-one-participant-out",
             "Remove one participant, repeat the original absolute "
             "group test with all remaining participants and 10,000 "
             "whole-period label permutations, then restore that "
             "participant and omit the next. Report the new group "
             "effect and two-sided p-value for each omission. "
             "This is an influence diagnostic, not an exclusion rule."),
            ("Why both checks?",
             "Normalisation reduces sensitivity to between-participant "
             "differences in overall power, although tiny denominators "
             "can be unstable. Leave-one-out checks whether a single "
             "participant strongly influences the absolute result. "
             "Neither replaces the primary analysis."),
        ])
        pdf.savefig(page)
        plt.close(page)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(norm_null, bins=50, color="lightsteelblue")
        ax.axvline(norm_observed, color="darkred", lw=2)
        ax.set(
            title=f"Normalised alpha: two-sided p={norm_p:.4g}",
            xlabel="Group mean (ON - OFF) / (ON + OFF)",
            ylabel="Permutations",
        )
        mark_two_sided_thresholds(ax, norm_null)
        fig.tight_layout()
        save_figure(fig, figures_dir, "normalised_alpha_permutation")
        pdf.savefig(fig)
        plt.close(fig)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.7, 8.3))
        ax1.scatter(loo["omitted_subject"], loo["observed_on_minus_off_v2"] * 1e12)
        ax1.axhline(0, color="gray", lw=1)
        ax1.set(ylabel="ON - OFF (microvolt squared)",
                title="Leave-one-out: absolute group effect")
        ax2.scatter(loo["omitted_subject"], loo["two_sided_p"])
        ax2.axhline(0.05, color="gray", linestyle="--")
        ax2.set(xlabel="Omitted participant", ylabel="Two-sided p",
                title="Leave-one-out: permutation p-values")
        for axis in (ax1, ax2):
            axis.tick_params(axis="x", rotation=50)
        fig.tight_layout()
        save_figure(fig, figures_dir, "leave_one_out_effect_and_p")
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        ax.set_title("Leave-one-out: full numerical results", fontsize=15)
        cells = [[
            str(row.omitted_subject),
            f"{row.observed_on_minus_off_v2 * 1e12:.4g}",
            f"{row.two_sided_p:.4g}",
        ] for row in loo.itertuples()]
        table = ax.table(
            cellText=cells,
            colLabels=["Omitted", "ON - OFF (microvolt squared)", "Two-sided p"],
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"Updated combined report: {report_path}")


def main():
    """Export optional percentage and leave-one-out sensitivity results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--platform", default="mac",
                        choices=("mac", "bluebear"))
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n_permutations < 1:
        parser.error("--n-permutations must be positive")
    subjects = [s.removeprefix("sub-") for s in args.subjects]
    if len(subjects) != len(set(subjects)):
        raise ValueError("Duplicate participant")
    root = resolve_project_root(args.platform, args.project_root)
    group = root / "derivatives" / "reports" / "group"
    path = args.input_csv or (
        group / "EEG_posterior_channels_block_mapping" / "trial_alpha.csv"
    )
    data = read_data(path, subjects)
    out = group / "absolute_alpha_block_permutation"
    out.mkdir(parents=True, exist_ok=True)
    percentages, observed, null, p_value = run_test(
        data, args.n_permutations, args.seed, percentage=True
    )
    pd.DataFrame({
        "subject": subjects, "normalised_on_minus_off": percentages
    }).to_csv(out / "participant_normalised_effects.csv", index=False)
    pd.DataFrame({
        "permuted_group_normalised": null
    }).to_csv(out / "permutation_null_normalised.csv", index=False)
    rows = []
    for index, subject in enumerate(subjects):
        remaining = [d for j, d in enumerate(data) if j != index]
        _, statistic, _, p = run_test(
            remaining, args.n_permutations, args.seed, percentage=False
        )
        rows.append({
            "omitted_subject": subject,
            "remaining_n": len(remaining),
            "observed_on_minus_off_v2": statistic,
            "two_sided_p": p,
        })
    loo = pd.DataFrame(rows)
    loo.to_csv(out / "leave_one_out_absolute.csv", index=False)
    create_report(
        out, subjects, None, (percentages, observed, null, p_value),
        loo, args.seed
    )
    print(f"Normalised (ON-OFF)/(ON+OFF) = {observed:.5g}, p = {p_value:.5g}")
    print(f"Sensitivity outputs: {out}")


if __name__ == "__main__":
    main()
