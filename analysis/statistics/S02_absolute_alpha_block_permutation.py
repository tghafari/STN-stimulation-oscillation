#!/usr/bin/env python3
"""S02: baseline-relative posterior alpha stimulation effects.

Read B03's verified trial-to-period mapping, then recompute trial-level
multitaper alpha from the final cleaned cue epochs. Percent baseline
(-0.3 to -0.1 s) is applied to EACH trial and sensor before averaging
the participant-specific OFF-derived alpha band (peak +/-1 Hz) and the
attention interval (0.3 to 0.8 s). This avoids attempting to baseline
correct the already time-averaged absolute alpha_power in trial_alpha.csv.

Each participant contributes one mean(ON) - mean(OFF) in percentage points.
The primary two-sided permutation test shuffles labels of WHOLE verified
stimulation periods, maintaining two ON and two OFF periods per participant.
Additional paired sign and Wilcoxon signed-rank tests, normalized-effect
sensitivity and leave-one-participant-out permutation tests are reported.

Run from the repository root:
    python analysis/statistics/S02_absolute_alpha_block_permutation.py \\
        --subjects 101 102 103 --platform mac

Output: <PROJECT_ROOT>/derivatives/reports/group/
        absolute_alpha_block_permutation/absolute_alpha_block_permutation.pdf
and publication-quality permutation figures in figures/.
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
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
SUBJECT_DIR = HERE.parent / "subject" / "EEG_all_channels"
sys.path.insert(0, str(SUBJECT_DIR))
from pipeline_config import resolve_project_root, stage_path

FREQS = np.arange(2.0, 32.0, 0.5)
BASELINE = (-0.3, -0.1)
ATTENTION = (0.3, 0.8)
HALF_WIDTH = 1.0
ROI8 = ("PO3", "POz", "PO4", "O1", "Oz", "O2", "PO7", "PO8")


def arguments():
    """Read input, subject and reproducibility options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--peaks-csv", type=Path, default=None)
    parser.add_argument("--platform", choices=("mac", "bluebear"), default="mac")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n_permutations < 1:
        parser.error("--n-permutations must be positive")
    args.subjects = [s.removeprefix("sub-") for s in args.subjects]
    if len(args.subjects) != len(set(args.subjects)):
        parser.error("Duplicate participant IDs")
    return args


def alpha_trials(epochs, channels, peak, n_jobs):
    """Return both unbaselined and percent-baseline alpha per trial.

    Trial-level TFR is baseline corrected per sensor/frequency first.
    The result is then averaged over retained sensors, peak +/-1 Hz
    frequencies, and the attention interval. Trial order is unchanged.
    """
    tfr = epochs.copy().pick(channels).compute_tfr(
        method="multitaper", freqs=FREQS, n_cycles=FREQS / 2,
        time_bandwidth=2.0, use_fft=True, zero_mean=True,
        return_itc=False, average=False, decim=2,
        n_jobs=n_jobs, verbose=False,
    )
    freq = (tfr.freqs >= peak - HALF_WIDTH - 1e-9) & (
        tfr.freqs <= peak + HALF_WIDTH + 1e-9
    )
    times = (tfr.times >= ATTENTION[0]) & (tfr.times <= ATTENTION[1])
    base = (tfr.times >= BASELINE[0]) & (tfr.times <= BASELINE[1])
    if not freq.any() or not times.any() or not base.any():
        raise ValueError("Required baseline/alpha/attention TFR bins absent")
    baseline_power = tfr.data[..., base].mean(axis=-1)
    if np.any(baseline_power <= 0) or not np.isfinite(baseline_power).all():
        raise ValueError("Nonpositive or invalid baseline power")
    raw_values = tfr.data[:, :, freq, :][:, :, :, times].mean(axis=(1, 2, 3))
    tfr.apply_baseline(BASELINE, mode="percent")
    percent_values = 100.0 * tfr.data[:, :, freq, :][:, :, :, times].mean(
        axis=(1, 2, 3)
    )
    if not np.isfinite(raw_values).all() or not np.isfinite(percent_values).all():
        raise ValueError("Nonfinite alpha power")
    return percent_values, raw_values


def validated_subject(frame, subject, root, args, peak):
    """Match each CSV trial to the corresponding cleaned epoch index."""
    rows = frame.loc[frame.subject == subject].copy()
    if rows.empty:
        raise ValueError(f"sub-{subject}: absent from validated B03 CSV")
    if not rows.label_verified.astype(str).str.lower().eq("true").all():
        raise ValueError(f"sub-{subject}: unverified stimulation labels")
    if not np.isfinite(peak) or not 8 <= peak <= 12:
        raise ValueError(f"sub-{subject}: invalid OFF-derived peak")
    periods = [list() for _ in range(4)]
    raw_periods = [list() for _ in range(4)]
    labels = np.zeros(4, dtype=bool)
    channels_used = None
    for condition in ("no-stim", "stim"):
        path = stage_path(
            root, subject, args.session, args.task, args.run,
            condition, "clean", "epo",
        )
        epochs = mne.read_epochs(path, preload=True, verbose=False)
        cues = [c for c in ("cue_onset_right", "cue_onset_left")
                if c in epochs.event_id]
        if not cues:
            raise ValueError(f"sub-{subject}: missing cue epochs")
        epochs = epochs[cues]
        if channels_used is None:
            channels_used = [
                ch for ch in ROI8 if ch in epochs.ch_names
                and ch not in epochs.info["bads"]
            ]
        else:
            channels_used = [
                ch for ch in channels_used if ch in epochs.ch_names
                and ch not in epochs.info["bads"]
            ]
        # Defer computation until the common channel set is known.
        if condition == "no-stim":
            off_epochs = epochs
        else:
            on_epochs = epochs
    if not channels_used:
        raise ValueError(f"sub-{subject}: no shared good posterior sensors")
    for condition, epochs in (("no-stim", off_epochs), ("stim", on_epochs)):
        condition_rows = rows.loc[rows.source_condition == condition].copy()
        if len(condition_rows) != len(epochs):
            raise ValueError(f"sub-{subject}: CSV/epoch count mismatch for {condition}")
        if condition_rows.trial_index_in_condition.duplicated().any():
            raise ValueError(f"sub-{subject}: duplicate trial indices")
        indices = condition_rows.trial_index_in_condition.to_numpy(dtype=int)
        if not np.array_equal(np.sort(indices), np.arange(len(epochs))):
            raise ValueError(f"sub-{subject}: incomplete trial-index mapping")
        values, raw_values = alpha_trials(epochs, channels_used, peak, args.n_jobs)
        for row in condition_rows.itertuples(index=False):
            idx = int(row.trial_index_in_condition)
            period = int(row.stimulation_period) - 1
            if period not in range(4):
                raise ValueError(f"sub-{subject}: invalid period")
            is_on = str(row.dbs_on).lower() == "true"
            if is_on != (condition == "stim"):
                raise ValueError(f"sub-{subject}: ON/OFF condition mismatch")
            side = str(row.stimulation_side).strip().lower()
            if is_on != (side != "off"):
                raise ValueError(f"sub-{subject}: side/condition mismatch")
            periods[period].append(float(values[idx]))
            raw_periods[period].append(float(raw_values[idx]))
    sides = []
    for period in range(4):
        p = rows.loc[rows.stimulation_period == period + 1]
        if p.empty or p.dbs_on.nunique() != 1 or p.stimulation_side.nunique() != 1:
            raise ValueError(f"sub-{subject}: invalid stimulation period {period + 1}")
        labels[period] = str(p.dbs_on.iloc[0]).lower() == "true"
        sides.append(str(p.stimulation_side.iloc[0]).strip().lower())
    if sorted(sides) != ["off", "off", "stim_left", "stim_right"]:
        raise ValueError(f"sub-{subject}: invalid stimulation sequence")
    if any(not p for p in periods):
        raise ValueError(f"sub-{subject}: empty period")
    return ([np.asarray(p) for p in periods],
            [np.asarray(p) for p in raw_periods], labels, peak, channels_used)


def condition_means(periods, labels):
    """Pool all trials within each assigned condition (trial-weighted)."""
    on = np.concatenate([p for p, flag in zip(periods, labels) if flag])
    off = np.concatenate([p for p, flag in zip(periods, labels) if not flag])
    return float(on.mean()), float(off.mean())


def effect(periods, labels, normalized=False):
    """Difference of percent changes, or ratio of original absolute power.

    The normalized ratio NEVER uses baseline-corrected power.
    """
    on, off = condition_means(periods, labels)
    if not normalized:
        return on - off
    denominator = on + off
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("Unbaselined ON+OFF denominator must be positive")
    return (on - off) / denominator


def run_permutation(data, iterations, seed, normalized=False, statistic="signed"):
    """Two-sided period-label permutation with equal subject weighting."""
    effects = np.asarray([effect(raw, labels, normalized=True)
                          for _, raw, labels in data])
    aggregate = {"signed": np.mean, "squared": lambda x: np.mean(np.square(x)), "absolute": lambda x: np.mean(np.abs(x))}[statistic]
    observed = float(aggregate(effects))
    rng = np.random.default_rng(seed)
    null = np.empty(iterations)
    for index in range(iterations):
        null[index] = aggregate(np.asarray([
            effect(raw, rng.permutation(labels), normalized=True)
            for _, raw, labels in data
        ]))
    extreme = (np.abs(null) >= abs(observed)) if statistic == "signed" else (null >= observed)
    p = (1 + np.count_nonzero(extreme)) / (iterations + 1)
    return effects, observed, null, float(p)


def text_page(title, sections):
    """Render readable, wrapped report text to a PDF page."""
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    ax.text(0.045, 0.95, title, fontsize=16, weight="bold",
            va="top", transform=ax.transAxes)
    y = 0.88
    for heading, content in sections:
        ax.text(0.045, y, heading, fontsize=11.5, weight="bold",
                va="top", transform=ax.transAxes)
        y -= 0.045
        lines = textwrap.wrap(str(content), width=98)
        for line in lines:
            ax.text(0.045, y, line, fontsize=9.3,
                    va="top", transform=ax.transAxes)
            y -= 0.024
        y -= 0.045
    return fig


def distribution_page(null, observed, p, label, two_sided=True, explanation=None):
    """Plot two-sided null, observed statistic and critical thresholds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(null, bins=50, color="lightsteelblue")
    ax.axvline(observed, color="darkred", linewidth=2, label="Observed")
    threshold = np.quantile(np.abs(null) if two_sided else null, 0.95)
    if two_sided:
        ax.axvline(-threshold, color="gray", linestyle="--")
    ax.axvline(threshold, color="gray", linestyle="--",
               label="Two-sided 5% threshold" if two_sided else "Upper-tail 5% threshold")
    ax.set(title=f"{label}: observed={observed:.4g}, p={p:.5g}",
           xlabel=label, ylabel="Permutations")
    ax.legend()
    if explanation:
        fig.text(0.08, 0.12, textwrap.fill(explanation, width=105),
                 fontsize=9, va="top", ha="left")
        fig.subplots_adjust(bottom=0.34)
    else:
        fig.tight_layout()
    return fig


def main():
    """Produce primary, paired, and sensitivity results in one PDF."""
    args = arguments()
    root = resolve_project_root(args.platform, args.project_root)
    group = root / "derivatives" / "reports" / "group"
    csv_path = args.input_csv or (
        group / "EEG_posterior_channels_block_mapping" / "trial_alpha.csv"
    )
    frame = pd.read_csv(csv_path, dtype={"subject": str})
    required = {
        "subject", "source_condition", "trial_index_in_condition",
        "stimulation_period", "stimulation_side", "dbs_on",
        "alpha_peak_hz", "label_verified",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"B03 CSV missing columns: {sorted(missing)}")
    frame["subject"] = frame.subject.str.removeprefix("sub-")
    peaks_path = args.peaks_csv or group / "individual_alpha_peaks.csv"
    peaks_frame = pd.read_csv(peaks_path, dtype={"subject": str})
    if not {"subject", "peak_hz"}.issubset(peaks_frame.columns):
        raise ValueError("Peak CSV requires subject and peak_hz")
    peaks_frame["subject"] = peaks_frame.subject.str.removeprefix("sub-")
    if peaks_frame.subject.duplicated().any():
        raise ValueError("Duplicate individual peak IDs")
    peaks = dict(zip(peaks_frame.subject, pd.to_numeric(peaks_frame.peak_hz, errors="coerce")))
    for subject in args.subjects:
        if subject not in peaks or not np.isfinite(peaks[subject]):
            raise ValueError(f"Missing peak for sub-{subject}")
    data, included, details = [], [], []
    for subject in args.subjects:
        periods, raw_periods, labels, peak, channels = validated_subject(
            frame, subject, root, args, float(peaks[subject])
        )
        data.append((periods, raw_periods, labels))
        included.append(subject)
        on, off = condition_means(periods, labels)
        details.append((subject, peak, channels, on, off, on - off))
        print(f"sub-{subject}: ON={on:.4g}%, OFF={off:.4g}%, "
              f"ON-OFF={on-off:.4g} percentage points")
    effects, observed, null, p_perm = run_permutation(
        data, args.n_permutations, args.seed
    )
    _, sq_obs, sq_null, sq_p = run_permutation(data, args.n_permutations, args.seed, statistic="squared")
    _, abs_obs, abs_null, abs_p = run_permutation(data, args.n_permutations, args.seed, statistic="absolute")
    n_nonzero = int(np.count_nonzero(effects))
    if n_nonzero:
        wilcox = wilcoxon(effects, zero_method="wilcox",
                         alternative="two-sided", method="auto")
        w_stat, w_p = float(wilcox.statistic), float(wilcox.pvalue)
    else:
        w_stat, w_p = 0.0, 1.0

    loo = []
    if len(data) > 1:
        for index, subject in enumerate(included):
            remaining = data[:index] + data[index + 1:]
            _, statistic, _, p = run_permutation(
                remaining, args.n_permutations, args.seed, statistic="squared"
            )
            loo.append((subject, statistic, p))
    output = group / "absolute_alpha_block_permutation"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    report_path = output / "absolute_alpha_block_permutation.pdf"
    pages = [
        text_page("S02: participants and data", [
            ("Included participants", ", ".join("sub-" + x for x in included)),
            ("Trial mapping", str(csv_path)),
            ("Individual peak frequencies", str(peaks_path)),
            ("Analysis outcome",
             "Each participant: R=(mean raw ON power - mean raw OFF power) / "
             "(mean raw ON power + mean raw OFF power). Power is unbaselined, "
             "averaged over usable posterior sensors, individual OFF-derived "
             "alpha peak +/-1 Hz, and 0.3-0.8 s. Trial mapping uses B03; "
             "no baseline correction is used for any statistical test."),
        ]),
        text_page("S02: statistics", [
            ("PRIMARY: squared normalized ratio",
             f"Mean R squared={sq_obs:.6g}; upper-tail whole-period "
             f"permutation p={sq_p:.6g}."),
            ("Sensitivity: absolute normalized ratio",
             f"Mean absolute R={abs_obs:.6g}; upper-tail permutation "
             f"p={abs_p:.6g}."),
            ("Secondary: signed normalized ratio",
             f"Mean R={observed:.6g}; two-sided whole-period permutation "
             f"p={p_perm:.6g}."),
            ("Paired Wilcoxon signed-rank on participant ratios",
             f"W={w_stat:.6g}; two-sided p={w_p:.6g}. "
             "The signed-rank test detects a common signed shift, "
             "not heterogeneous magnitude."),
            ("Leave-one-out",
             "Repeat the PRIMARY squared-ratio permutation after omitting "
             "each participant. Results are tabulated below."),
            ("Assumptions",
             "Four stimulation periods per participant, two ON and two OFF. "
             "Whole-period label permutations assume exchangeability and "
             "do not model carryover or order effects."),
        ]),
    ]
    for start in range(0, len(details), 14):
        batch = details[start:start + 14]
        pages.append(text_page("Participant effects", [
            (f"sub-{s}", f"OFF={off:.4g}%, ON={on:.4g}%, "
             f"ON-OFF={diff:.4g} percentage points; peak={peak:g} Hz")
            for s, peak, _, on, off, diff in batch
        ]))
    primary_fig = distribution_page(
        sq_null, sq_obs, sq_p, "Mean squared normalized ratio",
        two_sided=False,
        explanation="PRIMARY: Participant ratios use only unbaselined "
        "ON and OFF power. Squaring captures effects of either sign. "
        "Blue: period-label permutation null; red: observed statistic; "
        "dashed: upper 5% null threshold.",
    )
    absolute_fig = distribution_page(
        abs_null, abs_obs, abs_p, "Mean absolute normalized ratio",
        two_sided=False,
        explanation="SENSITIVITY: Absolute participant ratios retain "
        "either-direction effects, with less emphasis on outliers than "
        "squaring. Blue: null; red: observed; dashed: upper 5% threshold.",
    )
    signed_fig = distribution_page(
        null, observed, p_perm, "Mean signed normalized ratio",
        explanation="SECONDARY: Participant ratios retain their signs, "
        "so opposite directions may cancel. Blue: permutation null; "
        "red: observed; dashed: two-sided 5% null thresholds.",
    )
    for name, fig in (
        ("unbaselined_ratio_squared_primary", primary_fig),
        ("unbaselined_ratio_absolute_sensitivity", absolute_fig),
        ("unbaselined_ratio_signed_secondary", signed_fig),
    ):
        for ext in ("svg", "tiff"):
            fig.savefig(figures / f"{name}.{ext}",
                        dpi=600, bbox_inches="tight")
    loo_figures = []
    for start in range(0, len(loo), 16):
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis("off")
        ax.set_title("Leave-one-out: squared unbaselined normalized ratio")
        table = ax.table(
            cellText=[[f"sub-{sub}", f"{stat:.5f}", f"{p:.5g}"]
                      for sub, stat, p in loo[start:start + 16]],
            colLabels=["Omitted subject", "Mean squared ratio",
                       "Two-sided permutation p"],
            colWidths=[0.22, 0.48, 0.30],
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
        loo_figures.append(fig)

    paired_fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                                   constrained_layout=True)
    axes[0].hist(effects, bins=min(12, max(3, len(effects))))
    axes[0].axvline(0, color="black", linestyle="--")
    axes[0].set(title="Individual normalized ON-OFF ratios", ylabel="Participants")
    axes[1].axhline(0, color="black", linestyle="--")
    order = np.argsort(np.abs(effects))
    axes[1].scatter(np.arange(1, len(effects) + 1), effects[order])
    axes[1].set(
        title=f"Wilcoxon signed-rank on ratios, W={w_stat:g}, p={w_p:.5g}",
        xlabel="Participants sorted by absolute effect",
        ylabel="Unbaselined (ON - OFF)/(ON + OFF)",
    )
    for ext in ("svg", "tiff"):
        paired_fig.savefig(figures / f"paired_sign_wilcoxon.{ext}",
                           dpi=600, bbox_inches="tight")
    with PdfPages(report_path) as pdf:
        for page in pages[:2]:
            pdf.savefig(page)
            plt.close(page)
        pdf.savefig(primary_fig)
        plt.close(primary_fig)
        pdf.savefig(absolute_fig)
        plt.close(absolute_fig)
        pdf.savefig(signed_fig)
        plt.close(signed_fig)

        pdf.savefig(paired_fig)
        plt.close(paired_fig)
        for fig in loo_figures:
            pdf.savefig(fig)
            plt.close(fig)
        for page in pages[2:]:
            pdf.savefig(page)
            plt.close(page)
    print(f"Squared ratio primary p={sq_p:.6g}; absolute ratio sensitivity p={abs_p:.6g}; "
          f"Signed permutation p={p_perm:.6g}; "
          f"Wilcoxon signed-rank p={w_p:.6g}; sensitivity p={norm_p:.6g}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
