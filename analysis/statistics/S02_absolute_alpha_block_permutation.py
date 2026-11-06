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
from scipy.stats import binomtest, wilcoxon

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
    if rows.alpha_peak_hz.nunique() != 1:
        raise ValueError(f"sub-{subject}: inconsistent alpha peaks")
    if not np.isclose(float(rows.alpha_peak_hz.iloc[0]), peak, atol=0.26):
        raise ValueError("Individual peak CSV disagrees with B03; rerun B03")
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


def run_permutation(data, iterations, seed, normalized=False):
    """Two-sided period-label permutation with equal subject weighting."""
    effects = np.asarray([effect(raw if normalized else percent, labels, normalized)
                          for percent, raw, labels in data])
    observed = float(effects.mean())
    rng = np.random.default_rng(seed)
    null = np.empty(iterations)
    for index in range(iterations):
        null[index] = np.mean([
            effect(raw if normalized else percent, rng.permutation(labels), normalized)
            for percent, raw, labels in data
        ])
    p = (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (iterations + 1)
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


def distribution_page(null, observed, p, label):
    """Plot two-sided null, observed statistic and critical thresholds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(null, bins=50, color="lightsteelblue")
    ax.axvline(observed, color="darkred", linewidth=2, label="Observed")
    threshold = np.quantile(np.abs(null), 0.95)
    ax.axvline(-threshold, color="gray", linestyle="--")
    ax.axvline(threshold, color="gray", linestyle="--",
               label="Two-sided 5% threshold")
    ax.set(title=f"{label}: observed={observed:.4g}, p={p:.5g}",
           xlabel=label, ylabel="Permutations")
    ax.legend()
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
    n_positive = int(np.count_nonzero(effects > 0))
    n_negative = int(np.count_nonzero(effects < 0))
    n_nonzero = n_positive + n_negative
    sign_p = (float(binomtest(n_positive, n_nonzero, 0.5).pvalue)
              if n_nonzero else 1.0)
    if n_nonzero:
        wilcox = wilcoxon(effects, zero_method="wilcox",
                         alternative="two-sided", method="auto")
        w_stat, w_p = float(wilcox.statistic), float(wilcox.pvalue)
    else:
        w_stat, w_p = 0.0, 1.0
    norm_effects, norm_observed, norm_null, norm_p = run_permutation(
        data, args.n_permutations, args.seed, normalized=True
    )
    loo = []
    if len(data) > 1:
        for index, subject in enumerate(included):
            remaining = data[:index] + data[index + 1:]
            _, statistic, _, p = run_permutation(
                remaining, args.n_permutations, args.seed
            )
            loo.append((subject, statistic, p))
    output = group / "absolute_alpha_block_permutation"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    report_path = output / "absolute_alpha_block_permutation.pdf"
    pages = [
        text_page("S02: subjects and analysis details", [
            ("Included subjects (n=" + str(len(included)) + ")",
             ", ".join("sub-" + s for s in included)),
            ("Input", str(csv_path)),
            ("Individual peaks CSV", str(peaks_path)),
            ("Participant-specific alpha peaks and retained sensors",
             "; ".join(f"sub-{s}: {peak:g} Hz, {','.join(chs)}"
                       for s, peak, chs, _, _, _ in details)),
            ("Outcome and baseline",
             "ON and OFF are percent-baseline corrected (-0.3 to -0.1 s) "
             "per trial/sensor. Their difference is in percentage points. "
             "The normalized (ON-OFF)/(ON+OFF) sensitivity uses original "
             "UNBASELINED TFR power, with no baseline correction. "
             "Both use peak +/-1 Hz and 0.3-0.8 s. B03 alpha_power is not reused."),
        ]),
        text_page("S02: tests and interpretation", [
            ("Primary period-label permutation",
             f"Two-sided, {args.n_permutations:,} permutations, seed {args.seed}; "
             f"group mean ON-OFF = {observed:.5g} percentage points; p={p_perm:.6g}. "
             "Each subject contributes equally; trials stay attached to one "
             "of four verified stimulation periods; exactly two are ON."),
            ("Exact sign test",
             f"Positive={n_positive}, negative={n_negative}, ties="
             f"{len(effects)-n_nonzero}; two-sided binomial p={sign_p:.6g}."),
            ("Paired Wilcoxon signed-rank test",
             f"W={w_stat:.6g}; two-sided p={w_p:.6g}. "
             "Signed-rank is used instead of rank-sum because ON and OFF "
             "measurements are paired within participant."),
            ("Sensitivity: unbaselined normalized effect",
             f"Mean (raw ON-raw OFF)/(raw ON+raw OFF)={norm_observed:.5g}; "
             f"two-sided period permutation p={norm_p:.6g}. "
             "Both numerator and denominator use ORIGINAL, UNBASELINED "
             "absolute power. No percent-baseline values enter this ratio."),
            ("Interpretation",
             "These tests address participant-level cue-related alpha change. "
             "Permutation inference assumes stimulation-period exchangeability; "
             "order and carryover are not modelled. Sign and signed-rank "
             "tests are complementary paired analyses, not independent replications."),
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
        null, observed, p_perm, "ON - OFF (percentage points)"
    )
    normalized_fig = distribution_page(
        norm_null, norm_observed, norm_p, "Unbaselined (ON-OFF)/(ON+OFF)"
    )
    for name, fig in (("baseline_percent_permutation", primary_fig),
                      ("normalized_sensitivity_permutation", normalized_fig)):
        for extension in ("svg", "tiff"):
            fig.savefig(figures / f"{name}.{extension}",
                        dpi=600, bbox_inches="tight")
    loo_figures = []
    for start in range(0, len(loo), 16):
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis("off")
        ax.set_title("Leave-one-out: baseline-corrected ON - OFF")
        table = ax.table(
            cellText=[[f"sub-{sub}", f"{stat:.5f}", f"{p:.5g}"]
                      for sub, stat, p in loo[start:start + 16]],
            colLabels=["Omitted subject", "Mean difference (percentage points)",
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
    axes[0].bar(["Negative", "Zero", "Positive"],
                [n_negative, len(effects) - n_nonzero, n_positive])
    axes[0].set(title=f"Sign test, two-sided p={sign_p:.5g}",
                ylabel="Number of participants")
    axes[1].axhline(0, color="black", linestyle="--")
    order = np.argsort(np.abs(effects))
    axes[1].scatter(np.arange(1, len(effects) + 1), effects[order])
    axes[1].set(
        title=f"Wilcoxon signed-rank, W={w_stat:g}, p={w_p:.5g}",
        xlabel="Participants sorted by absolute effect",
        ylabel="Baseline-corrected ON - OFF (percentage points)",
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
        pdf.savefig(normalized_fig)
        plt.close(normalized_fig)
        pdf.savefig(paired_fig)
        plt.close(paired_fig)
        for fig in loo_figures:
            pdf.savefig(fig)
            plt.close(fig)
        for page in pages[2:]:
            pdf.savefig(page)
            plt.close(page)
    print(f"Permutation p={p_perm:.6g}; sign p={sign_p:.6g}; "
          f"Wilcoxon signed-rank p={w_p:.6g}; sensitivity p={norm_p:.6g}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
