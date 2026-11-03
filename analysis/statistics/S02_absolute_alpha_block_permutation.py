#!/usr/bin/env python3
"""
==================================================================
Block-label permutation test of absolute posterior alpha power.

Each participant contributes one ON-minus-OFF difference, calculated from
trial-level absolute multitaper power in ATTENTION_WINDOW. Stimulation labels
are shuffled across the four CONFIRMED stimulation periods (two OFF and two
ON); each period comprises two consecutive task blocks. All retained trials
within a period move together. The number of ON/OFF periods is preserved.

Requires cleaned cue epochs, original BIDS events, the preprocessing crop-time
JSON, a confirmed block CSV, and the existing individual-alpha-peaks CSV.
Never reconstruct block identities by guessing from epoch sample positions:
the existing validated BIDS-to-cleaned-epochs mapping is used.

Run from repository root:
python analysis/statistics/S02_absolute_alpha_block_permutation.py \\
    --subjects 101 102 103 

The output includes a PDF with methods, observed statistic, null histogram,
participant effects, and a CSV with per-participant results.
==================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
SUBJECT_DIR = ANALYSIS / "subject"
for folder in (SUBJECT_DIR / "EEG_all_channels", SUBJECT_DIR / "EEG_posterior_channels" / "block_preparation"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from pipeline_config import resolve_project_root, stage_path
from B01_review_bids_blocks import events_path
from B02_validate_epoch_mapping import map_condition_epochs

# Edit these values to change the analysis without changing functions.
ATTENTION_WINDOW = (0.3, 0.8)
ALPHA_PEAK_RANGE = 1.0  # Hz on each side of each participant's CSV peak
FREQS = np.arange(2.0, 32.0, 0.5)
POSTERIOR = ("PO3", "POz", "PO4", "O1", "Oz", "O2", "PO7", "PO8")
N_PERMUTATIONS = 10000
RANDOM_SEED = 42


def arguments():
    """Parse paths, participant identifiers and reproducibility settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--input-csv", type=Path, default=None, help="Validated trial-level alpha CSV produced by G04.")
    parser.add_argument("--peaks-csv", type=Path, default=None)
    parser.add_argument("--platform", choices=("mac", "bluebear"), default="mac")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if args.n_permutations < 1:
        parser.error("--n-permutations must be positive")
    if ATTENTION_WINDOW[0] >= ATTENTION_WINDOW[1]:
        parser.error("ATTENTION_WINDOW must have increasing bounds")
    return args


def read_peaks(path):
    """Read G03 peak CSV; reject missing/duplicate/invalid participants."""
    frame = pd.read_csv(path, dtype={"subject": str})
    required = {"subject", "peak_hz"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Peak CSV requires columns: {sorted(required)}")
    frame["subject"] = frame["subject"].str.removeprefix("sub-")
    frame["peak_hz"] = pd.to_numeric(frame["peak_hz"], errors="coerce")
    if frame["subject"].duplicated().any():
        raise ValueError("Duplicate participant in peak CSV")
    return frame.set_index("subject")["peak_hz"].to_dict()


def crop_intervals(values):
    """Convert preprocessing crop boundaries into interval pairs."""
    values = [float(value) for value in values]
    if len(values) not in (2, 4):
        raise ValueError(f"Expected two or four crop boundaries: {values}")
    return list(zip(values[::2], values[1::2]))


def cleaned_epochs(root, subject, condition, args):
    """Load cleaned epochs and retain only left/right cue events."""
    path = stage_path(
        root, subject, args.session, args.task, args.run,
        condition, "clean", "epo"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    epochs = mne.read_epochs(path, preload=True, verbose=False)
    cues = [
        name for name in ("cue_onset_left", "cue_onset_right")
        if name in epochs.event_id
    ]
    if not cues:
        raise ValueError(f"Missing cue events: {path}")
    return epochs[cues]


def confirmed_periods(review, subject):
    """Extract four verified periods, each containing two task blocks."""
    rows = review.loc[
        (review["subject"].str.removeprefix("sub-") == subject)
        & review["confirmed"].str.lower().isin(("yes", "y", "true", "1"))
    ].sort_values("block_index_chronological")
    if len(rows) != 4:
        raise ValueError(f"sub-{subject}: expected four confirmed periods")
    labels = rows["confirmed_condition"].str.strip().str.lower().tolist()
    if labels.count("off") != 2 or labels.count("stim_left") != 1 or labels.count("stim_right") != 1:
        raise ValueError(f"sub-{subject}: unexpected verified conditions {labels}")
    task_blocks = []
    for _, row in rows.iterrows():
        task_blocks.append((
            float(row["first_task_block_onset_s"]),
            float(row["first_task_block_offset_s"])
        ))
        task_blocks.append((
            float(row["second_task_block_onset_s"]),
            float(row["second_task_block_offset_s"])
        ))
    return np.asarray([label != "off" for label in labels]), task_blocks


def trial_alpha(epochs, channels, peak, args):
    """Return one absolute linear alpha-power value per retained trial.

    No baseline correction, logarithm or trial-wise normalization occurs.
    Average selected channels, peak +/- ALPHA_PEAK_RANGE frequency bins,
    and ATTENTION_WINDOW time points independently for each trial.
    """
    tfr = epochs.copy().pick(channels).compute_tfr(
        method="multitaper", freqs=FREQS, n_cycles=FREQS / 2,
        time_bandwidth=2.0, use_fft=True, zero_mean=True,
        return_itc=False, average=False, decim=2,
        n_jobs=args.n_jobs, verbose=False
    )
    freq_mask = (
        (tfr.freqs >= peak - ALPHA_PEAK_RANGE - 1e-9)
        & (tfr.freqs <= peak + ALPHA_PEAK_RANGE + 1e-9)
    )
    time_mask = (
        (tfr.times >= ATTENTION_WINDOW[0])
        & (tfr.times <= ATTENTION_WINDOW[1])
    )
    if not freq_mask.any() or not time_mask.any():
        raise ValueError("No TFR bins in chosen alpha/attention window")
    return tfr.data[:, :, freq_mask, :][:, :, :, time_mask].mean(axis=(1, 2, 3))


def participant_data(root, subject, peak, review, crops, args):
    """Validate all trial-to-block mappings, then return four period arrays."""
    labels, task_blocks = confirmed_periods(review, subject)
    off = cleaned_epochs(root, subject, "no-stim", args)
    on = cleaned_epochs(root, subject, "stim", args)
    channels = [
        channel for channel in POSTERIOR
        if channel in off.ch_names and channel in on.ch_names
        and channel not in off.info["bads"] and channel not in on.info["bads"]
    ]
    if not channels:
        raise ValueError(f"sub-{subject}: no common good posterior channels")
    bids = pd.read_csv(
        events_path(root, subject, args.session, args.task, args.run), sep="\t"
    )
    key = f"sub-{subject}"
    if key not in crops:
        raise ValueError(f"Missing crop intervals for {key}")
    periods = [None] * 4
    for condition, epochs in (("no-stim", off), ("stim", on)):
        mapping = map_condition_epochs(
            epochs, bids, crop_intervals(crops[key][condition]),
            task_blocks, condition
        )
        block_indices = np.asarray(mapping["matched_task_block_indices"])
        values = trial_alpha(epochs, channels, peak, args)
        period_indices = block_indices // 2
        expected = 0 if condition == "no-stim" else 1
        if np.any(labels[period_indices] != expected):
            raise ValueError(f"sub-{subject}: epoch/condition block mismatch")
        for period in range(4):
            selected = values[period_indices == period]
            if selected.size:
                if periods[period] is not None:
                    raise ValueError("Duplicate assignment to a stimulation period")
                periods[period] = selected
    if any(value is None or len(value) == 0 for value in periods):
        raise ValueError(f"sub-{subject}: empty/unmapped stimulation period")
    return periods, labels, len(channels)


def effect(periods, labels):
    """Calculate trial-weighted ON minus OFF means within a participant."""
    on = np.concatenate([
        values for values, is_on in zip(periods, labels) if is_on
    ])
    off = np.concatenate([
        values for values, is_on in zip(periods, labels) if not is_on
    ])
    return float(on.mean() - off.mean())


def permutation_distribution(data, iterations, seed):
    """Shuffle four whole-period labels independently within each subject."""
    rng = np.random.default_rng(seed)
    null = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        differences = []
        for periods, labels in data:
            permuted = rng.permutation(labels)
            differences.append(effect(periods, permuted))
        null[iteration] = np.mean(differences)
    return null


def methods_page(n_subjects, n_perm):
    """Write a readable PDF methods page including analysis steps 4, 5, 6."""
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    sections = [
        ("Data and alpha extraction",
         f"{n_subjects} participants. Absolute multitaper power is computed "
         "for every cleaned cue trial using the retained common posterior "
         "channels. The individual alpha peak is read from the prior G03 "
         "CSV (selected from original DBS OFF), and peak +/- "
         f"{ALPHA_PEAK_RANGE:g} Hz is used for both conditions. "
         f"Attention interval: {ATTENTION_WINDOW[0]:g}-"
         f"{ATTENTION_WINDOW[1]:g} s. No baseline correction."),
        ("4. Observed stimulation effect",
         "Within each participant, concatenate all trials assigned ON and "
         "all trials assigned OFF. Average absolute alpha power across trials "
         "in each condition; calculate D_i = mean(ON) - mean(OFF). "
         "The group statistic T_observed is the mean of participant D_i, "
         "so each participant contributes equally."),
        ("5. Block-label permutations",
         "Four stimulation periods per participant were obtained from "
         "the validated G04 trial-level dataset. The original block assignments "
         "and confirmed stimulation labels were preserved. Whole stimulation-period "
         "labels were permuted within participants, maintaining two ON and two OFF periods. " 
         "All trials stay with "
         "their original period. Recalculate D_i and their group mean. "
         f"Repeat {n_perm:,} times using the recorded random seed."),
        ("6. Permutation significance",
         "Use a two-sided statistic: p = (1 + number of permutations "
         "with abs(T_perm) >= abs(T_observed)) / (N_perm + 1). "
         "A negative observed statistic indicates reduced absolute "
         "alpha power during stimulation. The null histogram and "
         "observed statistic are plotted on the following page."),
        ("Interpretation and limitations",
         "Inference is conditional on the OFF-defined alpha peaks, "
         "which are held fixed across permutations. Validity of "
         "block-label randomization requires exchangeability under "
         "the actual experimental randomization; carryover or "
         "time-order effects are not modelled. This test is not "
         "a trial-level label shuffle."),
    ]
    y = 0.94
    for heading, body in sections:
        ax.text(0.04, y, heading, fontsize=12, fontweight="bold",
                va="top", transform=ax.transAxes)
        y -= 0.042
        wrapped = textwrap.fill(body, width=100)
        ax.text(0.04, y, wrapped, fontsize=9.5, va="top",
                linespacing=1.35, transform=ax.transAxes)
        y -= 0.029 * (wrapped.count("\n") + 1) + 0.048
    fig.tight_layout()
    return fig


def results_page(observed, null, pvalue, subjects, differences):
    """Plot permutation distribution and participant-specific differences."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.7, 8.3))
    ax1.hist(null, bins=55, color="lightsteelblue", edgecolor="white")
    ax1.axvline(observed, color="darkred", lw=2, label="Observed")
    ax1.axvline(-abs(observed), color="gray", ls="--", lw=1)
    ax1.axvline(abs(observed), color="gray", ls="--", lw=1)
    ax1.set(
        title=f"Block permutation | observed={observed:.4g}, two-sided p={pvalue:.5f}",
        xlabel="Group mean (DBS ON - DBS OFF), absolute alpha power",
        ylabel="Permutations"
    )
    ax1.legend()
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.scatter(range(len(subjects)), differences, color="steelblue", s=28)
    ax2.set(
        title="Participant-level ON - OFF effects",
        xlabel="Participant", ylabel="Absolute alpha-power difference"
    )
    ax2.set_xticks(range(len(subjects)), subjects, rotation=65)
    fig.tight_layout()
    return fig

def main():
    """Run block-label permutations using validated G04 trial data.

    Each participant contributes one DBS ON-minus-OFF difference.
    The permutation test shuffles entire stimulation periods, not
    individual trials. Each participant has four stimulation periods:
    two OFF, one left STN ON, and one right STN ON.

    Alpha power has already been calculated by G04 without baseline
    correction, using the participant's individual alpha peak.
    """
    args = arguments()
    root = resolve_project_root(args.platform, args.project_root)

    input_csv = args.input_csv or (
        root
        / "derivatives"
        / "reports"
        / "group"
        / "EEG_posterior_channels_block_mapping"
        / "trial_alpha.csv"
    )

    # Load validated trial-level absolute alpha power from G04.
    data_frame = pd.read_csv(
        input_csv,
        dtype={"subject": str},
    )

    required_columns = {
        "subject",
        "stimulation_period",
        "stimulation_side",
        "dbs_on",
        "alpha_power",
        "alpha_peak_hz",
        "label_verified",
    }

    missing_columns = required_columns - set(data_frame.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data_frame["subject"] = (
        data_frame["subject"].str.removeprefix("sub-")
    )

    subjects = [
        subject.removeprefix("sub-")
        for subject in args.subjects
    ]

    if len(subjects) != len(set(subjects)):
        raise ValueError("Duplicate participant IDs supplied.")

    participant_data = []
    results = []

    # Calculate one observed ON-minus-OFF effect per participant.
    for subject in subjects:
        subject_data = data_frame.loc[
            data_frame["subject"] == subject
        ]

        if subject_data.empty:
            raise ValueError(
                f"Participant {subject} is missing from G04 data."
            )

        verified = (
            subject_data["label_verified"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

        if not verified.all():
            raise ValueError(
                f"Participant {subject} has unverified labels."
            )

        if subject_data["alpha_peak_hz"].nunique() != 1:
            raise ValueError(
                f"Participant {subject} has inconsistent alpha peaks."
            )

        periods = []
        labels = []

        for period_number in range(1, 5):
            period_data = subject_data.loc[
                subject_data["stimulation_period"]
                == period_number
            ]

            if period_data.empty:
                raise ValueError(
                    f"Participant {subject}: "
                    f"period {period_number} is empty."
                )

            if period_data["dbs_on"].nunique() != 1:
                raise ValueError(
                    f"Participant {subject}: "
                    f"mixed stimulation labels in period "
                    f"{period_number}."
                )

            if period_data["stimulation_side"].nunique() != 1:
                raise ValueError(
                    f"Participant {subject}: "
                    f"mixed stimulation sides in period "
                    f"{period_number}."
                )

            alpha_values = period_data[
                "alpha_power"
            ].to_numpy(dtype=float)

            if not np.isfinite(alpha_values).all():
                raise ValueError(
                    f"Participant {subject}: "
                    "non-finite alpha power."
                )

            stimulation_side = str(
                period_data["stimulation_side"].iloc[0]
            ).strip().lower()

            is_on = (
                str(period_data["dbs_on"].iloc[0])
                .strip()
                .lower()
                == "true"
            )

            if is_on != (stimulation_side != "off"):
                raise ValueError(
                    f"Participant {subject}: inconsistent "
                    "stimulation condition."
                )

            periods.append(alpha_values)
            labels.append(is_on)

        labels = np.asarray(labels, dtype=bool)

        if labels.sum() != 2:
            raise ValueError(
                f"Participant {subject}: expected "
                "two ON and two OFF periods."
            )

        # effect() calculates trial-weighted ON minus OFF power.
        observed_difference = effect(periods, labels)

        n_on = sum(
            len(values)
            for values, is_on in zip(periods, labels)
            if is_on
        )

        n_off = sum(
            len(values)
            for values, is_on in zip(periods, labels)
            if not is_on
        )

        results.append({
            "subject": subject,
            "peak_hz": float(
                subject_data["alpha_peak_hz"].iloc[0]
            ),
            "n_on": n_on,
            "n_off": n_off,
            "on_minus_off": observed_difference,
        })

        participant_data.append((periods, labels))

        print(
            f"sub-{subject}: "
            f"ON - OFF = {observed_difference:.6g}"
        )

    # Average participant-level differences equally.
    differences = np.asarray([
        result["on_minus_off"]
        for result in results
    ])

    observed_statistic = float(differences.mean())

    # Randomly permute whole stimulation-period labels
    # independently within each participant.
    null_distribution = permutation_distribution(
        participant_data,
        args.n_permutations,
        args.seed,
    )

    # Two-sided Monte Carlo permutation p-value.
    p_value = (
        1
        + np.count_nonzero(
            np.abs(null_distribution)
            >= abs(observed_statistic)
        )
    ) / (args.n_permutations + 1)

    # Save statistical results and PDF report.
    output_dir = (
        root
        / "derivatives"
        / "reports"
        / "group"
        / "absolute_alpha_block_permutation"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = (
        output_dir / "absolute_alpha_block_permutation.pdf"
    )

    with PdfPages(report_path) as pdf:
        figures = [
            methods_page(
                len(subjects),
                args.n_permutations,
            ),
            results_page(
                observed_statistic,
                null_distribution,
                p_value,
                subjects,
                differences,
            ),
        ]

        for figure in figures:
            pdf.savefig(figure)
            plt.close(figure)

    print("\nPermutation analysis complete.")
    print(f"Participants: {len(subjects)}")
    print(f"Observed ON - OFF: {observed_statistic:.6g}")
    print(f"Two-sided p-value: {p_value:.6g}")
    print(f"Report: {report_path}")



if __name__ == "__main__":
    main()
