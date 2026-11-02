#!/usr/bin/env python3
"""Optional sensitivity tests for S02's validated G04 alpha data.

Symmetric normalised effect: (mean ON - mean OFF) / (mean ON + mean OFF).
Both ON and OFF means in the denominator are recalculated for EVERY permutation. Whole
stimulation-period labels are shuffled within each participant, keeping
two ON and two OFF periods. Participants contribute equally.

Leave-one-out: rerun the original absolute-power permutation test after
omitting each participant. These are influence checks, not confirmatory
independent tests.

Usage:
python analysis/statistics/S02_sensitivity_alpha.py --subjects 101 102 103
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    pd.DataFrame(rows).to_csv(
        out / "leave_one_out_absolute.csv", index=False
    )
    print(f"Normalised (ON-OFF)/(ON+OFF) = {observed:.5g}, p = {p_value:.5g}")
    print(f"Sensitivity outputs: {out}")


if __name__ == "__main__":
    main()
