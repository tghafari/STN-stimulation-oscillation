#!/usr/bin/env python3
"""Create presentation-quality permutation plots from S02 output CSV files.

Example:
python analysis/statistics/S03_plot_alpha_permutations.py --group-dir /path/to/absolute_alpha_block_permutation

Requires participant_effects.csv and permutation_null.csv from S02.
No permutations are rerun; the observed statistic is reconstructed from
participant effects and the p-value from the saved null distribution.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_distribution(values, observed, p_value, title, xlabel, output):
    """Export the same labelled histogram as PDF, PNG and SVG."""
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    ax.hist(values, bins=50, color="#a7c4d6", edgecolor="white")
    ax.axvline(observed, color="#b52330", linewidth=2.5,
               label="Observed group mean")
    ax.axvline(0, color="#444444", linestyle=":", linewidth=1)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of permutations")
    ax.legend(frameon=False)
    ax.text(0.98, 0.95, f"Two-sided p = {p_value:.4g}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11)
    fig.tight_layout()
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def main():
    """Read S02 outputs, verify statistics and generate figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.group_dir.expanduser().resolve()
    participants = pd.read_csv(folder / "participant_effects.csv")
    null = pd.read_csv(folder / "permutation_null.csv")
    observed = float(participants["on_minus_off"].mean()) * 1e12
    distribution = null["permuted_group_effect"].to_numpy() * 1e12
    if not np.isfinite(distribution).all():
        raise ValueError("Nonfinite values in permutation null")
    p_value = (1 + np.count_nonzero(
        np.abs(distribution) >= abs(observed)
    )) / (len(distribution) + 1)
    plot_distribution(
        distribution, observed, p_value,
        "STN stimulation: absolute posterior alpha during attention",
        "Mean DBS ON - OFF (microvolt squared)",
        folder / "absolute_alpha_permutation_figure",
    )
    percentage_file = folder / "permutation_null_normalised.csv"
    effects_file = folder / "participant_normalised_effects.csv"
    if percentage_file.exists() and effects_file.exists():
        effects = pd.read_csv(effects_file)
        percent_null = pd.read_csv(percentage_file)
        observed_percent = float(effects["normalised_on_minus_off"].mean())
        values = percent_null["permuted_group_normalised"].to_numpy()
        percent_p = (1 + np.count_nonzero(
            np.abs(values) >= abs(observed_percent)
        )) / (len(values) + 1)
        plot_distribution(
            values, observed_percent, percent_p,
            "STN stimulation: symmetric normalised alpha change",
            "Mean (DBS ON - OFF) / (DBS ON + OFF)",
            folder / "normalised_alpha_permutation_figure",
        )
    print(f"Figures saved to {folder}")


if __name__ == "__main__":
    main()
