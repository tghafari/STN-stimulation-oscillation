#!/usr/bin/env python3
"""
======================================================================
Exploratory alpha/theta frequency-time cluster permutation analysis.

Test two visually selected, post-hoc windows:
  alpha: individual OFF-derived peak -2.5 to +1 Hz, 0.1-0.7 s;
  theta: absolute 4-5.5 Hz, 0.1-0.4 s.

For each participant and sensor, compute trial-averaged linear multitaper
power, then separately baseline-correct ON and OFF before subtracting.
The baseline is -0.3 to -0.1 s, applied after trial averaging. Alpha is
aligned by each participant's OFF-derived peak; theta remains in Hz.
Within-participant sensor averages precede ROI-level tests.

Two-sided, one-sample cluster-mass sign-flip permutations test whether
the mean baseline-corrected ON-minus-OFF contrast differs from zero, correcting across the
frequency x time search space of each window. BH-FDR is additionally
applied across the family of all sensor/ROI x alpha/theta tests, using
each test's smallest cluster-corrected p (1 if no cluster). The windows
were selected after visual inspection: results are EXPLORATORY, not
independent confirmatory evidence.

One PDF is written. The script never modifies cleaned EEG files.

run:
    python analysis/statistics/S01_alpha_theta_targeted_clusters.py \
        --subjects 101 102 103 105 106 107 108 112 113 114 115 116 118 120 121 123
=======================================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE.parent / "subject" / "EEG_all_channels"
sys.path.insert(0, str(CONFIG_DIR))
from pipeline_config import resolve_project_root, stage_path  # noqa: E402

FREQS = np.arange(2.0, 32.0, 0.5)
ROI8 = ("PO3", "POz", "PO4", "O1", "Oz", "O2", "PO7", "PO8")
ROI3 = ("PO3", "POz", "PO4")
WINDOWS = {
    "alpha": {"freq": (-2.5, 1.0), "time": (0.1, 0.7)},
    "theta": {"freq": (4.0, 5.5), "time": (0.1, 0.4)},
}


def arguments():
    """Parse dataset, permutation and report settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--platform", choices=("mac", "bluebear"), default="mac")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--peaks-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-permutations", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.subjects = [str(s).removeprefix("sub-") for s in args.subjects]
    if len(args.subjects) != len(set(args.subjects)):
        parser.error("Duplicate subjects")
    if args.n_permutations < 2:
        parser.error("At least two permutations required")
    return args


def read_subject(root, subject, args):
    """Read only cue epochs and identify good shared EEG sensors."""
    epochs = {}
    for condition in ("stim", "no-stim"):
        path = stage_path(
            root, subject, args.session, args.task, args.run,
            condition, "clean", "epo",
        )
        ep = mne.read_epochs(path, preload=True, verbose=False)
        cues = [c for c in ("cue_onset_left", "cue_onset_right")
                if c in ep.event_id]
        if not cues:
            raise ValueError(f"sub-{subject}: no cue events for {condition}")
        epochs[condition] = ep[cues]
    good = [
        channel for channel in epochs["stim"].copy().pick("eeg").ch_names
        if channel in epochs["no-stim"].ch_names
        and channel not in epochs["stim"].info["bads"]
        and channel not in epochs["no-stim"].info["bads"]
    ]
    return epochs, good


def power(epochs, channels, n_jobs):
    """Trial-average multitaper power, without baseline correction."""
    tfr = epochs.copy().pick(channels).compute_tfr(
        method="multitaper", freqs=FREQS, n_cycles=FREQS / 2,
        time_bandwidth=2.0, use_fft=True, zero_mean=True,
        return_itc=False, average=True, decim=2,
        n_jobs=n_jobs, verbose=False,
    )
    return tfr.data, tfr.times


def baseline_percent(values, times):
    """Correct trial-averaged power using each sensor/frequency baseline.

    Each stimulation condition is corrected independently. The
    reference is mean power from -0.3 to -0.1 seconds.
    """
    baseline = (times >= -0.3) & (times <= -0.1)
    if not baseline.any():
        raise ValueError("No baseline samples")
    reference = values[..., baseline].mean(axis=-1, keepdims=True)
    if not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError("Invalid baseline reference power")
    return 100.0 * (values / reference - 1.0)


def contrast(on, off):
    """Subtract separately baseline-corrected ON and OFF power."""
    if not np.isfinite(on).all() or not np.isfinite(off).all():
        raise ValueError("Nonfinite baseline-corrected power")
    return on - off


def interpolate_alpha(values, peak):
    """Align a participant's frequency axis to their individual peak."""
    relative = np.arange(-2.5, 1.0 + 0.01, 0.5)
    absolute = relative + peak
    if absolute.min() < FREQS.min() or absolute.max() > FREQS.max():
        raise ValueError(f"Peak {peak} Hz lacks required alpha coverage")
    aligned = np.stack([
        np.interp(absolute, FREQS, values[:, index])
        for index in range(values.shape[1])
    ], axis=1)
    return aligned, relative


def subject_maps(epochs, good, peak, args):
    """Compute baselined sensor/ROI contrasts in both windows."""
    on, times = power(epochs["stim"], good, args.n_jobs)
    off, off_times = power(epochs["no-stim"], good, args.n_jobs)
    if not np.array_equal(times, off_times):
        raise ValueError("ON/OFF TFR time grids differ")
    on = baseline_percent(on, times)
    off = baseline_percent(off, times)
    available = {ch: index for index, ch in enumerate(good)}
    groups = {ch: [ch] for ch in good}
    for name, requested in (("ROI3", ROI3), ("ROI8", ROI8)):
        chosen = [ch for ch in requested if ch in available]
        if chosen:
            groups[name] = chosen

    output = {}
    for name, sensors in groups.items():
        indices = [available[ch] for ch in sensors]
        # Trial means are baseline corrected at each sensor/frequency.
        # ROI sensor means are then computed within each condition.
        full = contrast(
            on[indices].mean(axis=0),
            off[indices].mean(axis=0),
        )
        for band, limits in WINDOWS.items():
            time_mask = (
                (times >= limits["time"][0])
                & (times <= limits["time"][1])
            )
            if band == "alpha":
                selected, frequency = interpolate_alpha(full, peak)
            else:
                frequency_mask = (
                    (FREQS >= limits["freq"][0])
                    & (FREQS <= limits["freq"][1])
                )
                selected = full[frequency_mask]
                frequency = FREQS[frequency_mask]
            output[(name, band)] = (
                selected[:, time_mask], frequency, times[time_mask],
                sensors,
            )
    return output


def cluster_test(data, n_permutations, seed):
    """Two-sided frequency-time cluster test across participants."""
    n_subjects, n_freq, n_time = data.shape
    if n_subjects < 3:
        raise ValueError("At least three participants required")
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite participant contrast")
    statistic, clusters, pvals, _ = permutation_cluster_1samp_test(
        data, threshold=None, tail=0,
        n_permutations=n_permutations,
        adjacency=combine_adjacency(n_freq, n_time),
        out_type="mask", seed=seed, verbose=False,
    )
    records = []
    for index, (mask, p) in enumerate(zip(clusters, pvals)):
        if not np.any(mask):
            continue
        records.append({
            "index": index, "p": float(p),
            "mask": mask, "mass": float(np.abs(statistic[mask]).sum()),
            "n_bins": int(mask.sum()),
        })
    return records


def figure_for_test(data, frequency, times, records, title):
    """Plot mean baselined difference and cluster outlines."""
    mean = data.mean(axis=0)
    limit = max(float(np.quantile(np.abs(mean), 0.99)), 1e-8)
    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    image = ax.pcolormesh(
        times, frequency, mean, shading="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    mask = np.zeros_like(mean, dtype=bool)
    for row in records:
        if row["p"] <= 0.05:
            mask |= row["mask"]
    if mask.any():
        ax.contour(
            times, frequency, mask.astype(float),
            levels=[0.5], colors="black", linewidths=2,
        )
    ax.set(
        title=title, xlabel="Time from cue onset (s)",
        ylabel=(
            "Frequency relative to individual alpha peak (Hz)"
            if "alpha" in title.lower() else "Absolute frequency (Hz)"
        ),
    )
    fig.colorbar(image, ax=ax, label="Baselined ON - OFF (percentage points)")
    return fig


def text_page(title, paragraphs):
    """Make a readable PDF page of methods or numerical findings."""
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    ax.text(
        0.05, 0.95, title, transform=ax.transAxes,
        va="top", fontsize=16, weight="bold",
    )
    y = 0.87
    import textwrap
    for paragraph in paragraphs:
        for line in textwrap.wrap(str(paragraph), width=105) or [""]:
            ax.text(
                0.05, y, line, transform=ax.transAxes,
                va="top", fontsize=9.5,
            )
            y -= 0.025
        y -= 0.027
    return fig


def main():
    """Build an exploratory cluster report for all sensors and two ROIs."""
    args = arguments()
    root = resolve_project_root(args.platform, args.project_root)
    group = root / "derivatives" / "reports" / "group"
    peaks_path = args.peaks_csv or group / "individual_alpha_peaks.csv"
    peaks_frame = pd.read_csv(peaks_path, dtype={"subject": str})
    if not {"subject", "peak_hz"}.issubset(peaks_frame.columns):
        raise ValueError("Peak CSV needs subject and peak_hz")
    peaks_frame["subject"] = peaks_frame.subject.str.removeprefix("sub-")
    if peaks_frame.subject.duplicated().any():
        raise ValueError("Duplicate alpha peak subject")
    peaks = dict(zip(
        peaks_frame.subject,
        pd.to_numeric(peaks_frame.peak_hz, errors="coerce"),
    ))
    collected = {}
    for subject in args.subjects:
        if subject not in peaks or not np.isfinite(peaks[subject]):
            raise ValueError(f"Missing peak for sub-{subject}")
        epochs, good = read_subject(root, subject, args)
        if not good:
            raise ValueError(f"sub-{subject}: no shared good EEG sensors")
        collected[subject] = subject_maps(
            epochs, good, peaks[subject], args
        )
        print(f"sub-{subject}: computed baselined alpha/theta maps")

    keys = sorted(set().union(*(
        set(value) for value in collected.values()
    )))
    results = []
    for key in keys:
        subjects = [s for s in args.subjects if key in collected[s]]
        if len(subjects) < 3:
            continue
        data = np.stack([collected[s][key][0] for s in subjects])
        frequency, times, sensors = collected[subjects[0]][key][1:]
        for subject in subjects[1:]:
            other_freq, other_times, _ = collected[subject][key][1:]
            if not (
                np.array_equal(frequency, other_freq)
                and np.array_equal(times, other_times)
            ):
                raise ValueError(f"Inconsistent TFR grid for {key}")
        clusters = cluster_test(
            data, args.n_permutations, args.seed
        )
        minimum_p = min((row["p"] for row in clusters), default=1.0)
        results.append({
            "key": key, "subjects": subjects, "sensors": sensors,
            "data": data, "frequency": frequency, "times": times,
            "clusters": clusters, "minimum_p": minimum_p,
        })
    if not results:
        raise RuntimeError("No eligible sensor or ROI tests")
    reject, adjusted, _, _ = multipletests(
        [result["minimum_p"] for result in results],
        alpha=0.05, method="fdr_bh",
    )
    output = args.output_dir or (
        group / "EEG_posterior_alpha_theta_clusters"
    )
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / "S01_alpha_theta_targeted_clusters.pdf"
    with PdfPages(pdf_path) as pdf:
        page = text_page("Targeted alpha and theta cluster tests", [
            "EXPLORATORY POST-HOC WINDOWS: these frequency/time limits were "
            "selected after visually inspecting grand-average TFRs. "
            "Reported p-values cannot be interpreted as independent "
            "confirmation of the observed patterns.",
            "Alpha: participant OFF-derived peak -2.5 to +1 Hz, "
            "cue-relative time 0.1-0.7 s. Frequency is aligned to "
            "each participant's peak before group testing.",
            "Theta: absolute 4-5.5 Hz, cue-relative time 0.1-0.4 s.",
            "For each participant and condition: trial-average linear "
            "multitaper power (2-31.5 Hz, 0.5-Hz steps, cycles=f/2, "
            "bandwidth=2, decimation=2). "
            "Average sensors within each ROI and condition after "
            "baseline-correct ON and OFF independently at each "
            "sensor/frequency over -0.3 to -0.1 s, using "
            "100*(power/baseline - 1). Then subtract ON minus OFF. "
            "The resulting difference is in percentage points.",
            "Two-sided one-sample frequency x time cluster-mass "
            "sign-flip permutation tests. Cluster-level p values "
            "control the frequency/time search within each window. "
            "The smallest cluster p per test (or 1 if none) is "
            "additionally BH-FDR corrected across all eligible "
            "sensor/ROI x band tests.",
            "Cluster significance does not precisely localise onset "
            "or offset, and single-bin clusters are not evidence "
            "of a sustained effect. These tests assume participant "
            "differences are independent and sign-exchangeable.",
            f"Participants requested: {', '.join(args.subjects)}",
            f"Permutation count: {args.n_permutations}; seed: {args.seed}",
        ])
        pdf.savefig(page)
        plt.close(page)
        for result, is_rejected, q in zip(results, reject, adjusted):
            name, band = result["key"]
            title = f"{name}: {band} | N={len(result['subjects'])}"
            cluster_details = [
                (
                    f"Cluster {row['index']}: p={row['p']:.5g}; "
                    f"bins={row['n_bins']}; "
                    f"mass={row['mass']:.5g}"
                )
                for row in result["clusters"]
            ] or ["No supra-threshold clusters."]
            page = text_page(title + " | cluster details", [
                f"Sensors: {', '.join(result['sensors'])}",
                f"Participants: {', '.join(result['subjects'])}",
                f"Minimum cluster p={result['minimum_p']:.5g}; "
                f"across-test BH-FDR q={q:.5g}; "
                f"FDR significant={bool(is_rejected)}",
                *cluster_details,
            ])
            pdf.savefig(page)
            plt.close(page)
            figure = figure_for_test(
                result["data"], result["frequency"],
                result["times"], result["clusters"], title,
            )
            pdf.savefig(figure)
            plt.close(figure)
    print(f"Report: {pdf_path}")


if __name__ == "__main__":
    main()
