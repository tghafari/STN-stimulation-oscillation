"""Paired right-minus-left sensor DBS effect, independent PDF-only mode.

All subject comparisons use good, homologous channels from the same
participant and the same ON/OFF epochs. Baseline-correct each condition
before subtraction. Report two-sided one-sample spatiotemporal-free
(time-frequency) cluster permutations and three paired alpha tests.
"""
from __future__ import annotations

import json
import textwrap

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_1samp
from mne.stats import permutation_cluster_1samp_test

PAIRS = (("PO7", "PO8"), ("PO3", "PO4"), ("O1", "O2"))
BASELINE = (-0.3, -0.1)
ALPHA_WINDOW = (0.1, 0.7)
ALPHA_BAND = (8.0, 12.0)


def baseline_percent(power, times):
    """Return percent baseline change for frequency-by-time power."""
    baseline = (times >= BASELINE[0]) & (times <= BASELINE[1])
    if not baseline.any():
        raise ValueError("TFR does not contain the baseline window.")
    reference = power[:, baseline].mean(axis=1, keepdims=True)
    if not np.all(np.isfinite(reference)) or np.any(reference <= 0):
        raise ValueError("Invalid baseline power.")
    return 100.0 * (power / reference - 1.0)


def load_paired_subject(subject, args, root, trials, parent):
    """Compute ON-OFF for each good channel using validated B03 labels."""
    epochs = {
        condition: parent.load_cue_epochs(root, subject, condition, args)
        for condition in ("stim", "no-stim")
    }
    rows = parent.verified_rows(trials, subject, epochs)
    conditions = parent.extract_conditions(epochs, rows)
    # Both unilateral ON blocks contribute to the pooled ON condition.
    # Keep their original trials and do not relabel laterality.
    on = mne.concatenate_epochs(
        [conditions["stim_left"], conditions["stim_right"]],
        add_offset=True,
    )
    off = conditions["off"]
    valid = {}
    for left, right in PAIRS:
        if all(
            ch in on.ch_names and ch in off.ch_names
            and ch not in on.info["bads"] and ch not in off.info["bads"]
            for ch in (left, right)
        ):
            valid[(left, right)] = (left, right)
    if not valid:
        raise ValueError("No homologous good channel pairs.")
    picks = sorted({ch for pair in valid for ch in pair})
    times = None
    condition_maps = {}
    for name, ep in (("on", on), ("off", off)):
        current_times, power = parent.multitaper_power(
            ep, picks, args.n_jobs
        )
        if times is None:
            times = current_times
        elif not np.allclose(times, current_times):
            raise ValueError("ON/OFF time axes differ.")
        condition_maps[name] = {
            ch: baseline_percent(power[picks.index(ch)], times)
            for ch in picks
        }
    contrasts = {}
    for pair in valid:
        left, right = pair
        contrasts[pair] = (
            condition_maps["on"][right] - condition_maps["off"][right]
            - condition_maps["on"][left] + condition_maps["off"][left]
        )
    return times, contrasts, {"on": len(on), "off": len(off)}


def summary_page(pdf, counts, skipped, n_permutations):
    """Write methodological details and inclusion counts."""
    lines = [
        "PAIRED HEMISPHERIC STIMULATION-EFFECT ASYMMETRY",
        "",
        "Pair convention: RIGHT sensor minus LEFT sensor.",
        "Pairs: PO8-PO7, PO4-PO3, O2-O1.",
        "For each participant, use only good channels available in both",
        "ON and OFF epochs. Each pair has its own participant sample.",
        "ON pools verified left-STN and right-STN ON trials; OFF pools",
        "both verified OFF periods, using B03 trial_alpha.csv labels.",
        "Attention-left and attention-right cues are combined.",
        "Compute multitaper power per condition (2-31.5 Hz, 0.5-Hz",
        "steps, cycles=f/2, bandwidth=2, decim=2), then percent baseline",
        "correct each sensor and condition to -0.3 to -0.1 s.",
        "Within participant: (ON-OFF)_right - (ON-OFF)_left.",
        "Thus negative means stronger ON-related suppression on RIGHT.",
        "Positive means stronger ON-related suppression on LEFT.",
        "Group TFR is the equal-weight mean of within-subject contrasts.",
        "Two-sided cluster permutation is performed separately for each",
        f"pair ({n_permutations} permutations, cluster-forming p<0.05).",
        "Clusters are connected in frequency/time; p-values are",
        "uncorrected across the three pairs (exploratory).",
        "A prespecified mean 8-12 Hz, 0.1-0.7 s paired contrast is",
        "also tested against zero using a two-sided one-sample t-test.",
        "Holm correction is applied across the three scalar tests.",
        "",
    ]
    for pair, subjects in counts.items():
        lines.append(
            f"{pair[1]}-{pair[0]}: n={len(subjects)}; "
            f"sub-{', sub-'.join(subjects) if subjects else 'none'}"
        )
    if skipped:
        lines.extend(["", "Skipped subjects:"])
        lines.extend(f"sub-{k}: {v}" for k, v in skipped.items())
    fig, ax = plt.subplots(figsize=(11.7, 10))
    ax.axis("off")
    ax.text(
        0.06, 0.96, "\n".join(lines),
        va="top", family="monospace", fontsize=9.2,
        transform=ax.transAxes,
    )
    pdf.savefig(fig)
    plt.close(fig)


def plot_pair(pdf, pair, subjects, maps, times, freqs, stats):
    """Add a grand-average TFR and time course for a sensor pair."""
    mean_map = maps.mean(axis=0)
    limit = max(np.nanpercentile(np.abs(mean_map), 99), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), layout="constrained")
    image = axes[0].pcolormesh(
        times, freqs, mean_map, shading="auto",
        cmap="RdBu_r", vmin=-limit, vmax=limit,
    )
    axes[0].axvline(0, color="black", ls="--", lw=0.8)
    axes[0].set(
        title=f"Grand mean (ON-OFF)_{pair[1]} - (ON-OFF)_{pair[0]}",
        xlabel="Time after cue (s)", ylabel="Frequency (Hz)",
    )
    fig.colorbar(image, ax=axes[0], label="Percentage-point difference")
    fm = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    alpha = maps[:, fm, :].mean(axis=1)
    mean = alpha.mean(axis=0)
    sem = alpha.std(axis=0, ddof=1) / np.sqrt(len(subjects))
    axes[1].plot(times, mean, color="black")
    axes[1].fill_between(times, mean-sem, mean+sem, alpha=0.2)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].axvspan(*ALPHA_WINDOW, color="gray", alpha=0.1)
    axes[1].set(
        title="8-12 Hz mean +/- SEM",
        xlabel="Time after cue (s)", ylabel="Right minus left (pp)",
    )
    fig.suptitle(
        f"{pair[1]} minus {pair[0]} | n={len(subjects)} | "
        f"scalar p={stats['p_scalar']:.4g} | "
        f"cluster min p={stats['min_cluster_p']:.4g}",
    )
    pdf.savefig(fig)
    plt.close(fig)


def run_paired_asymmetry(args):
    """Run only the paired-sensor PDF; do not generate existing reports."""
    import run_stim_laterality_tfr as parent

    root = parent.resolve_project_root(args.platform, args.project_root)
    group = root / "derivatives" / "reports" / "group"
    trial_path = args.trial_csv or (
        group / "trials_assigning_to_blocks_report" / "trial_alpha.csv"
    )
    trials = pd.read_csv(trial_path, dtype={"subject": str})
    required = {
        "subject", "source_condition", "trial_index_in_condition",
        "label_verified", "dbs_on", "stimulation_period",
        "stimulation_side",
    }
    if not required.issubset(trials.columns):
        raise ValueError(f"Missing trial CSV fields: {required - set(trials.columns)}")
    trials["subject"] = trials["subject"].str.removeprefix("sub-")
    out = args.output_dir or (
        group / "EEG_posterior_channels_stim_laterality"
    )
    out.mkdir(parents=True, exist_ok=True)

    samples = {pair: [] for pair in PAIRS}
    membership = {pair: [] for pair in PAIRS}
    skipped = {}
    times = None
    for subject in args.subjects:
        try:
            current_times, contrasts, counts = load_paired_subject(
                subject, args, root, trials, parent
            )
            if times is None:
                times = current_times
            elif not np.allclose(times, current_times):
                raise ValueError("TFR time axis differs across participants.")
            for pair, values in contrasts.items():
                samples[pair].append(values)
                membership[pair].append(subject)
            print(f"sub-{subject}: ON={counts['on']}, OFF={counts['off']}; "
                  f"eligible pairs={len(contrasts)}")
        except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
            skipped[subject] = str(exc)
            print(f"sub-{subject}: skipped - {exc}")

    path = out / "paired_right_minus_left_sensor_stimulation_effect.pdf"
    results = {}
    pvals = []
    tested = []
    freqs = parent.FREQUENCIES
    for pair in PAIRS:
        if len(samples[pair]) < 3:
            results[str(pair)] = {"n": len(samples[pair]), "status": "n<3"}
            continue
        data = np.stack(samples[pair])
        fm = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
        tm = (times >= ALPHA_WINDOW[0]) & (times <= ALPHA_WINDOW[1])
        scalars = data[:, fm, :][:, :, tm].mean(axis=(1, 2))
        test = ttest_1samp(scalars, 0, alternative="two-sided")
        from scipy.stats import t as student_t
        threshold = float(student_t.ppf(0.975, len(data)-1))
        # Restrict cluster testing to the attention window and alpha band.
        cluster_data = data[:, fm, :][:, :, tm]
        t_obs, clusters, cluster_p, _ = permutation_cluster_1samp_test(
            cluster_data, threshold=threshold, tail=0,
            n_permutations=args.n_permutations, seed=42,
            out_type="mask", verbose=False,
        )
        pvals.append(float(test.pvalue))
        tested.append(pair)
        results[str(pair)] = {
            "n": len(data), "subjects": membership[pair],
            "mean_alpha_pp": float(scalars.mean()),
            "t_scalar": float(test.statistic),
            "p_scalar": float(test.pvalue),
            "min_cluster_p": float(np.min(cluster_p)) if len(cluster_p) else 1.0,
            "cluster_p_values": [float(p) for p in cluster_p],
            "cluster_window_hz": list(ALPHA_BAND),
            "cluster_window_s": list(ALPHA_WINDOW),
        }

    # Holm familywise adjustment across the three prespecified scalar tests.
    order = np.argsort(pvals)
    corrected = np.zeros(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvals)-rank)*pvals[idx]))
        corrected[idx] = running
    for pair, adjusted in zip(tested, corrected):
        results[str(pair)]["p_holm"] = float(adjusted)

    with PdfPages(path) as pdf:
        summary_page(pdf, membership, skipped, args.n_permutations)
        for pair in tested:
            data = np.stack(samples[pair])
            plot_pair(pdf, pair, membership[pair], data, times, freqs,
                      results[str(pair)])
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        lines = ["PAIRED SENSOR ASYMMETRY - STATISTICAL RESULTS", ""]
        for pair in PAIRS:
            entry = results[str(pair)]
            lines.append(
                f"{pair[1]} - {pair[0]}: n={entry['n']}, "
                + (
                    f"mean={entry['mean_alpha_pp']:.3f} pp, "
                    f"t={entry['t_scalar']:.3f}, "
                    f"p(two-sided)={entry['p_scalar']:.5f}, "
                    f"Holm p={entry['p_holm']:.5f}, "
                    f"min cluster p={entry['min_cluster_p']:.5f}"
                    if pair in tested else "not tested (n<3)"
                )
            )
        lines += [
            "",
            "Scalar tests: mean 8-12 Hz, 0.1-0.7 s per participant.",
            "Holm correction across three pair tests.",
            "Cluster p-values are exploratory and uncorrected across pairs.",
            "Negative: stronger stimulation-related suppression on RIGHT.",
            "Positive: stronger stimulation-related suppression on LEFT.",
        ]
        ax.text(0.05, 0.95, "\n".join(lines), va="top",
                family="monospace", fontsize=10, transform=ax.transAxes)
        pdf.savefig(fig)
        plt.close(fig)
    (out / "paired_sensor_asymmetry_stats.json").write_text(
        json.dumps({"results": results, "skipped": skipped}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved paired asymmetry PDF: {path}")
