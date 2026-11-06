#!/usr/bin/env python
"""Individual-alpha-peak-based ON-minus-OFF cluster statistics.

Recompute G02-style trial-averaged TFRs from cleaned cue epochs, selecting
each participant's OFF-derived alpha peak +/- 1 Hz from the individual peak
CSV. Percent-baseline correct ON and OFF separately (-0.3 to -0.1 s).
Test their paired difference against ZERO over 0.3-0.8 s with two-sided
temporal cluster permutation tests for sensors, ROI3 and ROI8.

Also perform a whole 0.3-0.8 s mean one-sample t-test against zero for
each ROI. Results are descriptive/inferential at the participant level;
ROI channel averages precede subject averaging. G02 writes peak estimates,
not TFR objects, so the TFR is computed again from the same epochs.
"""
from __future__ import annotations
import argparse, csv, json, sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats
from mne.stats import permutation_cluster_1samp_test, combine_adjacency, fdr_correction

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parents[1]
for p in (ANALYSIS_DIR / "subject" / "EEG_all_channels", ANALYSIS_DIR / "utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from pipeline_config import CONDITIONS, resolve_project_root, stage_path
from pdf_report import ParticipantPDF

DEFAULT_ALPHA = (8.0, 12.0)  # legacy --alpha option; peak range is primary
WINDOW = (0.3, 0.8)
BASELINE = (-0.3, -0.1)
FREQS = np.arange(2.0, 32.0, 0.5)
N_CYCLES = FREQS / 2.0
TIME_BANDWIDTH = 2.0
DECIM = 2
ROI8 = ("PO3", "POz", "PO4", "O1", "Oz", "O2", "PO7", "PO8")
ROI3 = ("PO3", "POz", "PO4")
PTHRESH = 0.05
CLUSTER_FORMING_ALPHA = 0.10


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="SpAtt")
    p.add_argument("--run", default="01")
    p.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    p.add_argument("--project-root", default=None)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument(
        "--alpha",
        nargs=2,
        type=float,
        metavar=("FMIN", "FMAX"),
        default=DEFAULT_ALPHA,
        help="Frequency range in Hz to average before cluster testing, e.g. --alpha 8 12",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--peaks-csv", type=Path, default=None)
    p.add_argument("--peak-half-width", type=float, default=1.0)
    p.add_argument(
        "--sensor-fdr",
        action="store_true",
        help="Optionally apply BH-FDR across single-sensor cluster p-values; default is no FDR.",
    )
    a = p.parse_args()
    if a.alpha[0] >= a.alpha[1]:
        p.error("--alpha requires FMIN < FMAX")
    if a.alpha[0] < FREQS.min() or a.alpha[1] > FREQS.max():
        p.error(f"--alpha must lie within {FREQS.min():g}-{FREQS.max():g} Hz")
    return a


def load(root, s, a):
    d = {}
    for c in CONDITIONS:
        f = stage_path(root, s, a.session, a.task, a.run, c, "clean", "epo")
        if not f.exists():
            raise FileNotFoundError(f)
        e = mne.read_epochs(f, preload=True, verbose=False)
        k = [x for x in ("cue_onset_right", "cue_onset_left") if x in e.event_id]
        d[c] = e[k] if k else e
    return d


def good(x):
    st, no = x["stim"], x["no-stim"]
    return [
        ch
        for ch in st.copy().pick("eeg").ch_names
        if ch in no.ch_names and ch not in st.info["bads"] and ch not in no.info["bads"]
    ]


def alpha(ep, chs, a, peak):
    t = (
        ep.copy()
        .pick(chs)
        .compute_tfr(
            method="multitaper",
            freqs=FREQS,
            n_cycles=N_CYCLES,
            time_bandwidth=TIME_BANDWIDTH,
            use_fft=True,
            zero_mean=True,
            return_itc=False,
            average=True,
            decim=DECIM,
            n_jobs=a.n_jobs,
            verbose=False,
        )
    )
    lo, hi = (peak - a.peak_half_width, peak + a.peak_half_width)
    fi = (t.freqs >= lo - 1e-9) & (t.freqs <= hi + 1e-9)
    ti = (t.times >= WINDOW[0]) & (t.times <= WINDOW[1])
    t.apply_baseline(BASELINE, mode='percent')
    return 100.0 * t.data[:, fi][:, :, ti].mean(1), t.times[ti], t.info


def one_d(X, seed):
    return permutation_cluster_1samp_test(
        X,
        n_permutations="all",
        threshold=None,
        tail=0,
        adjacency=None,
        out_type="mask",
        seed=seed,
        verbose=False,
    )


def sig_1d(T, cl, p, t):
    out = []
    for i, (m, pv) in enumerate(zip(cl, p)):
        if pv <= PTHRESH:
            ix = np.where(np.asarray(m, bool))[0]
            out.append(
                {
                    "cluster": i,
                    "p": float(pv),
                    "t_start": float(t[ix].min()),
                    "t_end": float(t[ix].max()),
                    "peak_t": float(t[ix[np.argmax(np.abs(T[ix]))]]),
                    "peak_stat": float(T[ix][np.argmax(np.abs(T[ix]))]),
                }
            )
    return out


def shade(ax, rows):
    ymin, ymax = ax.get_ylim()
    for r in rows:
        ax.axvspan(
            r["t_start"],
            r["t_end"],
            facecolor="0.72",
            edgecolor="0.25",
            alpha=0.5,
            zorder=0,
        )
        mid = (r["t_start"] + r["t_end"]) / 2
        p = r.get("p_cluster", r.get("p", np.nan))
        ax.text(
            mid,
            ymax - (ymax - ymin) * 0.04,
            f"cluster p={p:.4f}\n{r['t_start']:.3f}-{r['t_end']:.3f} s",
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.4"),
        )


def roi_plot(st, no, t, rows, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ms = st.mean(0)
    mn = no.mean(0)
    sem_s = st.std(0, ddof=1) / np.sqrt(len(st))
    sem_n = no.std(0, ddof=1) / np.sqrt(len(no))
    ax.plot(t, mn, label="No stimulation")
    ax.fill_between(t, mn - sem_n, mn + sem_n, alpha=0.15)
    ax.plot(t, ms, label="Stimulation")
    ax.fill_between(t, ms - sem_s, ms + sem_s, alpha=0.15)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.relim()
    ax.autoscale_view()
    shade(ax, rows)
    return fig


def sensor_plot(st, no, t, ch, rows):
    return roi_plot(
        st,
        no,
        t,
        rows,
        f"{ch}: alpha power, stimulation vs no stimulation",
        "Alpha power",
    )


def participant_difference_traces(diff, t, subjects, title):
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for sub, y in zip(subjects, diff):
        ax.plot(t, y, linewidth=0.9, alpha=0.55, label=f"sub-{sub}")
    mean = diff.mean(0)
    sem = diff.std(0, ddof=1) / np.sqrt(len(diff))
    ax.plot(t, mean, color="k", linewidth=3, label="Group mean")
    ax.fill_between(t, mean - sem, mean + sem, color="k", alpha=0.12)
    ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stim - no-stim alpha power")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=7)
    return fig


def participant_window_effects(diff, subjects, title):
    values = diff.mean(1)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(len(subjects))
    ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
    ax.scatter(x, values, s=45, zorder=3)
    ax.plot(x, values, linewidth=0.7, alpha=0.45)
    ax.axhline(
        values.mean(), color="k", linewidth=2, label=f"Group mean = {values.mean():.4g}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"sub-{z}" for z in subjects], rotation=60, ha="right")
    ax.set_ylabel(f"Mean stim - no-stim alpha, {WINDOW[0]:g}-{WINDOW[1]:g} s")
    ax.set_title(title)
    ax.legend()
    return fig, values


def cluster_threshold(n):
    return float(stats.t.ppf(1 - CLUSTER_FORMING_ALPHA / 2, n - 1))


def summarize_clusters(T, clusters, pvals, t, chs=None):
    out = []
    for i, (m, pv) in enumerate(zip(clusters, pvals)):
        mask = np.asarray(m, bool)
        if chs is None:
            ix = np.where(mask)[0]
            if not ix.size:
                continue
            vals = T[ix]
            out.append(
                dict(
                    cluster=i,
                    p=float(pv),
                    cluster_stat=float(vals.sum()),
                    n_samples=int(ix.size),
                    n_sensors=1,
                    t_start=float(t[ix].min()),
                    t_end=float(t[ix].max()),
                    sensors="ROI",
                )
            )
        else:
            ti, ci = np.where(mask)
            if not ti.size:
                continue
            vals = T[mask]
            used = [chs[j] for j in sorted(set(ci))]
            out.append(
                dict(
                    cluster=i,
                    p=float(pv),
                    cluster_stat=float(vals.sum()),
                    n_samples=int(mask.sum()),
                    n_sensors=len(used),
                    t_start=float(t[ti].min()),
                    t_end=float(t[ti].max()),
                    sensors=", ".join(used),
                )
            )
    return out


def spatial_rows(T, clusters, pvals, t, chs):
    out = []
    for i, (m, p) in enumerate(zip(clusters, pvals)):
        if p > PTHRESH:
            continue
        ti, ci = np.where(np.asarray(m, bool))
        used = [chs[j] for j in sorted(set(ci))]
        out.append(
            {
                "cluster": i,
                "p": float(p),
                "t_start": float(t[ti].min()),
                "t_end": float(t[ti].max()),
                "sensors": used,
                "mask": np.asarray(m, bool),
            }
        )
    return out


def spatial_topomap(row, T, info, chs):
    mask = row["mask"]
    vals = np.zeros(len(chs))
    active = np.zeros(len(chs), bool)
    for ci in range(len(chs)):
        x = T[:, ci][mask[:, ci]]
        if x.size:
            vals[ci] = x.mean()
            active[ci] = True
    fig, ax = plt.subplots(figsize=(7, 6))
    im, _ = mne.viz.plot_topomap(
        vals,
        info,
        axes=ax,
        show=False,
        contours=0,
        mask=active,
        mask_params=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="k",
            linewidth=0,
            markersize=8,
        ),
    )
    ax.set_title(
        f"Spatial cluster p={row['p']:.4f}, {row['t_start']:.3f}-{row['t_end']:.3f} s"
    )
    fig.colorbar(im, ax=ax, label="Mean cluster t statistic")
    return fig


def main():
    a = args()
    subs = [s.removeprefix("sub-") for s in a.subjects]
    root = resolve_project_root(a.platform, a.project_root)
    peaks_path = a.peaks_csv or (
        root / "derivatives" / "reports" / "group" / "individual_alpha_peaks.csv"
    )
    peaks_frame = pd.read_csv(peaks_path, dtype={"subject": str})
    if not {"subject", "peak_hz"}.issubset(peaks_frame.columns):
        raise ValueError("Peak CSV needs subject and peak_hz")
    peaks_frame["subject"] = peaks_frame.subject.str.removeprefix("sub-")
    if peaks_frame.subject.duplicated().any():
        raise ValueError("Duplicate subject in peak CSV")
    peaks = dict(zip(peaks_frame.subject,
                     pd.to_numeric(peaks_frame.peak_hz, errors="coerce")))
    for sub in subs:
        if sub not in peaks or not np.isfinite(peaks[sub]) or not 8 <= peaks[sub] <= 12:
            raise ValueError(f"Missing or invalid G02 alpha peak for sub-{sub}")
    ep = {s: load(root, s, a) for s in subs}
    g = {s: good(ep[s]) for s in subs}
    out = root / "derivatives" / "reports" / "group" / "EEG_all_channels_alpha_cluster"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    report = ParticipantPDF(str(out), "alpha_cluster_" + "_".join(subs))
    section = "Alpha cluster permutation"
    report.add_text(
        "Analysis structure",
        "This report intentionally contains NO whole-scalp sensor x time cluster test. Single sensors are tested separately using every participant retaining that sensor in both stimulation conditions. ROI3 uses participants retaining PO3, POz and PO4 in both conditions. ROI8 uses participants retaining PO3, POz, PO4, O1, Oz, O2, PO7 and PO8 in both conditions. Thus a bad channel excludes a participant only from analyses that require that channel.",
        section,
    )
    details = (
        f"Participants included: {', '.join('sub-' + x for x in subs)}. "
        f"Peak file: {peaks_path}. Peak +/-{a.peak_half_width:g} Hz per subject. "
        "G02-compatible multitaper TFR: 2-31.5 Hz in 0.5-Hz steps, "
        "cycles=f/2, bandwidth=2, decimation=2, trial-averaged power. "
        "For EACH participant/sensor/frequency, Stim ON and OFF power are "
        "separately percent-baseline corrected (-0.3 to -0.1 s). "
        "ON-percent minus OFF-percent (percentage points) is then tested "
        "against zero during 0.3-0.8 s. Two-sided temporal cluster "
        "permutation alpha=0.05 with forming alpha=0.10. "
        "ROI means average sensors within participants first."
    )
    report.add_text("Analysis details", details, section)
    report.add_text("Individual peaks", "\n".join(
        f"sub-{sub}: {peaks[sub]:.2f} Hz" for sub in subs), section)
    methods = (
        "Alpha power was measured using participant-specific OFF-derived "
        "peak frequencies +/-1 Hz. Each cleaned participant-condition TFR "
        "was separately baseline-corrected in percent mode over "
        "-0.3 to -0.1 seconds. Differences in percentage points "
        "(ON minus OFF) were tested against zero during 0.3-0.8 seconds "
        "with two-sided one-sample temporal cluster permutations. "
        "Posterior ROI sensors were averaged within participants before "
        "testing; a whole-window one-sample t-test additionally tested "
        "the mean participant difference against zero."
    )
    report.add_text("Manuscript-style statistical analysis", methods, section)
    # Whole-scalp sensor x time testing is intentionally NOT run here.
    # Reason: it requires a common sensor set across all participants, which in this
    # dataset discards many posterior/occipital sensors because one bad channel in
    # one participant removes that sensor from the entire whole-scalp test.
    # The inferential analyses below instead use channel/ROI-specific complete cases.
    # Single-sensor tests: sensor-specific complete-case participants.
    sensor_results = {}
    sensor_cluster_records = []
    sensor_pvals = []
    all_sensor_names = []
    for sub in subs:
        for ch in g[sub]:
            if ch not in all_sensor_names:
                all_sensor_names.append(ch)
    for ch in all_sensor_names:
        ch_subs = [sub for sub in subs if ch in g[sub]]
        excluded = [sub for sub in subs if sub not in ch_subs]
        if len(ch_subs) < 2:
            continue
        ch_st = []
        ch_no = []
        for sub in ch_subs:
            xs, ct, _ = alpha(ep[sub]["stim"], [ch], a, peaks[sub])
            xn, ct2, _ = alpha(ep[sub]["no-stim"], [ch], a, peaks[sub])
            if not np.allclose(ct, ct2):
                raise RuntimeError(f"Sensor time mismatch sub-{sub} {ch}")
            ch_st.append(xs[0])
            ch_no.append(xn[0])
        ch_st = np.asarray(ch_st)
        ch_no = np.asarray(ch_no)
        ch_thresh = cluster_threshold(len(ch_subs))
        Ts, cs, ps, _ = permutation_cluster_1samp_test(
            ch_st - ch_no,
            n_permutations="all",
            threshold=ch_thresh,
            tail=0,
            adjacency=None,
            out_type="mask",
            seed=a.seed,
            verbose=False,
        )
        all_ch = summarize_clusters(Ts, cs, ps, ct)
        sig_ch = [r for r in all_ch if r["p"] <= PTHRESH]
        sensor_results[ch] = {
            "n_subjects": len(ch_subs),
            "included_subjects": ch_subs,
            "excluded_subjects": excluded,
            "cluster_forming_threshold_t": ch_thresh,
            "significant_clusters": sig_ch,
            "all_clusters": all_ch,
        }
        for r in all_ch:
            rec = {
                "sensor": ch,
                "n_subjects": len(ch_subs),
                "cluster": r["cluster"],
                "p_cluster": float(r["p"]),
                "cluster_stat": r["cluster_stat"],
                "n_samples": r["n_samples"],
                "t_start": r["t_start"],
                "t_end": r["t_end"],
            }
            sensor_cluster_records.append(rec)
            sensor_pvals.append(rec["p_cluster"])
    if a.sensor_fdr and sensor_pvals:
        reject, qvals = fdr_correction(
            np.asarray(sensor_pvals), alpha=PTHRESH, method="indep"
        )
        for rec, rej, q in zip(sensor_cluster_records, reject, qvals):
            rec["p_fdr"] = float(q)
            rec["significant_fdr"] = bool(rej)
    sig_uncorrected = sorted(
        {r["sensor"] for r in sensor_cluster_records if r["p_cluster"] <= PTHRESH}
    )
    sig_fdr = sorted(
        {r["sensor"] for r in sensor_cluster_records if r.get("significant_fdr")}
    )
    sensor_text = (
        "No single-sensor clusters had cluster p<=0.05."
        if not sig_uncorrected
        else "Sensors with at least one cluster p<=0.05 (not corrected across sensors): "
        + ", ".join(sig_uncorrected)
    )
    sensor_text += "\nEach sensor uses all participants for whom that sensor is good in BOTH stim and no-stim; missing another channel does not exclude that participant."
    sensor_text += (
        (
            "\nOptional BH-FDR was applied. FDR-significant sensors: "
            + (", ".join(sig_fdr) if sig_fdr else "None")
        )
        if a.sensor_fdr
        else "\nNo across-sensor FDR was applied. Use --sensor-fdr to request it."
    )
    report.add_text("Single-sensor cluster-permutation results", sensor_text, section)
    for ch in sig_uncorrected:
        dat = sensor_results[ch]
        sig = dat["significant_clusters"]
        detail = (
            f"{ch}: N={dat['n_subjects']}; cluster-forming threshold=+/-{dat['cluster_forming_threshold_t']:.4f}.\n"
            + (
                "\n".join(
                    f"Significant cluster {r['cluster']}: {r['t_start']:.3f}-{r['t_end']:.3f} s; cluster statistic={r['cluster_stat']:.4f}; cluster p={r['p']:.4f}; n_samples={r['n_samples']}."
                    for r in sig
                )
                if sig
                else "No significant clusters."
            )
        )
        report.add_text(
            f"{ch}: significant single-sensor cluster details", detail, section
        )
    for ch in sig_uncorrected:
        dat = sensor_results[ch]
        ch_subs = dat["included_subjects"]
        st = []
        no = []
        for sub in ch_subs:
            xs, ct, _ = alpha(ep[sub]["stim"], [ch], a, peaks[sub])
            xn, _, _ = alpha(ep[sub]["no-stim"], [ch], a, peaks[sub])
            st.append(xs[0])
            no.append(xn[0])
        rows = [
            r
            for r in sensor_cluster_records
            if r["sensor"] == ch and r["p_cluster"] <= PTHRESH
        ]
        report.add_figure(
            sensor_plot(np.asarray(st), np.asarray(no), ct, ch, rows),
            str(figs / f"sensor_{ch}_significant_clusters.png"),
            f"{ch}: single-sensor alpha cluster(s)",
            f"Sensor-specific N={len(ch_subs)}. Gray vertical bands mark significant within-sensor cluster-permutation intervals and are labelled with cluster p-value and time range. Across-sensor FDR is not implied.",
            section,
        )
    # ROI analyses use ROI-specific complete-case subject samples.
    # A participant contributes only when EVERY channel in that ROI is good in BOTH conditions.
    roi_results = {}
    for label, requested in (
        ("8-channel posterior/occipital ROI", ROI8),
        ("3-channel posterior ROI", ROI3),
    ):
        roi_subs = [sub for sub in subs if any(ch in g[sub] for ch in requested)]
        excluded_subs = [sub for sub in subs if sub not in roi_subs]
        if len(roi_subs) < 2:
            report.add_text(
                label + " results",
                f"Insufficient complete-case participants. Required channels: {', '.join(requested)}. Included: {', '.join('sub-'+x for x in roi_subs) if roi_subs else 'None'}. Excluded because at least one required ROI channel was bad/missing: {', '.join('sub-'+x for x in excluded_subs) if excluded_subs else 'None'}.",
                section,
            )
            roi_results[label] = {
                "required_channels": list(requested),
                "included_subjects": roi_subs,
                "excluded_subjects": excluded_subs,
                "rows": [],
            }
            continue
        roi_st = []
        roi_no = []
        for sub in roi_subs:
            available = [ch for ch in requested if ch in g[sub]]
            st_sub, rt, rinfo = alpha(ep[sub]["stim"], available, a, peaks[sub])
            no_sub, rt2, _ = alpha(ep[sub]["no-stim"], available, a, peaks[sub])
            if not np.allclose(rt, rt2):
                raise RuntimeError("ROI time mismatch " + sub)
            roi_st.append(st_sub.mean(0))
            roi_no.append(no_sub.mean(0))
        st = np.asarray(roi_st)
        no = np.asarray(roi_no)
        roi_diff = st - no
        window_means = roi_diff.mean(axis=1)
        test = stats.ttest_1samp(window_means, popmean=0.0)
        window_t, window_p = float(test.statistic), float(test.pvalue)
        roi_thresh = cluster_threshold(len(roi_subs))
        Tr, cr, pr, _ = permutation_cluster_1samp_test(
            roi_diff,
            n_permutations="all",
            threshold=roi_thresh,
            tail=0,
            adjacency=None,
            out_type="mask",
            seed=a.seed,
            verbose=False,
        )
        rows = sig_1d(Tr, cr, pr, rt)
        all_roi = summarize_clusters(Tr, cr, pr, rt)
        roi_results[label] = {
            "required_channels": list(requested),
            "included_subjects": roi_subs,
            "excluded_subjects": excluded_subs,
            "n_subjects": len(roi_subs),
            "rows": rows,
            "all_clusters": all_roi,
            "whole_window_mean_t": window_t,
            "whole_window_mean_p": window_p,
        }
        top_roi = sorted(all_roi, key=lambda r: r["p"])[:10]
        text = (
            f"Required channels: {', '.join(requested)}. N={len(roi_subs)} complete-case participants. Included: {', '.join('sub-'+x for x in roi_subs)}. Excluded because at least one required ROI channel was bad/missing: {', '.join('sub-'+x for x in excluded_subs) if excluded_subs else 'None'}. "
            + (
                "No significant temporal clusters."
                if not rows
                else " ".join(
                    f"Cluster {r['cluster']}: p={r['p']:.4f}, {r['t_start']:.3f}-{r['t_end']:.3f} s."
                    for r in rows
                )
            )
        )
        text += f"\n0.3-0.8 s participant mean vs zero: t={window_t:.5g}; two-sided p={window_p:.5g}."
        text += (
            f"\nActual ROI cluster-forming threshold: +/-{roi_thresh:.4f} (df={len(roi_subs)-1}; two-sided alpha={CLUSTER_FORMING_ALPHA:.2f}, 0.05 each tail).\nStrongest observed clusters, including non-significant:\n"
            + (
                "\n".join(
                    f"Cluster {r['cluster']}: p={r['p']:.4f}; stat={r['cluster_stat']:.3f}; n_samples={r['n_samples']}; {r['t_start']:.3f}-{r['t_end']:.3f} s"
                    for r in top_roi
                )
                if top_roi
                else "None"
            )
        )
        report.add_text(label + " results", text, section)
        tag = "ROI8" if requested == ROI8 else "ROI3"
        report.add_figure(
            roi_plot(st, no, rt, rows, label, "Alpha power"),
            str(figs / f"{tag}_alpha_clusters.png"),
            label + ": stimulation vs no stimulation",
            f"Complete-case ROI sample N={len(roi_subs)}. All required ROI channels are retained in every included participant. Lines are group means; ribbons are SEM. Gray vertical bands mark significant cluster-permutation intervals and are labelled with cluster p-value and time range.",
            section,
        )
        report.add_figure(
            participant_difference_traces(
                roi_diff, rt, roi_subs, label + ": individual participant differences"
            ),
            str(figs / f"{tag}_individual_difference_traces.png"),
            label + ": individual stim - no-stim traces",
            f"Only participants retaining every required ROI channel are shown (N={len(roi_subs)}).",
            section,
        )
        effect_fig, effect_values = participant_window_effects(
            roi_diff, roi_subs, label + ": participant mean effects"
        )
        report.add_figure(
            effect_fig,
            str(figs / f"{tag}_participant_mean_effects.png"),
            label + ": participant-level mean 0.2-1.2 s effects",
            f"Complete-case ROI N={len(roi_subs)}. Each point is one participant's mean stim - no-stim alpha effect.",
            section,
        )
        roi_results[label]["participant_window_mean_effects"] = {
            f"sub-{sub}": float(v) for sub, v in zip(roi_subs, effect_values)
        }
        print(f"\n{label}: N={len(roi_subs)} included; excluded={excluded_subs}")
        for r in all_roi:
            print(
                f"{label} cluster {r['cluster']}: stat={r['cluster_stat']:.4f}, p={r['p']:.4f}, samples={r['n_samples']}, time={r['t_start']:.3f}-{r['t_end']:.3f}"
            )
    # audit
    audit = {
        "subjects": subs,
        "n_subjects_requested": len(subs),
        "individual_alpha_peaks_hz": {x: float(peaks[x]) for x in subs},
        "peaks_csv": str(peaks_path),
        "peak_half_width_hz": a.peak_half_width,
        "window_s": WINDOW,
        "baseline_s": BASELINE,
        "contrast_power": "stim percent minus no-stim percent, percentage points",
        "cluster_forming_alpha_two_sided": CLUSTER_FORMING_ALPHA,
        "cluster_forming_alpha_each_tail": CLUSTER_FORMING_ALPHA / 2,
        "cluster_level_significance": PTHRESH,
        "tail": 0,
        "single_sensor_sampling": "sensor-specific complete cases: sensor good in stim and no-stim",
        "single_sensor_results": sensor_results,
        "sensor_fdr_requested": a.sensor_fdr,
        "sensor_fdr_significant_sensors": sig_fdr,
        "ROI8": roi_results.get("8-channel posterior/occipital ROI"),
        "ROI3": roi_results.get("3-channel posterior ROI"),
        "whole_scalp_spatiotemporal_test": "not run",
    }
    (out / "alpha_cluster_report_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    with (out / "sensorwise_clusters_all.csv").open("w", newline="") as f:
        fields = [
            "sensor",
            "n_subjects",
            "cluster",
            "p_cluster",
            "cluster_stat",
            "n_samples",
            "t_start",
            "t_end",
        ]
        if a.sensor_fdr:
            fields += ["p_fdr", "significant_fdr"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sensor_cluster_records)
    print("\nPDF report:", report.pdf_fname)
    print("Whole-scalp sensor x time test: not run")
    print("Single sensors with cluster p<=.05:", sig_uncorrected or "none")
    print(
        "Optional sensor FDR:",
        ("applied; significant=" + str(sig_fdr) if a.sensor_fdr else "not applied"),
    )
    for k, v in roi_results.items():
        print(k, ":", len(v["rows"]), "significant cluster(s)")


if __name__ == "__main__":
    main()
