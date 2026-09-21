#!/usr/bin/env python
"""Paired cluster-permutation tests of stimulation effects on alpha power.

Primary inferential window: 8-12 Hz, 0.2-1.2 s after cue onset.
The test is within-participant: stimulation minus no-stimulation.

By default, alpha power is percent-baseline corrected separately within each
condition using -0.3 to -0.1 s, matching the subject-level descriptive TFRs.
Use --power-mode raw to test unbaselined power instead.

Three complementary tests are run:
1) Spatio-temporal cluster test across sensors x time. This is the primary test
   for identifying significant groups of neighboring sensors and contiguous times.
2) One-dimensional time-cluster test for each sensor. Cluster p-values from all
   sensors are Benjamini-Hochberg FDR corrected; use this to describe individual
   sensors, not as a replacement for the spatial cluster test.
3) A priori posterior/occipital ROI time-cluster test using
   PO3, POz, PO4, O1, Oz, O2, PO7, PO8.

For the spatial and sensor-wise analyses, only EEG sensors that are good in BOTH
conditions for EVERY supplied participant are used. This keeps the paired design
at the requested whole-sample N rather than silently changing N by sensor.

Example
-------
%run S01_alpha_cluster_permutation.py --subjects 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.stats import permutation_cluster_1samp_test, combine_adjacency, fdr_correction

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parents[1]
SUBJECT_DIR = ANALYSIS_DIR / "subject" / "EEG_all_channels"
if str(SUBJECT_DIR) not in sys.path:
    sys.path.insert(0, str(SUBJECT_DIR))
from pipeline_config import CONDITIONS, resolve_project_root, stage_path

ALPHA = (8.0, 12.0)
TEST_WINDOW = (0.2, 1.2)
BASELINE = (-0.3, -0.1)
FREQS = np.arange(2.0, 31.0, 1.0)
N_CYCLES = FREQS / 2.0
TIME_BANDWIDTH = 2.0
DECIM = 2
ROI8 = ("PO3", "POz", "PO4", "O1", "Oz", "O2", "PO7", "PO8")
ALPHA_LEVEL = 0.05

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="SpAtt")
    p.add_argument("--run", default="01")
    p.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    p.add_argument("--project-root", default=None)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--power-mode", choices=["percent", "raw"], default="percent",
                   help="percent = condition-wise -0.3:-0.1 baseline; raw = unbaselined power")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def load_epochs(root, s, a):
    out = {}
    for c in CONDITIONS:
        f = stage_path(root, s, a.session, a.task, a.run, c, "clean", "epo")
        if not f.exists():
            raise FileNotFoundError(f"Missing cleaned epochs: {f}")
        ep = mne.read_epochs(f, preload=True, verbose=False)
        keys = [k for k in ("cue_onset_right", "cue_onset_left") if k in ep.event_id]
        out[c] = ep[keys] if keys else ep
    return out

def good_pair_channels(pair):
    st, no = pair["stim"], pair["no-stim"]
    return [ch for ch in st.copy().pick("eeg").ch_names
            if ch in no.ch_names and ch not in st.info["bads"] and ch not in no.info["bads"]]

def alpha_tfr(ep, channels, jobs, power_mode):
    t = ep.copy().pick(channels).compute_tfr(
        method="multitaper", freqs=FREQS, n_cycles=N_CYCLES,
        time_bandwidth=TIME_BANDWIDTH, use_fft=True, return_itc=False,
        average=True, decim=DECIM, n_jobs=jobs, verbose=False)
    if power_mode == "percent":
        t.apply_baseline(BASELINE, mode="percent")
    fi = (t.freqs >= ALPHA[0]) & (t.freqs <= ALPHA[1])
    ti = (t.times >= TEST_WINDOW[0]) & (t.times <= TEST_WINDOW[1])
    # channels x time: alpha averaged across 8-12 Hz
    return t.data[:, fi][:, :, ti].mean(axis=1), t.times[ti], t.info

def cluster_summary(T, clusters, pvals, times, channels=None):
    rows = []
    for i, (cl, p) in enumerate(zip(clusters, pvals)):
        if p > ALPHA_LEVEL:
            continue
        mask = np.asarray(cl, dtype=bool)
        if channels is None:
            tids = np.where(mask)[0]
            rows.append(dict(cluster=i, p=float(p), t_start=float(times[tids].min()),
                             t_end=float(times[tids].max()), sensors="ROI"))
        else:
            tids, cids = np.where(mask)
            used = [channels[j] for j in sorted(set(cids))]
            rows.append(dict(cluster=i, p=float(p), t_start=float(times[tids].min()),
                             t_end=float(times[tids].max()), sensors=", ".join(used)))
    return rows

def run_time_cluster(X, seed):
    return permutation_cluster_1samp_test(
        X, n_permutations="all", threshold=None, tail=0,
        adjacency=None, out_type="mask", seed=seed, verbose=False)

def main():
    a = parse_args()
    subjects = [s.removeprefix("sub-") for s in a.subjects]
    if len(subjects) != 16:
        print(f"WARNING: you supplied {len(subjects)} participants, not 16.")
    root = resolve_project_root(a.platform, a.project_root)
    epochs = {s: load_epochs(root, s, a) for s in subjects}
    good = {s: good_pair_channels(epochs[s]) for s in subjects}

    # Strict complete-case sensor set for a true N=len(subjects) paired spatial test.
    first_order = epochs[subjects[0]]["stim"].copy().pick("eeg").ch_names
    common = [ch for ch in first_order if all(ch in good[s] for s in subjects)]
    if not common:
        raise RuntimeError("No EEG sensor is good in both conditions for every participant.")
    print(f"Whole-sample common good EEG sensors: {len(common)}")
    print(", ".join(common))

    stim, nostim = [], []
    times = None
    info = None
    for s in subjects:
        st, t, inf = alpha_tfr(epochs[s]["stim"], common, a.n_jobs, a.power_mode)
        no, t2, _ = alpha_tfr(epochs[s]["no-stim"], common, a.n_jobs, a.power_mode)
        if not np.allclose(t, t2):
            raise RuntimeError(f"Stim/no-stim time mismatch in sub-{s}")
        stim.append(st); nostim.append(no); times=t; info=inf
    stim = np.asarray(stim)       # subjects x channels x time
    nostim = np.asarray(nostim)
    diff = stim - nostim

    out = root/"data"/"BIDS"/"derivatives"/"group"/"EEG_all_channels_alpha_cluster"
    out.mkdir(parents=True, exist_ok=True)

    # 1) PRIMARY spatio-temporal test: X = subjects x time x channels.
    sensor_adj, adj_names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    if list(adj_names) != list(common):
        order = [adj_names.index(ch) for ch in common]
        sensor_adj = sensor_adj[order][:, order]
    adjacency = combine_adjacency(len(times), sensor_adj)
    X = diff.transpose(0, 2, 1)
    T, clusters, pvals, H0 = permutation_cluster_1samp_test(
        X, n_permutations="all", threshold=None, tail=0, adjacency=adjacency,
        out_type="mask", seed=a.seed, n_jobs=a.n_jobs, verbose=True)
    spatial_rows = cluster_summary(T, clusters, pvals, times, common)

    # 2) Sensor-wise time clusters, then FDR across ALL sensor-cluster p-values.
    sensor_cluster_records = []
    all_ps = []
    for ci, ch in enumerate(common):
        Ts, cls, ps, _ = run_time_cluster(diff[:, ci, :], a.seed)
        for k, (cl, p) in enumerate(zip(cls, ps)):
            tids = np.where(np.asarray(cl, bool))[0]
            rec = dict(sensor=ch, cluster=k, p_uncorrected=float(p),
                       t_start=float(times[tids].min()), t_end=float(times[tids].max()),
                       statistic_peak=float(Ts[tids][np.argmax(np.abs(Ts[tids]))]))
            sensor_cluster_records.append(rec); all_ps.append(float(p))
    if all_ps:
        reject, qvals = fdr_correction(np.asarray(all_ps), alpha=ALPHA_LEVEL, method="indep")
        for rec, rej, q in zip(sensor_cluster_records, reject, qvals):
            rec["p_fdr"] = float(q); rec["significant_fdr"] = bool(rej)

    # 3) Predefined ROI8 test. Require all eight channels to be common-good in all subjects.
    roi_channels = [ch for ch in ROI8 if ch in common]
    if len(roi_channels) < len(ROI8):
        missing = [ch for ch in ROI8 if ch not in roi_channels]
        print("WARNING: ROI8 cannot use all eight channels in the full N sample. Missing:", missing)
    roi_idx = [common.index(ch) for ch in roi_channels]
    roi_rows = []
    if roi_idx:
        roi_diff = diff[:, roi_idx, :].mean(axis=1)
        Tr, clr, pr, _ = run_time_cluster(roi_diff, a.seed)
        roi_rows = cluster_summary(Tr, clr, pr, times)

    # Save auditable tables.
    with (out/"spatiotemporal_significant_clusters.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=["cluster","p","t_start","t_end","sensors"]);w.writeheader();w.writerows(spatial_rows)
    with (out/"sensorwise_time_clusters_fdr.csv").open("w", newline="") as f:
        fields=["sensor","cluster","p_uncorrected","p_fdr","significant_fdr","t_start","t_end","statistic_peak"]
        w=csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows(sensor_cluster_records)
    with (out/"roi8_significant_clusters.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=["cluster","p","t_start","t_end","sensors"]);w.writeheader();w.writerows(roi_rows)

    # Plot mean alpha difference and mark sensors belonging to any significant spatial cluster.
    sig_sensors = sorted({ch for row in spatial_rows for ch in row["sensors"].split(", ") if ch})
    mean_diff = diff.mean(axis=(0,2))
    fig, ax = plt.subplots(figsize=(8,6))
    im,_ = mne.viz.plot_topomap(mean_diff, info, axes=ax, show=False, contours=0)
    ax.set_title(f"Stim - no-stim alpha ({ALPHA[0]:g}-{ALPHA[1]:g} Hz), {TEST_WINDOW[0]:g}-{TEST_WINDOW[1]:g} s")
    fig.colorbar(im, ax=ax, label="Alpha power difference")
    fig.savefig(out/"alpha_mean_difference_topomap.png", dpi=200, bbox_inches="tight");plt.close(fig)

    meta = {
        "subjects": subjects, "n_subjects": len(subjects), "common_good_channels": common,
        "alpha_hz": list(ALPHA), "test_window_s": list(TEST_WINDOW),
        "power_mode": a.power_mode, "baseline_s": list(BASELINE) if a.power_mode=="percent" else None,
        "tfr": {"method":"multitaper","freqs_hz":FREQS.tolist(),"n_cycles":"frequency/2",
                "time_bandwidth":TIME_BANDWIDTH,"decim":DECIM},
        "contrast":"stim - no-stim", "test":"paired one-sample cluster permutation on within-subject differences",
        "permutations":"all possible sign flips supported by MNE", "tail":0, "alpha":ALPHA_LEVEL,
        "spatial_adjacency":"MNE EEG sensor adjacency + temporal adjacency",
        "sensorwise_multiple_comparison":"Benjamini-Hochberg FDR across all sensor-time cluster p-values",
        "roi8_requested":list(ROI8),"roi8_used":roi_channels,
        "n_significant_spatiotemporal_clusters":len(spatial_rows),
        "n_significant_roi_clusters":len(roi_rows),
        "significant_spatial_sensors":sig_sensors}
    (out/"analysis_parameters_and_summary.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("\nFinished.")
    print("Results:", out)
    print("Significant spatio-temporal clusters:", len(spatial_rows))
    print("Sensors appearing in significant spatial clusters:", sig_sensors or "none")
    print("Significant ROI8 time clusters:", len(roi_rows))

if __name__ == "__main__":
    main()
