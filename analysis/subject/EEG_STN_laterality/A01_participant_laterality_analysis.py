#!/usr/bin/env python
"""Participant-specific STN laterality TFR analysis, using confirmed block labels."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
SUBJECT_DIR = HERE.parent
PIPELINE_DIR = SUBJECT_DIR / "EEG_all_channels"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
from pipeline_config import CONDITIONS, resolve_project_root, stage_path

LEFT_POST = ("PO3", "PO7", "O1")
RIGHT_POST = ("PO4", "PO8", "O2")
POSTERIOR = LEFT_POST + RIGHT_POST
FREQS = np.arange(2.0, 32.0, 0.5)
N_CYCLES = FREQS / 2.0
TIME_BANDWIDTH = 2.0
DECIM = 2
ALPHA = (8.0, 12.0)
ALPHA_TIME = (0.3, 0.8)
TMIN_PLOT, TMAX_PLOT = -0.5, 1.5
BASELINE = (-0.3, -0.1)

def intervals(values):
    vals = [float(v) for v in values]
    if len(vals) not in (2, 4):
        raise ValueError(f"Expected 2 or 4 crop times, got {values}")
    return [(vals[i], vals[i+1]) for i in range(0, len(vals), 2)]

def load_clean_epochs(root, sub, args, condition):
    path = stage_path(root, sub, args.session, args.task, args.run, condition, "clean", "epo")
    if not path.exists():
        raise FileNotFoundError(f"Missing cleaned cue epochs for sub-{sub}, {condition}: {path}")
    epochs = mne.read_epochs(path, preload=True, verbose=False)
    keep = [name for name in ("cue_onset_left", "cue_onset_right") if name in epochs.event_id]
    if keep:
        epochs = epochs[keep]
    return epochs, path

def map_source_block_to_condition_time(block_start, block_end, crop_ranges):
    """Map source BIDS seconds to the concatenated condition-epoch time axis."""
    offset = 0.0
    mapped = []
    for start, end in crop_ranges:
        left = max(float(block_start), start)
        right = min(float(block_end), end)
        if right > left:
            mapped.append((offset + left - start, offset + right - start))
        offset += end - start
    return mapped

def subset_epochs_for_block(epochs, block_start, block_end, source_ranges, label):
    """Select cue epochs falling inside a source block after P01 condition concatenation."""
    sfreq = float(epochs.info["sfreq"])
    ranges = map_source_block_to_condition_time(block_start, block_end, source_ranges)
    if not ranges:
        raise ValueError(
            f"Block {label} [{block_start:.3f},{block_end:.3f}] s does not overlap the "
            f"preprocessing crop intervals {source_ranges}. Check the BIDS/crop-time correspondence."
        )
    # MNE keeps event sample indices from the concatenated, cropped Raw. The first
    # crop's original source-time offset is therefore the origin of the concatenated
    # condition time axis.
    source_origin = source_ranges[0][0]
    event_times = epochs.events[:, 0].astype(float) / sfreq - source_origin
    mask = np.zeros(len(epochs), dtype=bool)
    for a, b in ranges:
        mask |= (event_times >= a - 1.0/sfreq) & (event_times <= b + 1.0/sfreq)
    selected = epochs[mask]
    if len(selected) == 0:
        raise ValueError(
            f"No cleaned epochs mapped to {label} block [{block_start:.3f},{block_end:.3f}] s. "
            "The stored Epochs event samples may not share the BIDS/crop-table time origin; "
            "do not proceed until this mapping is checked."
        )
    return selected

def good_posterior_channels(stim, off):
    eeg_stim = set(stim.copy().pick("eeg").ch_names)
    eeg_off = set(off.copy().pick("eeg").ch_names)
    good = []
    for ch in POSTERIOR:
        if ch in eeg_stim and ch in eeg_off:
            if ch not in stim.info["bads"] and ch not in off.info["bads"]:
                good.append(ch)
    return good

def average_tfr_power(epochs, channels, n_jobs):
    if len(epochs) < 2:
        # Power is calculable with one trial, but report it clearly; don't fabricate inference.
        print(f"WARNING: only {len(epochs)} clean epochs in block.")
    x = epochs.copy().pick(channels).compute_tfr(
        method="multitaper", freqs=FREQS, n_cycles=N_CYCLES,
        time_bandwidth=TIME_BANDWIDTH, use_fft=True, zero_mean=True,
        return_itc=False, average=True, decim=DECIM, n_jobs=n_jobs, verbose=False,
    )
    return {ch: np.asarray(x.data[i]).copy() for i, ch in enumerate(x.ch_names)}, np.asarray(x.times)

def baseline_percent(power, times):
    arr = np.asarray(power)
    mask = (times >= BASELINE[0]) & (times <= BASELINE[1])
    if not mask.any():
        raise ValueError(f"No TFR samples fall in baseline interval {BASELINE}.")
    base = arr[:, mask].mean(axis=1, keepdims=True)
    denom = np.where(np.abs(base) < np.finfo(float).eps, 1.0, base)
    return 100.0 * (arr - base) / denom

def normalized_ratio(stim, off):
    return (stim - off) / np.maximum(stim + off, np.finfo(float).tiny)

def robust_symmetric_limit(arrays, percentile=98):
    finite = [np.asarray(a)[np.isfinite(a)].ravel() for a in arrays if np.asarray(a).size]
    finite = [x for x in finite if x.size]
    if not finite:
        return 1e-12
    value = float(np.percentile(np.abs(np.concatenate(finite)), percentile))
    return value if np.isfinite(value) and value > 0 else 1e-12

def roi_mean(channel_data, channels):
    available = [ch for ch in channels if ch in channel_data]
    if not available:
        return None, []
    return np.mean([channel_data[ch] for ch in available], axis=0), available

def make_tfr_grid(tfrs, times, title, scale, unit):
    """tfrs: mapping of subplot title -> frequency x time image."""
    fig, axes = plt.subplots(1, len(tfrs), figsize=(5.2*len(tfrs), 4.5), squeeze=False)
    im = None
    for ax, (name, data) in zip(axes[0], tfrs.items()):
        if data is None:
            ax.axis("off"); ax.set_title(f"{name}\nnot available"); continue
        im = ax.imshow(data, origin="lower", aspect="auto",
                       extent=[times[0], times[-1], FREQS[0], FREQS[-1]],
                       cmap="RdBu_r", vmin=-scale if scale else None, vmax=scale if scale else None)
        ax.axvline(0, color="k", linestyle="--", linewidth=0.7)
        ax.set_title(name); ax.set_xlabel("Time from cue (s)")
        ax.set_ylabel("Frequency (Hz)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.05, 0.92, 0.92])
    if im is not None:
        cax = fig.add_axes([0.935, 0.18, 0.018, 0.62])
        fig.colorbar(im, cax=cax, label=unit)
    return fig

def add_text_page(pdf, title, lines):
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(0.06, 0.94, title, fontsize=16, weight="bold", va="top")
    y = 0.88
    for line in lines:
        fig.text(0.06, y, str(line), fontsize=9.5, va="top", wrap=True)
        y -= 0.045
        if y < 0.08:
            pdf.savefig(fig); plt.close(fig)
            fig = plt.figure(figsize=(11.7, 8.3))
            y = 0.92
    pdf.savefig(fig); plt.close(fig)

def analyze_subject(root, sub, review_df, args):
    crop_file = SUBJECT_DIR / "stimulation_cropped_time.json"
    if not crop_file.exists():
        raise FileNotFoundError(f"Missing crop-time table: {crop_file}")
    crop_table = json.loads(crop_file.read_text(encoding="utf-8"))
    key = f"sub-{sub}"
    if key not in crop_table:
        raise KeyError(f"No crop intervals for {key} in {crop_file}")
    block_rows = review_df[(review_df["subject"].astype(str) == sub) &
                           (review_df["confirmed"].astype(str).str.lower().isin(["yes", "y", "true", "1"]))]
    if len(block_rows) != 4:
        raise ValueError(f"sub-{sub}: expected 4 confirmed BIDS block rows, got {len(block_rows)}")
    block_rows = block_rows.sort_values("block_index_chronological")
    labels = block_rows["confirmed_condition"].astype(str).str.strip().str.lower().tolist()
    if labels.count("off") != 2 or labels.count("stim_left") != 1 or labels.count("stim_right") != 1:
        if "exclude" in labels:
            raise ValueError(f"sub-{sub} is marked for exclusion; skip this participant.")
        raise ValueError(f"sub-{sub}: confirmed labels must be two off, one stim_left, one stim_right; got {labels}")

    stim_epochs, stim_path = load_clean_epochs(root, sub, args, "stim")
    off_epochs, off_path = load_clean_epochs(root, sub, args, "no-stim")
    good = good_posterior_channels(stim_epochs, off_epochs)
    if len(good) == 0:
        raise RuntimeError(f"sub-{sub}: none of the six lateral posterior sensors are available in both conditions.")
    if not any(ch in good for ch in LEFT_POST) or not any(ch in good for ch in RIGHT_POST):
        raise RuntimeError(f"sub-{sub}: need at least one available good sensor in each posterior hemisphere; found {good}.")
    crop_stim = intervals(crop_table[key]["stim"])
    crop_off = intervals(crop_table[key]["no-stim"])
    split_blocks = {}
    for _, row in block_rows.iterrows():
        condition = str(row["confirmed_condition"]).strip().lower()
        start, end = float(row["onset_s"]), float(row["offset_s"])
        if condition == "off":
            subset = subset_epochs_for_block(off_epochs, start, end, crop_off, f"sub-{sub} OFF block")
        else:
            subset = subset_epochs_for_block(stim_epochs, start, end, crop_stim, f"sub-{sub} {condition} block")
        split_blocks[condition if condition != "off" else f"off_{int(row['block_index_chronological'])}"] = subset

    # Compute each block separately. The OFF reference is the equal-weight mean of
    # the two OFF block TFRs, independent of how many accepted trials each has.
    block_power, times = {}, None
    for label, ep in split_blocks.items():
        block_power[label], this_times = average_tfr_power(ep, good, args.n_jobs)
        if times is None:
            times = this_times
        elif not np.allclose(times, this_times):
            raise ValueError(f"sub-{sub}: inconsistent TFR time axes between blocks.")
    off_labels = [f"off_{int(i)}" for i in block_rows.loc[block_rows["confirmed_condition"].astype(str).str.lower()=="off","block_index_chronological"]]
    off_ref = {ch: np.mean([block_power[k][ch] for k in off_labels], axis=0) for ch in good}
    left_power = block_power["stim_left"]
    right_power = block_power["stim_right"]

    # Per-sensor contrasts from original linear unbaselined power, then ROI averages.
    diffs = {
        "left_stim_ipsi": {ch: left_power[ch] - off_ref[ch] for ch in LEFT_POST if ch in good},
        "left_stim_contra": {ch: left_power[ch] - off_ref[ch] for ch in RIGHT_POST if ch in good},
        "right_stim_ipsi": {ch: right_power[ch] - off_ref[ch] for ch in RIGHT_POST if ch in good},
        "right_stim_contra": {ch: right_power[ch] - off_ref[ch] for ch in LEFT_POST if ch in good},
    }
    ratios = {
        "left_stim_ipsi": {ch: normalized_ratio(left_power[ch], off_ref[ch]) for ch in LEFT_POST if ch in good},
        "left_stim_contra": {ch: normalized_ratio(left_power[ch], off_ref[ch]) for ch in RIGHT_POST if ch in good},
        "right_stim_ipsi": {ch: normalized_ratio(right_power[ch], off_ref[ch]) for ch in RIGHT_POST if ch in good},
        "right_stim_contra": {ch: normalized_ratio(right_power[ch], off_ref[ch]) for ch in LEFT_POST if ch in good},
    }
    roi_diff, roi_ratio = {}, {}
    for name, chdata in diffs.items():
        roi_diff[name], _ = roi_mean(chdata, list(chdata))
        roi_ratio[name], _ = roi_mean(ratios[name], list(ratios[name]))

    # Per-condition lateral/posterior descriptive power maps are percent-baseline
    # visualizations only. These never feed Difference or Ratio.
    left_roi, left_chs = roi_mean(left_power, LEFT_POST)
    right_roi, right_chs = roi_mean(left_power, RIGHT_POST)
    rleft_roi, _ = roi_mean(right_power, LEFT_POST)
    rright_roi, _ = roi_mean(right_power, RIGHT_POST)
    off_left, _ = roi_mean(off_ref, LEFT_POST)
    off_right, _ = roi_mean(off_ref, RIGHT_POST)
    display = {
        "OFF ref / left posterior": baseline_percent(off_left, times),
        "OFF ref / right posterior": baseline_percent(off_right, times),
        "Left STN / left posterior": baseline_percent(left_roi, times),
        "Left STN / right posterior": baseline_percent(right_roi, times),
        "Right STN / left posterior": baseline_percent(rleft_roi, times),
        "Right STN / right posterior": baseline_percent(rright_roi, times),
    }
    # Summarize unbaselined linear power in the prespecified cue-locked alpha window.
    fmask = (FREQS >= ALPHA[0]) & (FREQS <= ALPHA[1])
    tmask = (times >= ALPHA_TIME[0]) & (times <= ALPHA_TIME[1])
    if not fmask.any() or not tmask.any():
        raise ValueError("Selected alpha band/time interval not represented in TFR.")
    alpha_rows = []
    for stim_label, power in (("OFF", off_ref), ("Left STN ON", left_power), ("Right STN ON", right_power)):
        for side_name, group in (("left_posterior", LEFT_POST), ("right_posterior", RIGHT_POST)):
            arrays = [power[ch] for ch in group if ch in power]
            value = float(np.mean([a[np.ix_(fmask, tmask)].mean() for a in arrays])) if arrays else np.nan
            alpha_rows.append({"subject": sub, "condition": stim_label, "posterior_side": side_name,
                               "alpha_hz": f"{ALPHA[0]}-{ALPHA[1]}",
                               "time_s": f"{ALPHA_TIME[0]}-{ALPHA_TIME[1]}",
                               "mean_unbaselined_linear_power": value})
    out = root / "derivatives" / "reports" / "group" / "EEG_STN_laterality"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"sub-{sub}_STN_laterality_report.pdf"
    summary_path = out / f"sub-{sub}_STN_laterality_alpha.csv"
    meta_path = out / f"sub-{sub}_STN_laterality_analysis.json"
    pd.DataFrame(alpha_rows).to_csv(summary_path, index=False)

    with PdfPages(report_path) as pdf:
        add_text_page(pdf, f"STN stimulation laterality — sub-{sub}", [
            "Purpose: describe participant-specific posterior alpha/TFR changes associated with left- and right-STN stimulation.",
            f"Confirmed block labels in chronological order: {list(zip(block_rows['block_index_chronological'].astype(int), labels))}",
            f"Cleaned epochs: stim={stim_path}; no-stim={off_path}",
            "OFF reference: equal-weight average of the two OFF block-level TFRs; chronology deliberately ignored as requested.",
            "Difference and ratio are calculated per sensor from original unbaselined linear power, then sensors are averaged within ipsilateral/contralateral ROIs.",
            "Left posterior ROI: PO3, PO7, O1. Right posterior ROI: PO4, PO8, O2.",
            f"Alpha summary: {ALPHA[0]}–{ALPHA[1]} Hz, {ALPHA_TIME[0]}–{ALPHA_TIME[1]} s after cue; absolute unbaselined power.",
            "Design limitation: one block per stimulated STN side and two OFF blocks. No conventional per-participant cluster-permutation significance test is claimed; the results are effect estimates/descriptive contrasts.",
            f"Channels available and used: {', '.join(good)}",
            f"Block trials: " + "; ".join(f"{k}={len(v)}" for k,v in split_blocks.items()),
        ])
        # Condition-wise descriptive views with percent baseline; contrasts stay raw.
        for title, channels_data in [
            ("Condition power: OFF reference (left/right posterior)", {"Left posterior": off_left, "Right posterior": off_right}),
            ("Condition power: Left STN ON (left/right posterior)", {"Left posterior": left_roi, "Right posterior": right_roi}),
            ("Condition power: Right STN ON (left/right posterior)", {"Left posterior": rleft_roi, "Right posterior": rright_roi}),
        ]:
            vals = [baseline_percent(a, times) for a in channels_data.values()]
            lim = robust_symmetric_limit(vals)
            fig = make_tfr_grid({k:v for k,v in zip(channels_data, vals)}, times, title, lim, "% baseline")
            pdf.savefig(fig); plt.close(fig)
        for metric, dct, unit in [("Raw difference (Stim - OFF)", roi_diff, "Power difference"),
                                  ("Normalized ratio (Stim - OFF)/(Stim + OFF)", roi_ratio, "Normalized ratio")]:
            values = [v for v in dct.values() if v is not None]
            lim = robust_symmetric_limit(values)
            for stim_side in ("left_stim", "right_stim"):
                tfrs = {
                    "Ipsilateral": dct.get(stim_side + "_ipsi"),
                    "Contralateral": dct.get(stim_side + "_contra"),
                }
                # Difference and ratio can have any sign, symmetric scales aid comparison.
                fig = make_tfr_grid(tfrs, times, f"{metric}: {stim_side.replace('_',' ')}", lim, unit)
                pdf.savefig(fig); plt.close(fig)
        # Channel-level contrasts per stimulation side and sensor, to inspect topography.
        for stim_side, power in (("Left STN", left_power), ("Right STN", right_power)):
            contrast = {ch: power[ch] - off_ref[ch] for ch in good}
            lim = robust_symmetric_limit(list(contrast.values()))
            fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
            for ax, ch in zip(axes.ravel(), [x for x in POSTERIOR if x in contrast]):
                im = ax.imshow(contrast[ch], origin="lower", aspect="auto",
                               extent=[times[0], times[-1], FREQS[0], FREQS[-1]],
                               cmap="RdBu_r", vmin=-lim, vmax=lim)
                ax.axvline(0, color="k", linestyle="--", linewidth=0.6)
                ax.set_title(f"{ch}: {stim_side} - OFF")
                ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Raw power difference")
            pdf.savefig(fig); plt.close(fig)

    metadata = {
        "subject": sub,
        "confirmed_blocks": block_rows.to_dict(orient="records"),
        "clean_epoch_files": {"stim": str(stim_path), "no_stim": str(off_path)},
        "block_epoch_counts": {k: len(v) for k,v in split_blocks.items()},
        "channels_used": good,
        "off_reference": "equal-weight mean of two OFF block-level raw TFRs",
        "difference": "per sensor: raw P_stim - raw P_off; then mean sensors",
        "ratio": "per sensor: (raw P_stim - raw P_off)/(raw P_stim + raw P_off); then mean sensors",
        "baseline": {"condition_visualization_only": list(BASELINE), "difference": "none", "ratio": "none"},
        "alpha_summary": {"frequencies_hz": list(ALPHA), "times_s": list(ALPHA_TIME), "baseline": "none"},
        "note": "Descriptive participant-level effects; one block per stimulation side does not support a robust conventional within-participant cluster permutation test.",
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"sub-{sub}: report {report_path}")
    print(f"sub-{sub}: alpha summary {summary_path}")
    return report_path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True)
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="SpAtt")
    p.add_argument("--run", default="01")
    p.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    p.add_argument("--project-root", default=None)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--confirmation-csv", required=True)
    a = p.parse_args()
    root = resolve_project_root(a.platform, a.project_root)
    review_df = pd.read_csv(a.confirmation_csv, dtype=str).fillna("")
    analyze_subject(root, a.subject.removeprefix("sub-"), review_df, a)

if __name__ == "__main__":
    main()
