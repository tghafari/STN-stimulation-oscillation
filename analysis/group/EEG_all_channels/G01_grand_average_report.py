#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G01: cue-locked all-channel EEG grand-average ERP and TFR analysis.

This group analysis uses only the final cleaned cue-locked epochs produced by
analysis/subject/EEG_all_channels/preprocessing. There is no concatenated-epoch
analysis. Each participant is averaged first and contributes equally to the
group result for every EEG channel that is available in BOTH stimulation
conditions after preprocessing. Missing channels are never interpolated here.

For every channel, the report lists the exact contributing subjects before the
analysis/methods section. ERP and TFR scalp-layout views therefore permit a
different number of contributing subjects at different sensors.

TFR parameters intentionally match analysis/group/EEG_posterior_channels/
G02_grand_average_report.py exactly:
    2-31.5 Hz in 0.5-Hz steps; multitaper; n_cycles=f/2;
    time-bandwidth=2; decim=2; FFT=True; ITC=False; trial average=True.
Condition displays use percent baseline correction from -0.3 to -0.1 s.
Difference handling matches the posterior grand-average script and is selected
at runtime. The normalized ratio always uses the original unbaselined power.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parents[1]
SUBJECT_ALL_DIR = ANALYSIS_DIR / "subject" / "EEG_all_channels"
UTILS_DIR = ANALYSIS_DIR / "utils"
for p in (SUBJECT_ALL_DIR, UTILS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline_config import CONDITIONS, resolve_project_root, stage_path  # noqa: E402
from pdf_report import ParticipantPDF  # noqa: E402

POSTERIOR = ("PO3", "POz", "PO4")
ERP_BASELINE = (-0.1, 0.0)
ERP_LP_HZ = 30.0
ERP_TMIN = -0.1
ERP_TMAX = 1.0

BASELINE = (-0.3, -0.1)
FREQS = np.arange(2.0, 32.0, 0.5)
N_CYCLES = FREQS / 2.0
TIME_BANDWIDTH = 2.0
DECIM = 2
PLOT_TMIN = -0.5
PLOT_TMAX = 1.5
ROBUST_PERCENTILE = 98.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", required=True,
                   help="Subjects, e.g. --subjects 115 116 118")
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="SpAtt")
    p.add_argument("--run", default="01")
    p.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    p.add_argument("--project-root", default=None)
    p.add_argument("--n-jobs", type=int, default=4)
    return p.parse_args()


def get_difference_baseline_choice():
    while True:
        choice = input("\nApply percent baseline correction (-0.3 to -0.1 s) "
                       "before calculating the DIFFERENCE TFR? (y/n): ").strip().lower()
        if choice in {"y", "yes"}:
            return True
        if choice in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


def load_clean_epochs(root, subject, a):
    out = {}
    for condition in CONDITIONS:
        path = stage_path(root, subject, a.session, a.task, a.run,
                          condition, "clean", "epo")
        if not path.exists():
            raise FileNotFoundError(f"Missing final cleaned epochs for sub-{subject} "
                                    f"{condition}: {path}")
        ep = mne.read_epochs(path, preload=True)
        keep = [k for k in ("cue_onset_right", "cue_onset_left") if k in ep.event_id]
        out[condition] = ep[keep] if keep else ep
    return out


def common_good_eeg(epoch_pair):
    stim = epoch_pair["stim"]
    nostim = epoch_pair["no-stim"]
    stim_eeg = stim.copy().pick("eeg").ch_names
    return [ch for ch in stim_eeg
            if ch in nostim.ch_names
            and ch not in stim.info["bads"]
            and ch not in nostim.info["bads"]]


def make_evoked(ep, picks):
    ev = ep.copy().pick(picks).average(method="mean")
    ev.filter(None, ERP_LP_HZ)
    ev.apply_baseline(ERP_BASELINE)
    ev.crop(ERP_TMIN, ERP_TMAX)
    return ev


def compute_tfr(ep, picks, jobs):
    return ep.copy().pick(picks).compute_tfr(
        method="multitaper",
        freqs=FREQS,
        return_itc=False,
        average=True,
        decim=DECIM,
        n_jobs=jobs,
        n_cycles=N_CYCLES,
        time_bandwidth=TIME_BANDWIDTH,
        use_fft=True,
        zero_mean=True,
    )


def choose_template_info(subject_epochs, channels):
    """Build EEG Info in one stable channel order with montage metadata."""
    first_subject = next(iter(subject_epochs))
    for condition in CONDITIONS:
        info = subject_epochs[first_subject][condition].copy().pick("eeg").info
        ordered = [ch for ch in info.ch_names if ch in channels]
        remaining = [ch for ch in channels if ch not in ordered]
        if not remaining:
            return mne.pick_info(info, [info.ch_names.index(ch) for ch in ordered], copy=True)
    ordered = []
    source_infos = {}
    for s, pair in subject_epochs.items():
        info = pair["stim"].copy().pick("eeg").info
        source_infos[s] = info
        for ch in info.ch_names:
            if ch in channels and ch not in ordered:
                ordered.append(ch)
    template = None
    for info in source_infos.values():
        if all(ch in info.ch_names for ch in ordered):
            template = mne.pick_info(info, [info.ch_names.index(ch) for ch in ordered], copy=True)
            break
    if template is None:
        template = mne.create_info(ordered,
                                   sfreq=subject_epochs[first_subject]["stim"].info["sfreq"],
                                   ch_types="eeg")
        try:
            template.set_montage(subject_epochs[first_subject]["stim"].get_montage(),
                                 on_missing="ignore")
        except Exception:
            pass
    return template


def channelwise_evoked(subject_evokeds, subjects_by_channel, template_info, channels, condition):
    data = []
    for ch in channels:
        values = []
        for s in subjects_by_channel[ch]:
            ev = subject_evokeds[s][condition]
            values.append(ev.copy().pick([ch]).data[0])
        if not values:
            raise RuntimeError(f"No ERP data for {ch}")
        data.append(np.mean(values, axis=0))
    first_s = subjects_by_channel[channels[0]][0]
    times = subject_evokeds[first_s][condition].times
    info = template_info.copy()
    evoked = mne.EvokedArray(np.asarray(data), info, tmin=float(times[0]),
                             nave=len(subject_evokeds), comment=f"grand {condition}")
    return evoked


def channelwise_tfr(subject_tfrs, subjects_by_channel, template_info, channels, condition):
    data = []
    for ch in channels:
        values = []
        for s in subjects_by_channel[ch]:
            tfr = subject_tfrs[s][condition]
            values.append(tfr.copy().pick([ch]).data[0])
        if not values:
            raise RuntimeError(f"No TFR data for {ch}")
        data.append(np.mean(values, axis=0))
    first_s = subjects_by_channel[channels[0]][0]
    ref = subject_tfrs[first_s][condition]
    info = template_info.copy()
    return mne.time_frequency.AverageTFRArray(
        info, np.asarray(data), ref.times, ref.freqs,
        nave=len(subject_tfrs), comment=f"grand {condition}")


def robust_vlim(tfr):
    ti = (tfr.times >= PLOT_TMIN) & (tfr.times <= PLOT_TMAX)
    fi = (tfr.freqs >= FREQS.min()) & (tfr.freqs <= 31.5)
    x = np.asarray(tfr.data)[:, fi][:, :, ti]
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return (None, None)
    vmax = float(np.percentile(np.abs(finite), ROBUST_PERCENTILE))
    if not np.isfinite(vmax) or vmax == 0:
        return (None, None)
    return (-vmax, vmax)


def scalp_tfr(tfr, vlim):
    fig = tfr.plot_topo(tmin=PLOT_TMIN, tmax=PLOT_TMAX, fmin=2, fmax=31.5,
                        baseline=None, mode=None, vlim=vlim, cmap="RdBu_r", show=False)
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
    return fig


def posterior_tfr(tfr, posterior, title, vlim):
    fig, axes = plt.subplots(1, len(posterior), figsize=(5 * len(posterior), 4),
                             constrained_layout=True)
    axes = [axes] if len(posterior) == 1 else list(axes)
    for ax, ch in zip(axes, posterior):
        tfr.plot(picks=ch, tmin=PLOT_TMIN, tmax=PLOT_TMAX,
                 fmin=2, fmax=31.5, baseline=None, mode=None, axes=ax,
                 show=False, colorbar=True, vlim=vlim, cmap="RdBu_r")
        ax.set_title(ch)
    fig.suptitle(title)
    return fig


def subject_level_posterior_mean_tfr(subject_tfrs, condition_or_contrast,
                                     subjects, apply_diff_baseline=False):
    """Average available PO3/POz/PO4 within each subject, then across subjects."""
    subject_roi = []
    roi_subjects = []
    for s in subjects:
        available = [ch for ch in POSTERIOR
                     if ch in subject_tfrs[s]["stim"].ch_names
                     and ch in subject_tfrs[s]["no-stim"].ch_names]
        if not available:
            continue
        stim = subject_tfrs[s]["stim"].copy().pick(available)
        nostim = subject_tfrs[s]["no-stim"].copy().pick(available)
        if condition_or_contrast == "stim":
            x = stim.copy().apply_baseline(BASELINE, mode="percent")
        elif condition_or_contrast == "no-stim":
            x = nostim.copy().apply_baseline(BASELINE, mode="percent")
        elif condition_or_contrast == "difference":
            if apply_diff_baseline:
                stim.apply_baseline(BASELINE, mode="percent")
                nostim.apply_baseline(BASELINE, mode="percent")
            x = stim.copy()
            x.data = stim.data - nostim.data
        elif condition_or_contrast == "ratio":
            x = stim.copy()
            x.data = (stim.data - nostim.data) / (stim.data + nostim.data + np.finfo(float).eps)
        else:
            raise ValueError(condition_or_contrast)
        x.data = x.data.mean(axis=0, keepdims=True)
        x.info = mne.pick_info(x.info, [0], copy=True)
        x.info["chs"][0]["ch_name"] = "Posterior mean"
        x.info["ch_names"][0] = "Posterior mean"
        subject_roi.append(x)
        roi_subjects.append(s)
    if not subject_roi:
        return None, []
    roi = subject_roi[0].copy()
    roi.data = np.mean([x.data for x in subject_roi], axis=0)
    roi.nave = len(subject_roi)
    return roi, roi_subjects


def plot_roi_tfr(roi, title, vlim):
    fig = roi.plot(picks="Posterior mean", tmin=PLOT_TMIN, tmax=PLOT_TMAX,
                   fmin=2, fmax=31.5, baseline=None, mode=None, show=False,
                   colorbar=True, vlim=vlim, cmap="RdBu_r")
    fig = fig[0] if isinstance(fig, list) else fig
    fig.axes[0].set_title(title)
    return fig


def add_tfr_views(report, fig_dir, tfr, stem, title, caption, posterior,
                  subject_tfrs, subjects, contrast, apply_diff_baseline=False):
    vlim = robust_vlim(tfr)
    scale_text = (f"Shared robust symmetric color scale: {vlim[0]:.4g} to {vlim[1]:.4g}."
                  if None not in vlim else "Automatic color scale used.")
    report.add_figure(scalp_tfr(tfr, vlim), str(fig_dir / f"{stem}_all_channels_scalp.png"),
                      f"{title}: all channels in scalp layout",
                      caption + " " + scale_text +
                      " Each channel's grand average includes only subjects with that channel.",
                      "Time-frequency analysis")
    if posterior:
        report.add_figure(posterior_tfr(tfr, posterior, title, vlim),
                          str(fig_dir / f"{stem}_PO3_POz_PO4.png"),
                          f"{title}: PO3, POz and PO4",
                          caption + " " + scale_text + " Posterior order: PO3 | POz | PO4.",
                          "Time-frequency analysis")
        roi, roi_subjects = subject_level_posterior_mean_tfr(
            subject_tfrs, contrast, subjects, apply_diff_baseline)
        if roi is not None:
            roi_vlim = robust_vlim(roi)
            report.add_figure(plot_roi_tfr(roi, f"{title}: posterior mean", roi_vlim),
                              str(fig_dir / f"{stem}_posterior_mean.png"),
                              f"{title}: mean of PO3, POz and PO4",
                              caption + " Posterior channels were first averaged within each subject "
                              f"using whichever of PO3/POz/PO4 were available; then subjects were grand-averaged (n={len(roi_subjects)}).",
                              "Time-frequency analysis")
    return vlim


def main():
    a = parse_args()
    subjects = [s.removeprefix("sub-") for s in a.subjects]
    apply_diff_baseline = get_difference_baseline_choice()
    root = resolve_project_root(a.platform, a.project_root)

    group_root = root / "derivatives" / "reports" / "group" / "EEG_all_channels_grand_average"
    fig_dir = group_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    group_deriv = root / "data" / "BIDS" / "derivatives" / "group" / "EEG_all_channels_grand_average"
    group_deriv.mkdir(parents=True, exist_ok=True)
    report_id = "group_all_channels_" + "_".join(subjects)
    report = ParticipantPDF(str(group_root), report_id)

    epochs = {s: load_clean_epochs(root, s, a) for s in subjects}
    subject_good = {s: common_good_eeg(epochs[s]) for s in subjects}
    channels = []
    for s in subjects:
        for ch in subject_good[s]:
            if ch not in channels:
                channels.append(ch)
    if not channels:
        raise RuntimeError("No EEG channel is available in both conditions for any subject.")

    subjects_by_channel = {ch: [s for s in subjects if ch in subject_good[s]] for ch in channels}
    availability_lines = []
    for ch in channels:
        availability_lines.append(
            f"{ch} (n={len(subjects_by_channel[ch])}): " +
            ", ".join(f"sub-{s}" for s in subjects_by_channel[ch]))
    report.add_text("Subjects included", ", ".join(f"sub-{s}" for s in subjects),
                    "Group overview")
    report.add_text("Subjects contributing to each available EEG channel",
                    "\n".join(availability_lines), "Group overview")

    csv_path = group_deriv / f"{report_id}_subjects_by_channel.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel", "n_subjects", "subjects"])
        for ch in channels:
            w.writerow([ch, len(subjects_by_channel[ch]),
                        ";".join(f"sub-{s}" for s in subjects_by_channel[ch])])

    diff_text = ("Stimulation and no-stimulation TFRs were percent-baseline corrected "
                 "separately before subtraction."
                 if apply_diff_baseline else
                 "The stimulation-minus-no-stimulation TFR was calculated from unbaselined power.")
    manuscript = (
        "EEG group analysis. Final cleaned cue-locked epochs from the all-channel preprocessing pipeline were analysed. "
        "Only cue-onset epochs were retained; cue-left and cue-right trials were combined within stimulation condition. "
        "No epochs were concatenated across participants. For every sensor, only participants in whom that EEG channel "
        "was retained and marked good in both stimulation and no-stimulation datasets contributed to that sensor's grand average; "
        "missing channels were not interpolated at the group stage. Participant-level ERPs were formed by arithmetic trial averaging, "
        "low-pass filtering at 30 Hz, baseline correction from -0.1 to 0 s, and display from -0.1 to 1.0 s. Participant ERPs were then "
        "averaged across the eligible participants independently for each sensor. Time-frequency power was estimated separately for each "
        "participant and stimulation condition with a multitaper transform from 2 to 31.5 Hz in 0.5-Hz steps, n_cycles=f/2, "
        "time-bandwidth=2, FFT enabled, ITC disabled, trial averaging enabled, and decimation=2. Stimulation and no-stimulation power "
        "were displayed as percent change from the -0.3 to -0.1 s baseline. " + diff_text + " The normalized contrast "
        "(stimulation - no stimulation)/(stimulation + no stimulation) was calculated from the original unbaselined power. "
        "For each TFR result, all-channel scalp-layout and enlarged PO3/POz/PO4 views use the same underlying group TFR, time/frequency "
        "limits, baseline state, colormap, and one robust symmetric scale based on the 98th percentile of absolute displayed values, "
        "preventing outlying sensors from making the scalp-layout panels visually blank. Posterior-mean TFRs were calculated by first "
        "averaging the available PO3/POz/PO4 channels within each participant and then averaging those participant-level posterior means."
    )
    report.add_text("Analysis (manuscript style)", manuscript, "Analysis")
    report.add_text("Exact analysis parameters",
                    f"Epochs: cue onset only, original window -0.5 to +1.6 s; attention-left/right combined.\n"
                    f"ERP: low-pass {ERP_LP_HZ:g} Hz; baseline {ERP_BASELINE}; display {ERP_TMIN} to {ERP_TMAX} s.\n"
                    f"TFR: multitaper; 2-31.5 Hz, 0.5-Hz steps; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; "
                    f"FFT=True; ITC=False; average=True; decim={DECIM}; condition baseline {BASELINE} percent.\n"
                    f"Difference baseline correction: {apply_diff_baseline}. Ratio baseline correction: never.\n"
                    f"Scales: symmetric +/- {ROBUST_PERCENTILE:g}th percentile of absolute displayed values per TFR result, shared between all-channel and posterior views.",
                    "Analysis")

    template_info = choose_template_info(epochs, channels)
    channels = template_info.ch_names
    subjects_by_channel = {ch: subjects_by_channel[ch] for ch in channels}

    subject_evokeds = {}
    for s in subjects:
        subject_evokeds[s] = {}
        for c in CONDITIONS:
            subject_evokeds[s][c] = make_evoked(epochs[s][c], subject_good[s])

    grand_evoked = {c: channelwise_evoked(subject_evokeds, subjects_by_channel,
                                          template_info, channels, c)
                    for c in CONDITIONS}
    for c in CONDITIONS:
        mne.write_evokeds(group_deriv / f"{report_id}_{c}_cue_grand-ave.fif",
                          grand_evoked[c], overwrite=True)

    compare = {"No stimulation": grand_evoked["no-stim"],
               "Stimulation": grand_evoked["stim"]}
    fig_topo = mne.viz.plot_compare_evokeds(compare, picks=channels, combine=None,
                                            axes="topo", show=False, ci=False,
                                            truncate_xaxis=False, truncate_yaxis=False,
                                            legend=True)
    if isinstance(fig_topo, list):
        fig_topo = fig_topo[0]
    report.add_figure(fig_topo, str(fig_dir / "ERP_stim_vs_no_stim_all_channels_scalp.png"),
                      "Cue-locked ERP: stimulation vs no stimulation, all channels in scalp layout",
                      "Each sensor is grand-averaged only across subjects retaining that sensor in both conditions. "
                      f"ERP low-pass {ERP_LP_HZ:g} Hz; baseline {ERP_BASELINE}; cue onset=0 s.", "ERP")

    posterior = [ch for ch in POSTERIOR if ch in channels]
    if posterior:
        fig_post, axes = plt.subplots(1, len(posterior), figsize=(5 * len(posterior), 4),
                                      constrained_layout=True)
        axes = [axes] if len(posterior) == 1 else list(axes)
        for ax, ch in zip(axes, posterior):
            mne.viz.plot_compare_evokeds(compare, picks=ch, combine=None, axes=ax,
                                         show=False, ci=False, truncate_xaxis=False,
                                         truncate_yaxis=False)
            ax.axvline(0, color="k", linestyle="--", linewidth=1)
            ax.set_xlim(ERP_TMIN, ERP_TMAX)
            ax.set_title(f"{ch} (n={len(subjects_by_channel[ch])})")
        fig_post.suptitle("Grand-average cue-locked ERP: PO3 | POz | PO4")
        report.add_figure(fig_post, str(fig_dir / "ERP_stim_vs_no_stim_PO3_POz_PO4.png"),
                          "Cue-locked ERP: PO3, POz and PO4",
                          "Posterior channels are shown separately in anatomical left-to-right order. "
                          "Each channel uses only its available subjects.", "ERP")

    subject_tfrs = {}
    for s in subjects:
        subject_tfrs[s] = {}
        for c in CONDITIONS:
            print(f"Computing TFR: sub-{s}, {c}, {len(subject_good[s])} common good EEG channels")
            subject_tfrs[s][c] = compute_tfr(epochs[s][c], subject_good[s], a.n_jobs)

    grand_raw = {c: channelwise_tfr(subject_tfrs, subjects_by_channel,
                                    template_info, channels, c)
                 for c in CONDITIONS}
    for c in CONDITIONS:
        grand_raw[c].save(group_deriv / f"{report_id}_{c}_cue_grand-tfr.h5", overwrite=True)

    display_no = grand_raw["no-stim"].copy().apply_baseline(BASELINE, mode="percent")
    display_stim = grand_raw["stim"].copy().apply_baseline(BASELINE, mode="percent")

    if apply_diff_baseline:
        stim_for_diff = grand_raw["stim"].copy().apply_baseline(BASELINE, mode="percent")
        no_for_diff = grand_raw["no-stim"].copy().apply_baseline(BASELINE, mode="percent")
    else:
        stim_for_diff = grand_raw["stim"].copy()
        no_for_diff = grand_raw["no-stim"].copy()
    diff = stim_for_diff.copy()
    diff.data = stim_for_diff.data - no_for_diff.data

    ratio = grand_raw["stim"].copy()
    ratio.data = ((grand_raw["stim"].data - grand_raw["no-stim"].data) /
                  (grand_raw["stim"].data + grand_raw["no-stim"].data + np.finfo(float).eps))
    diff.save(group_deriv / f"{report_id}_stim-minus-no-stim_grand-tfr.h5", overwrite=True)
    ratio.save(group_deriv / f"{report_id}_stim-normalized-difference_grand-tfr.h5", overwrite=True)

    caption_condition = (f"Cue-locked combined attention-left/right grand-average TFR; multitaper 2-31.5 Hz in 0.5-Hz steps; "
                         f"n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; decim={DECIM}; percent baseline {BASELINE}; cue onset=0 s.")
    add_tfr_views(report, fig_dir, display_no, "TFR_no_stim",
                  "No stimulation: combined attention-left/right TFR", caption_condition,
                  posterior, subject_tfrs, subjects, "no-stim", apply_diff_baseline)
    add_tfr_views(report, fig_dir, display_stim, "TFR_stim",
                  "Stimulation: combined attention-left/right TFR", caption_condition,
                  posterior, subject_tfrs, subjects, "stim", apply_diff_baseline)
    diff_caption = (f"Cue-locked stimulation - no-stimulation grand-average TFR; multitaper 2-31.5 Hz; "
                    f"difference baseline correction={apply_diff_baseline}; cue onset=0 s.")
    add_tfr_views(report, fig_dir, diff, "TFR_stim_minus_no_stim",
                  "TFR: stimulation - no stimulation", diff_caption,
                  posterior, subject_tfrs, subjects, "difference", apply_diff_baseline)
    ratio_caption = ("Cue-locked normalized TFR contrast (stimulation - no stimulation)/(stimulation + no stimulation), "
                     "calculated from original unbaselined power; no baseline correction; cue onset=0 s.")
    add_tfr_views(report, fig_dir, ratio, "TFR_stim_normalized_difference",
                  "TFR: (stimulation - no stimulation) / (stimulation + no stimulation)", ratio_caption,
                  posterior, subject_tfrs, subjects, "ratio", apply_diff_baseline)

    details = {
        "subjects": [f"sub-{s}" for s in subjects],
        "subjects_by_channel": {ch: [f"sub-{s}" for s in subjects_by_channel[ch]] for ch in channels},
        "cue_only": True,
        "concatenated_epochs_used": False,
        "attention_conditions_combined": ["cue_onset_right", "cue_onset_left"],
        "erp": {"low_pass_hz": ERP_LP_HZ, "baseline_s": list(ERP_BASELINE),
                "display_s": [ERP_TMIN, ERP_TMAX]},
        "tfr": {"method": "multitaper", "frequencies_hz": FREQS.tolist(),
                "frequency_step_hz": 0.5, "n_cycles": "frequency / 2",
                "time_bandwidth": TIME_BANDWIDTH, "use_fft": True,
                "return_itc": False, "average_trials": True, "decim": DECIM,
                "condition_baseline_s": list(BASELINE), "condition_baseline_mode": "percent",
                "difference_baseline_correction": apply_diff_baseline,
                "ratio_baseline_correction": None,
                "ratio_formula": "(stim - no-stim) / (stim + no-stim)",
                "vlim_method": f"symmetric +/- {ROBUST_PERCENTILE:g}th percentile of absolute displayed values per result"},
        "posterior_order": list(POSTERIOR),
        "posterior_mean_method": "average available PO3/POz/PO4 within subject, then average subjects",
    }
    (group_deriv / f"{report_id}_analysis_details.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8")

    print(f"Group all-channel grand-average analysis complete. PDF: {report.pdf_fname}")
    print(f"Channel availability CSV: {csv_path}")


if __name__ == "__main__":
    main()
