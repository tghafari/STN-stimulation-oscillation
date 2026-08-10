#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os.path as op
from pathlib import Path

import mne
import numpy as np

from group_utils import (
    add_analysis_notes_section,
    add_subject_summary,
    ensure_dir,
    load_subject_list,
    make_report,
    plot_compare_evokeds_by_channel,
    subset_present_channels,
    read_subject_evokeds,
    read_subject_tfrs,
    save_subject_list,
)

# -----------------------
# Config
# -----------------------
PROJECT_ROOT = "/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/STN-in-PD"
BIDS_ROOT = op.join(PROJECT_ROOT, "data", "BIDS")

GROUP_REPORT_DIR = op.join(PROJECT_ROOT, "derivatives", "reports", "group", "grand_average")
GROUP_DERIV_DIR = op.join(BIDS_ROOT, "derivatives", "group", "grand_average")
SUBJECT_LIST_PATH = op.join(PROJECT_ROOT, "derivatives", "reports", "group", "subjects_for_group_analysis.json")

REPORT_NAME = "group_grand_average"
REPORT_TITLE = "Grand average across subjects"

DEFAULT_SUBJECTS = ["115", "116", "118", "119"]
OCCIPITAL_CHANNELS = ["PO3", "PO4", "POz"]
SESSION = "01"
TASK = "SpAtt"
RUN = "01"

BASELINE = (-0.3, -0.1)
FREQS = np.arange(2, 32, 0.5)
N_CYCLES = FREQS / 2.0
TIME_BANDWIDTH = 2.0
TFR_PARAMS = dict(
    method="multitaper",
    freqs=FREQS,
    return_itc=False,
    average=True,
    decim=2,
    n_jobs=4,
    verbose=True,
    n_cycles=N_CYCLES,
    time_bandwidth=TIME_BANDWIDTH,
    use_fft=True,
    zero_mean=True,
)

def build_grand_average_report(subjects):
    ensure_dir(GROUP_REPORT_DIR)
    ensure_dir(GROUP_DERIV_DIR)

    report = make_report(GROUP_REPORT_DIR, REPORT_NAME)
    report.add_text(
        "Group analysis summary",
        f"Report: {REPORT_TITLE}\nSubjects analysed: {', '.join('sub-' + s for s in subjects)}\nNumber of subjects: {len(subjects)}\nEach subject contributes equally.",
        "Group analysis",
    )
    add_subject_summary(report, subjects)

    save_subject_list(SUBJECT_LIST_PATH, subjects)

    # -----------------------
    # ERP grand averages
    # -----------------------
    evoked_by_condition = {"no-stim": {"cue": [], "grating": []}, "stim": {"cue": [], "grating": []}}

    for subject in subjects:
        evokeds = read_subject_evokeds(BIDS_ROOT, subject)
        for stim_label in ["no-stim", "stim"]:
            cue_evoked = evokeds[(stim_label, "cue")].copy()
            grating_evoked = evokeds[(stim_label, "grating")].copy()

            present = subset_present_channels(cue_evoked.ch_names, OCCIPITAL_CHANNELS)
            if not present:
                raise RuntimeError(f"No occipital channels present for sub-{subject} {stim_label}")
            cue_evoked.pick(present)
            grating_evoked.pick(present)

            evoked_by_condition[stim_label]["cue"].append(cue_evoked)
            evoked_by_condition[stim_label]["grating"].append(grating_evoked)

    grand_evoked = {"no-stim": {}, "stim": {}}
    for stim_label in ["no-stim", "stim"]:
        grand_evoked[stim_label]["cue"] = mne.grand_average(evoked_by_condition[stim_label]["cue"])
        grand_evoked[stim_label]["grating"] = mne.grand_average(evoked_by_condition[stim_label]["grating"])

    fig = plot_compare_evokeds_by_channel(
        {"No stimulation": grand_evoked["no-stim"]["cue"], "Stimulation": grand_evoked["stim"]["cue"]},
        channels=OCCIPITAL_CHANNELS,
        title=f"Grand average ERP across subjects ({', '.join('sub-' + s for s in subjects)})",
    )
    report.add_figure(
        fig,
        title="Grand average ERP across subjects - cue locked",
        caption=f"ERP computed as a grand average across subjects: {', '.join('sub-' + s for s in subjects)}",
        section="ERP",
    )

    fig2 = plot_compare_evokeds_by_channel(
        {"No stimulation": grand_evoked["no-stim"]["grating"], "Stimulation": grand_evoked["stim"]["grating"]},
        channels=OCCIPITAL_CHANNELS,
        title=f"Grand average ERP across subjects ({', '.join('sub-' + s for s in subjects)}) - grating locked",
    )
    report.add_figure(
        fig2,
        title="Grand average ERP across subjects - grating locked",
        caption="Grating-locked ERP computed as a grand average across subjects.",
        section="ERP",
    )

    # Save evokeds
    for stim_label in ["no-stim", "stim"]:
        mne.write_evokeds(
            op.join(GROUP_DERIV_DIR, f"group_{stim_label}_grand-evoked-cue.fif"),
            grand_evoked[stim_label]["cue"],
            overwrite=True,
        )
        mne.write_evokeds(
            op.join(GROUP_DERIV_DIR, f"group_{stim_label}_grand-evoked-grating.fif"),
            grand_evoked[stim_label]["grating"],
            overwrite=True,
        )

    # -----------------------
    # TFR grand averages
    # -----------------------
    tfrs = {"no-stim": {"both": [], "right": [], "left": []}, "stim": {"both": [], "right": [], "left": []}}

    for subject in subjects:
        subj_tfrs = read_subject_tfrs(BIDS_ROOT, subject)
        for stim_label in ["no-stim", "stim"]:
            for cue in ["both", "right", "left"]:
                tfr = subj_tfrs[(stim_label, cue)].copy().pick(OCCIPITAL_CHANNELS)
                tfrs[stim_label][cue].append(tfr)

    grand_tfr = {"no-stim": {}, "stim": {}}
    for stim_label in ["no-stim", "stim"]:
        for cue in ["both", "right", "left"]:
            grand_tfr[stim_label][cue] = mne.grand_average(tfrs[stim_label][cue])
            grand_tfr[stim_label][cue].save(
                op.join(GROUP_DERIV_DIR, f"group_{stim_label}_{cue}_grand-tfr.h5"),
                overwrite=True,
            )

    # Same style as single-subject TFR plots
    for stim_label in ["no-stim", "stim"]:
        fig_tfr = grand_tfr[stim_label]["both"].plot_topo(
            tmin=-0.5, tmax=1.5, baseline=BASELINE, mode="percent",
            title=f"Grand average TFR across subjects ({', '.join('sub-' + s for s in subjects)}) - {stim_label}",
            show=False, fig_facecolor="w", font_color="k"
        )
        report.add_figure(
            fig_tfr,
            title=f"Grand average TFR across subjects - {stim_label}",
            caption=f"Grand average TFR for {stim_label} across subjects.",
            section="TFR",
        )

    diff = grand_tfr["no-stim"]["both"].copy()
    diff.data = grand_tfr["no-stim"]["both"].data - grand_tfr["stim"]["both"].data
    fig_diff = diff.plot_topo(
        tmin=-0.5, tmax=1.5, baseline=None, mode=None,
        title=f"Grand average TFR across subjects ({', '.join('sub-' + s for s in subjects)}) - difference",
        show=False, fig_facecolor="w", font_color="k"
    )
    report.add_figure(
        fig_diff,
        title="Grand average TFR across subjects - difference",
        caption="Difference = no-stim - stim from subject-level grand averages.",
        section="TFR",
    )

    ratio = grand_tfr["no-stim"]["both"].copy()
    ratio.data = (grand_tfr["no-stim"]["both"].data - grand_tfr["stim"]["both"].data) / (
        grand_tfr["no-stim"]["both"].data + grand_tfr["stim"]["both"].data + np.finfo(float).eps
    )
    fig_ratio = ratio.plot_topo(
        tmin=-0.5, tmax=1.5, baseline=None, mode=None,
        title=f"Grand average TFR across subjects ({', '.join('sub-' + s for s in subjects)}) - ratio",
        show=False, fig_facecolor="w", font_color="k"
    )
    report.add_figure(
        fig_ratio,
        title="Grand average TFR across subjects - ratio",
        caption="Ratio = (no-stim - stim) / (no-stim + stim) from subject-level grand averages.",
        section="TFR",
    )

    add_analysis_notes_section(report, prompt_text="Analysis notes")
    report.save(op.join(GROUP_REPORT_DIR, f"{REPORT_NAME}.hdf5"), overwrite=True)
    report.save(op.join(GROUP_REPORT_DIR, f"{REPORT_NAME}.html"), overwrite=True, open_browser=True)


if __name__ == "__main__":
    subjects = load_subject_list(SUBJECT_LIST_PATH) if Path(SUBJECT_LIST_PATH).exists() else DEFAULT_SUBJECTS
    build_grand_average_report(subjects)