#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os.path as op
from pathlib import Path

import mne
import numpy as np
from mne_bids import BIDSPath

from group_utils import (
    add_analysis_notes_section,
    add_subject_summary,
    ensure_dir,
    load_subject_list,
    make_report,
    plot_compare_evokeds_by_channel,
    plot_tfr_channel_grid,
    read_interpolation_summary,
    read_subject_epochs,
    save_subject_list,
    subset_present_channels,
)

# -----------------------
# Config
# -----------------------
PROJECT_ROOT = "/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/STN-in-PD"
BIDS_ROOT = op.join(PROJECT_ROOT, "data", "BIDS")

GROUP_REPORT_DIR = op.join(PROJECT_ROOT, "derivatives", "reports", "group", "concatenated_epochs")
GROUP_DERIV_DIR = op.join(BIDS_ROOT, "derivatives", "group", "concatenated_epochs")
SUBJECT_LIST_PATH = op.join(PROJECT_ROOT, "derivatives", "reports", "group", "subjects_for_group_analysis.json")

REPORT_NAME = "group_concatenated_epochs"
REPORT_TITLE = "Concatenated epochs across subjects"

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

# -----------------------
# Helpers
# -----------------------
def build_concat_epoch_report(subjects):
    ensure_dir(GROUP_REPORT_DIR)
    ensure_dir(GROUP_DERIV_DIR)

    report = make_report(GROUP_REPORT_DIR, REPORT_NAME)
    report.add_text(
        "Group analysis summary",
        f"Report: {REPORT_TITLE}\nSubjects analysed: {', '.join('sub-' + s for s in subjects)}\nNumber of subjects: {len(subjects)}",
        "Group analysis",
    )
    add_subject_summary(report, subjects)

    # Document interpolated posterior channels
    analysis_data_lines = []

    for subject in subjects:

        interpolated = read_interpolation_summary(
            BIDS_ROOT,
            subject,
        )

        if interpolated:
            analysis_data_lines.append(
                f"sub-{subject}: interpolated "
                f"{', '.join(interpolated)}"
            )
        else:
            analysis_data_lines.append(
                f"sub-{subject}: no interpolation"
            )

    report.add_text(
        "Group-analysis channel handling",
        "\n".join(analysis_data_lines),
        "Group analysis",
    )
    # Save/update the list for future reruns
    save_subject_list(SUBJECT_LIST_PATH, subjects)

    group_epochs = {"no-stim": [], "stim": []}

    for subject in subjects:
        subj_epochs = read_subject_epochs(BIDS_ROOT, subject)
        for stim_label in ["no-stim", "stim"]:
            epochs = subj_epochs[stim_label].copy()
            present = subset_present_channels(epochs.ch_names, OCCIPITAL_CHANNELS)
            if not present:
                raise RuntimeError(f"No occipital channels present for sub-{subject} {stim_label}")
            epochs.pick(present)
            group_epochs[stim_label].append(epochs)

    concat_epochs = {}
    for stim_label in ["no-stim", "stim"]:
        concat_epochs[stim_label] = mne.concatenate_epochs(group_epochs[stim_label])
        concat_fname = op.join(GROUP_DERIV_DIR, f"group_{stim_label}_concat-epo.fif")
        concat_epochs[stim_label].save(concat_fname, overwrite=True)

    # ERP from concatenated epochs
    evoked = {}
    for stim_label in ["no-stim", "stim"]:
        evoked[stim_label] = concat_epochs[stim_label].average(method="mean")
        evoked[stim_label] = evoked[stim_label].filter(l_freq=None, h_freq=30)
        evoked[stim_label] = evoked[stim_label].crop(tmin=-0.1, tmax=1.0)
        evoked[stim_label].apply_baseline(baseline=(-0.1, 0))
        evoked_fname = op.join(GROUP_DERIV_DIR, f"group_{stim_label}_concat-evoked.fif")
        mne.write_evokeds(evoked_fname, evoked[stim_label], overwrite=True)

    fig = plot_compare_evokeds_by_channel(
        {"No stimulation": evoked["no-stim"], "Stimulation": evoked["stim"]},
        channels=OCCIPITAL_CHANNELS,
        title=f"Concatenated epochs across subjects ({', '.join('sub-' + s for s in subjects)})",
    )
    report.add_figure(
        fig,
        title="Concatenated epochs across subjects - ERP",
        caption=f"ERP computed from concatenated epochs across subjects: {', '.join('sub-' + s for s in subjects)}",
        section="ERP",
    )

    # TFR from concatenated epochs
    tfr = {}
    for stim_label in ["no-stim", "stim"]:
        epochs = concat_epochs[stim_label].copy().pick(OCCIPITAL_CHANNELS)
        tfr[stim_label] = epochs.compute_tfr(**TFR_PARAMS)
        tfr_fname = op.join(GROUP_DERIV_DIR, f"group_{stim_label}_concat-tfr.h5")
        tfr[stim_label].save(tfr_fname, overwrite=True)

    fig_tfr_no = tfr["no-stim"].plot_topo(
        tmin=-0.5, tmax=1.5, baseline=BASELINE, mode="percent",
        title=f"Concatenated epochs across subjects ({', '.join('sub-' + s for s in subjects)}) - no stimulation",
        show=False, fig_facecolor="w", font_color="k"
    )
    report.add_figure(
        fig_tfr_no,
        title="Concatenated epochs across subjects - no stimulation TFR",
        caption="Grand TFR-style topo plot from concatenated no-stim epochs.",
        section="TFR",
    )

    fig_tfr_stim = tfr["stim"].plot_topo(
        tmin=-0.5, tmax=1.5, baseline=BASELINE, mode="percent",
        title=f"Concatenated epochs across subjects ({', '.join('sub-' + s for s in subjects)}) - stimulation",
        show=False, fig_facecolor="w", font_color="k"
    )
    report.add_figure(
        fig_tfr_stim,
        title="Concatenated epochs across subjects - stimulation TFR",
        caption="Grand TFR-style topo plot from concatenated stimulation epochs.",
        section="TFR",
    )

    diff = tfr["no-stim"].copy()
    diff.data = tfr["no-stim"].data - tfr["stim"].data
    fig_diff = diff.plot_topo(
        tmin=-0.5, tmax=1.5, baseline=None, mode=None,
        title=f"Concatenated epochs across subjects ({', '.join('sub-' + s for s in subjects)}) - difference",
        show=False, fig_facecolor="w", font_color="k"
    )
    report.add_figure(
        fig_diff,
        title="Concatenated epochs across subjects - TFR difference",
        caption="Difference = no-stim - stim from concatenated epochs.",
        section="TFR",
    )

    ratio = tfr["no-stim"].copy()
    ratio.data = (tfr["no-stim"].data - tfr["stim"].data) / (tfr["no-stim"].data + tfr["stim"].data + np.finfo(float).eps)
    fig_ratio = ratio.plot_topo(
        tmin=-0.5, tmax=1.5, baseline=None, mode=None,
        title=f"Concatenated epochs across subjects ({', '.join('sub-' + s for s in subjects)}) - ratio",
        show=False, fig_facecolor="w", font_color="k"
    )
    report.add_figure(
        fig_ratio,
        title="Concatenated epochs across subjects - TFR ratio",
        caption="Ratio = (no-stim - stim) / (no-stim + stim) from concatenated epochs.",
        section="TFR",
    )

    add_analysis_notes_section(report, prompt_text="Analysis notes")
    report.save(op.join(GROUP_REPORT_DIR, f"{REPORT_NAME}.hdf5"), overwrite=True)
    report.save(op.join(GROUP_REPORT_DIR, f"{REPORT_NAME}.html"), overwrite=True, open_browser=True)


if __name__ == "__main__":
    subjects = load_subject_list(SUBJECT_LIST_PATH) if Path(SUBJECT_LIST_PATH).exists() else DEFAULT_SUBJECTS
    build_concat_epoch_report(subjects)