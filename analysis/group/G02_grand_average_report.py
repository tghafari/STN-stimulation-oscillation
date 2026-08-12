#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================
G02. Grand-average analysis across subjects
==============================================================

This script performs descriptive group-level ERP and TFR analyses in
which each participant is averaged separately before the group average
is calculated. Therefore, unlike G01, every participant contributes
equally regardless of the number of retained trials.

Posterior-channel handling
--------------------------
Analyses use PO3, PO4 and POz. For each channel, only subjects in whom
that channel remains after subject-level cleaning contribute to its
grand average. Missing channels are not interpolated.

A combined posterior ROI is also calculated. For each subject, the
available posterior channels are averaged first, so subjects with a
missing posterior channel can still contribute to the ROI.

Outputs
-------
The report contains:
    - cue-locked ERP (cue onset = 0 s)
    - grating-locked ERP (grating onset = 0 s)
    - cue-locked TFR for combined left/right attention
    - no-stim, stim, difference and ratio TFRs
    - separate PO3, PO4 and POz results
    - combined posterior-ROI results
    - subjects contributing to each channel
    - analysis notes

TFR settings match G01:
    2-31.5 Hz, 0.5-Hz steps, multitaper,
    n_cycles = frequency / 2, time-bandwidth = 2,
    decimation = 2, baseline = -0.3 to -0.1 s (percent).

No PAF, MI or statistical testing is performed here.

written by Tara Ghafari
tara.ghafari@gmail.com
==============================================================
"""


from __future__ import annotations

import os.path as op
from pathlib import Path
import matplotlib.pyplot as plt

import mne
import numpy as np

from group_utils import (
    add_analysis_notes_section,
    add_subject_summary,
    ensure_dir,
    load_subject_list,
    make_report,
    read_subject_epochs,
    save_subject_list,
    read_subject_evokeds,
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

def plot_three_channel_tfrs(
    tfr_dict,
    channels,
    subject_counts,
    title,
    baseline=None,
    mode=None,
    vlim=None,
):
    """Plot PO3, PO4 and POz TFRs in one row."""

    fig, axes = plt.subplots(
        1,
        len(channels),
        figsize=(18, 5),
        constrained_layout=True,
    )

    if len(channels) == 1:
        axes = [axes]

    for ax, ch in zip(
        axes,
        channels,
    ):

        plot_kwargs = dict(
            picks=[ch],
            tmin=-0.5,
            tmax=1.5,
            fmin=2,
            fmax=31,
            baseline=baseline,
            mode=mode,
            axes=ax,
            show=False,
            colorbar=True,
        )

        if vlim is not None:
            plot_kwargs["vlim"] = vlim

        tfr_dict[ch].plot(
            **plot_kwargs
        )

        ax.set_title(
            f"{ch} "
            f"(n={subject_counts[ch]} subjects)"
        )

    fig.suptitle(
        title,
        fontsize=14,
    )

    return fig

def build_grand_average_report(subjects):
    ensure_dir(GROUP_REPORT_DIR)
    ensure_dir(GROUP_DERIV_DIR)

    report = make_report(GROUP_REPORT_DIR, REPORT_NAME)
    report.add_text(
        "Group analysis summary",
        (
            f"Report: {REPORT_TITLE}\n"
            f"Subjects analysed: "
            f"{', '.join('sub-' + s for s in subjects)}\n"
            f"Number of subjects: {len(subjects)}\n"
            "Each subject contributes equally."
        ),
        "Group analysis",
    )
    add_subject_summary(report, subjects)

    save_subject_list(SUBJECT_LIST_PATH, subjects)

    # ----------------------------------------------------------
    # Load cleaned subject-level epochs
    # ----------------------------------------------------------

    subject_epochs = {}

    for subject in subjects:
        subject_epochs[subject] = read_subject_epochs(
            BIDS_ROOT,
            subject,
        )

    # ----------------------------------------------------------
    # Determine which subjects contribute to each posterior channel
    # ----------------------------------------------------------

    subjects_by_channel = {}

    for ch in OCCIPITAL_CHANNELS:

        subjects_by_channel[ch] = [
            subject
            for subject in subjects
            if (
                ch in subject_epochs[subject]["no-stim"].ch_names
                and
                ch in subject_epochs[subject]["stim"].ch_names
            )
        ]


    # ----------------------------------------------------------
    # ERP grand averages
    # ----------------------------------------------------------

    evoked_by_condition = {
        "no-stim": {"cue": [], "grating": []},
        "stim": {"cue": [], "grating": []},
    }
    # ----------------------------------------------------------
    # Prepare subject-level ERP data
    # ----------------------------------------------------------

    cue_evoked_by_channel = {
        "no-stim": {ch: [] for ch in OCCIPITAL_CHANNELS},
        "stim": {ch: [] for ch in OCCIPITAL_CHANNELS},
    }

    grating_evoked_by_channel = {
        "no-stim": {ch: [] for ch in OCCIPITAL_CHANNELS},
        "stim": {ch: [] for ch in OCCIPITAL_CHANNELS},
    }


    for subject in subjects:

        # ------------------------------------------------------
        # Cue-locked ERP from cleaned epochs
        # ------------------------------------------------------

        for stim_label in ["no-stim", "stim"]:

            epochs = (
                subject_epochs[subject][stim_label]
                .copy()
            )

            for ch in OCCIPITAL_CHANNELS:

                if subject not in subjects_by_channel[ch]:
                    continue

                subject_channel_epochs = (
                    epochs
                    .copy()
                    .pick([ch])
                )

                cue_evoked = (
                    subject_channel_epochs
                    .average()
                    .filter(
                        l_freq=None,
                        h_freq=30,
                    )
                    .crop(
                        tmin=-0.1,
                        tmax=1.0,
                    )
                )

                cue_evoked.apply_baseline(
                    baseline=(-0.1, 0),
                )

                cue_evoked.comment = (
                    f"sub-{subject}, {stim_label}, "
                    f"{ch}, cue-locked"
                )

                cue_evoked_by_channel[
                    stim_label
                ][ch].append(
                    cue_evoked
                )

        # ------------------------------------------------------
        # Grating-locked ERP from existing subject-level evokeds
        # ------------------------------------------------------

        subject_evokeds = read_subject_evokeds(
            BIDS_ROOT,
            subject,
        )

        for stim_label in ["no-stim", "stim"]:

            grating_evoked = subject_evokeds[
                (stim_label, "grating")
            ].copy()

            for ch in OCCIPITAL_CHANNELS:

                if (
                    subject not in subjects_by_channel[ch]
                    or ch not in grating_evoked.ch_names
                ):
                    continue

                subject_grating = (
                    grating_evoked
                    .copy()
                    .pick([ch])
                )

                subject_grating.comment = (
                    f"sub-{subject}, {stim_label}, "
                    f"{ch}, grating-locked"
                )

                grating_evoked_by_channel[
                    stim_label
                ][ch].append(
                    subject_grating
                )
    # ----------------------------------------------------------
    # Grand-average ERPs
    # ----------------------------------------------------------

    grand_cue_evoked = {
        "no-stim": {},
        "stim": {},
    }

    grand_grating_evoked = {
        "no-stim": {},
        "stim": {},
    }

    for stim_label in ["no-stim", "stim"]:

        for ch in OCCIPITAL_CHANNELS:

            if cue_evoked_by_channel[
                stim_label
            ][ch]:

                grand_cue_evoked[
                    stim_label
                ][ch] = mne.grand_average(
                    cue_evoked_by_channel[
                        stim_label
                    ][ch]
                )

            if grating_evoked_by_channel[
                stim_label
            ][ch]:

                grand_grating_evoked[
                    stim_label
                ][ch] = mne.grand_average(
                    grating_evoked_by_channel[
                        stim_label
                    ][ch]
                )

    # ----------------------------------------------------------
    # Cue-locked grand-average ERP
    # ----------------------------------------------------------

    fig_cue, axes = plt.subplots(
        len(OCCIPITAL_CHANNELS),
        1,
        figsize=(10, 4 * len(OCCIPITAL_CHANNELS)),
        constrained_layout=True,
    )

    if len(OCCIPITAL_CHANNELS) == 1:
        axes = [axes]

    for ax, ch in zip(
        axes,
        OCCIPITAL_CHANNELS,
    ):

        if (
            ch not in grand_cue_evoked["no-stim"]
            or ch not in grand_cue_evoked["stim"]
        ):
            ax.set_title(
                f"{ch}: insufficient data"
            )
            continue

        mne.viz.plot_compare_evokeds(
            {
                "No stimulation": grand_cue_evoked[
                    "no-stim"
                ][ch],
                "Stimulation": grand_cue_evoked[
                    "stim"
                ][ch],
            },
            picks=[ch],
            axes=ax,
            show=False,
            ci=False,
            truncate_xaxis=False,
            truncate_yaxis=False,
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_title(
            f"{ch} "
            f"(n={len(cue_evoked_by_channel['no-stim'][ch])} subjects)\n"
            "Cue onset = 0 s"
        )

    fig_cue.suptitle(
        "Grand-average cue-locked ERP across subjects "
        "(cue onset = 0 s)",
        fontsize=14,
    )

    cue_fig_fname = op.join(
        GROUP_REPORT_DIR,
        "group_grand_average_ERP_cue.png",
    )

    fig_cue.savefig(
        cue_fig_fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_cue,
        cue_fig_fname,
        "Grand-average cue-locked ERP",
        (
            "Each subject is averaged first, then subject averages are "
            "grand-averaged. Cue onset = 0 s. "
            "Only subjects with the relevant posterior channel contribute."
        ),
        "ERP",
    )

    # ----------------------------------------------------------
    # Grating-locked grand-average ERP
    # ----------------------------------------------------------

    fig_grating, axes = plt.subplots(
        len(OCCIPITAL_CHANNELS),
        1,
        figsize=(10, 4 * len(OCCIPITAL_CHANNELS)),
        constrained_layout=True,
    )

    if len(OCCIPITAL_CHANNELS) == 1:
        axes = [axes]

    for ax, ch in zip(
        axes,
        OCCIPITAL_CHANNELS,
    ):

        if (
            ch not in grand_grating_evoked["no-stim"]
            or ch not in grand_grating_evoked["stim"]
        ):
            ax.set_title(
                f"{ch}: insufficient data"
            )
            continue

        mne.viz.plot_compare_evokeds(
            {
                "No stimulation": grand_grating_evoked[
                    "no-stim"
                ][ch],
                "Stimulation": grand_grating_evoked[
                    "stim"
                ][ch],
            },
            picks=[ch],
            axes=ax,
            show=False,
            ci=False,
            truncate_xaxis=False,
            truncate_yaxis=False,
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_title(
            f"{ch} "
            f"(n={len(grating_evoked_by_channel['no-stim'][ch])} subjects)\n"
            "Grating onset = 0 s"
        )

    fig_grating.suptitle(
        "Grand-average grating-locked ERP across subjects "
        "(grating onset = 0 s)",
        fontsize=14,
    )

    grating_fig_fname = op.join(
        GROUP_REPORT_DIR,
        "group_grand_average_ERP_grating.png",
    )

    fig_grating.savefig(
        grating_fig_fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_grating,
        grating_fig_fname,
        "Grand-average grating-locked ERP",
        (
            "Each subject is averaged first, then subject averages are "
            "grand-averaged. Grating onset = 0 s."
        ),
        "ERP",
    )

    # ==========================================================
    # TFR grand averages
    # ==========================================================
    #
    # TFRs are calculated separately for each subject from the same
    # cleaned cue-locked epochs used for the group ERP.
    #
    # Left- and right-attention trials are combined because the
    # cue-locked epochs contain both cue directions.
    #
    # Each subject contributes equally:
    #
    #     subject epochs
    #            ↓
    #       subject TFR
    #            ↓
    #     grand average
    #
    # TFR settings match G01:
    #     frequency range: 2-31.5 Hz
    #     frequency step: 0.5 Hz
    #     multitaper
    #     n_cycles = frequency / 2
    #     time-bandwidth = 2
    #     decimation = 2
    #     baseline = -0.3 to -0.1 s
    #
    # Missing channels are handled separately for each channel.
    # ==========================================================

    tfr_by_channel = {
        "no-stim": {
            ch: []
            for ch in OCCIPITAL_CHANNELS
        },
        "stim": {
            ch: []
            for ch in OCCIPITAL_CHANNELS
        },
    }


    # ----------------------------------------------------------
    # Calculate one TFR per subject and posterior channel
    # ----------------------------------------------------------

    for subject in subjects:

        for stim_label in [
            "no-stim",
            "stim",
        ]:

            epochs = (
                subject_epochs[subject][stim_label]
                .copy()
            )

            for ch in OCCIPITAL_CHANNELS:

                # Skip this subject if the channel was unavailable.
                if subject not in subjects_by_channel[ch]:
                    continue

                channel_epochs = (
                    epochs
                    .copy()
                    .pick([ch])
                )

                print(
                    f"Computing subject TFR: "
                    f"sub-{subject}, {stim_label}, {ch}"
                )

                subject_tfr = (
                    channel_epochs
                    .compute_tfr(
                        **TFR_PARAMS
                    )
                )

                # Keep the subject identity in the comment.
                subject_tfr.comment = (
                    f"sub-{subject}, {stim_label}, "
                    f"{ch}, cue-locked, combined attention"
                )

                tfr_by_channel[
                    stim_label
                ][ch].append(
                    subject_tfr
                )


    # ----------------------------------------------------------
    # Grand-average the subject-level TFRs
    # ----------------------------------------------------------

    grand_tfr_by_channel = {
        "no-stim": {},
        "stim": {},
    }

    for stim_label in [
        "no-stim",
        "stim",
    ]:

        for ch in OCCIPITAL_CHANNELS:

            subject_tfrs = tfr_by_channel[
                stim_label
            ][ch]

            if not subject_tfrs:
                raise RuntimeError(
                    f"No subject TFRs available for "
                    f"{stim_label}, {ch}."
                )

            grand_tfr_by_channel[
                stim_label
            ][ch] = mne.grand_average(
                subject_tfrs
            )

            fname = op.join(
                GROUP_DERIV_DIR,
                f"group_{stim_label}_{ch}_grand-tfr.h5",
            )

            grand_tfr_by_channel[
                stim_label
            ][ch].save(
                fname,
                overwrite=True,
            )


    # ----------------------------------------------------------
    # Channel-specific difference and ratio
    # ----------------------------------------------------------

    grand_tfr_diff = {}
    grand_tfr_ratio = {}

    for ch in OCCIPITAL_CHANNELS:

        stim_bc = grand_tfr_by_channel[
            "stim"
        ][ch].copy()

        no_stim_bc = grand_tfr_by_channel[
            "no-stim"
        ][ch].copy()

        # Baseline correction for the difference.
        stim_bc.apply_baseline(
            baseline=BASELINE,
            mode="percent",
        )

        no_stim_bc.apply_baseline(
            baseline=BASELINE,
            mode="percent",
        )

        diff = no_stim_bc.copy()

        diff.data = (
            no_stim_bc.data
            - stim_bc.data
        )

        grand_tfr_diff[ch] = diff

        # Ratio uses the original, non-baseline-corrected TFRs.
        no_stim = grand_tfr_by_channel[
            "no-stim"
        ][ch]

        stim = grand_tfr_by_channel[
            "stim"
        ][ch]

        ratio = no_stim.copy()

        ratio.data = (
            no_stim.data
            - stim.data
        ) / (
            no_stim.data
            + stim.data
            + np.finfo(float).eps
        )

        grand_tfr_ratio[ch] = ratio



    add_analysis_notes_section(report, prompt_text="Analysis notes")
    report.save(op.join(GROUP_REPORT_DIR, f"{REPORT_NAME}.hdf5"), overwrite=True)
    report.save(op.join(GROUP_REPORT_DIR, f"{REPORT_NAME}.html"), overwrite=True, open_browser=True)


if __name__ == "__main__":
    subjects = load_subject_list(SUBJECT_LIST_PATH) if Path(SUBJECT_LIST_PATH).exists() else DEFAULT_SUBJECTS
    build_grand_average_report(subjects)