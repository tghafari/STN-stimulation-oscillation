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
import argparse

import os.path as op
from pathlib import Path
import matplotlib.pyplot as plt

import mne
import numpy as np

from group_utils import (
    add_analysis_notes_section,
    add_subject_summary,
    ensure_dir,
    make_report,
    read_subject_epochs,
    read_subject_evokeds,
)

# -----------------------
# Config
# -----------------------
PROJECT_ROOT = "/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/STN-in-PD"
BIDS_ROOT = op.join(PROJECT_ROOT, "data", "BIDS")

GROUP_REPORT_DIR = op.join(PROJECT_ROOT, "derivatives", "reports", "group", "grand_average")
GROUP_DERIV_DIR = op.join(BIDS_ROOT, "derivatives", "group", "grand_average")

REPORT_TITLE = "Grand average across subjects"

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

def plot_single_roi_tfr(
    tfr,
    title,
    baseline=None,
    mode=None,
    vlim=None,
):
    """Plot one posterior-ROI TFR."""

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 5),
        constrained_layout=True,
    )

    plot_kwargs = dict(
        picks=["posterior_ROI"],
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

    tfr.plot(
        **plot_kwargs
    )

    ax.set_title(title)

    return fig

def parse_args():
    """Get the subjects to include in the group analysis."""

    parser = argparse.ArgumentParser(
        description=(
            "Run G02 grand-average group analysis "
            "for selected subjects."
        )
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        required=True,
        help=(
            "Subject numbers to include, e.g. "
            "--subjects 115 116 118 119"
        ),
    )

    return parser.parse_args()

def build_grand_average_report(subjects):
    ensure_dir(GROUP_REPORT_DIR)
    ensure_dir(GROUP_DERIV_DIR)

    # Create a unique report name for this exact subject set.
    REPORT_NAME = "group_grand_average_"+ "_".join(subjects)

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
    channel_summary = []

    for ch in OCCIPITAL_CHANNELS:

        included = subjects_by_channel[ch]

        excluded = [
            subject
            for subject in subjects
            if subject not in included
        ]

        text = (
            f"{ch}: included "
            f"{', '.join('sub-' + s for s in included)}"
        )

        if excluded:
            text += (
                f"\nExcluded because channel was unavailable: "
                f"{', '.join('sub-' + s for s in excluded)}"
            )

        channel_summary.append(text)

    report.add_text(
        "Subjects contributing to each posterior channel",
        "\n\n".join(channel_summary),
        "Group analysis",
    )

    report.add_text(
        "Analysis details",
        (
            "ERP\n"
            "• Cue-locked ERP: cue onset = 0 s.\n"
            "• Epoch window: -0.5 to 1.6 s.\n"
            "• ERP is low-pass filtered at 30 Hz.\n"
            "• Cue ERP baseline: -0.1 to 0 s.\n"
            "• Grating-locked ERP: grating onset = 0 s.\n"
            "• Each subject is averaged first, then subject averages "
            "are grand-averaged, so each participant contributes equally.\n\n"

            "TFR\n"
            "• TFRs are calculated from cue-locked epochs.\n"
            "• Left- and right-attention trials are combined ('both').\n"
            "• Epoch window: -0.5 to 1.6 s.\n"
            "• Frequency range: 2-31.5 Hz in 0.5-Hz steps.\n"
            "• Multitaper method.\n"
            "• n_cycles = frequency / 2.\n"
            "• Time-bandwidth = 2.\n"
            "• Decimation = 2.\n"
            "• Stim/no-stim TFRs: baseline -0.3 to -0.1 s, "
            "percent change.\n"
            "• Difference: stimulation minus no-stimulation, "
            "using the original non-baseline-corrected TFRs.\n"
            "• Ratio: (stimulation - no-stimulation) / "
            "(stimulation + no-stimulation), using the original "
            "non-baseline-corrected TFRs.\n"

            "Posterior channels\n"
            "• PO3, PO4 and POz are analysed separately.\n"
            "• A channel contributes only for subjects in whom it "
            "was available after subject-level cleaning.\n\n"

            "Combined posterior ROI\n"
            "• Within each subject, the available posterior channels "
            "are averaged within each epoch.\n"
            "• A subject-level ROI TFR is then calculated.\n"
            "• Subject-level ROI TFRs are grand-averaged, so every "
            "subject contributes equally."
        ),
        "Analysis details",
    )

    # ----------------------------------------------------------
    # ERP grand averages
    # ----------------------------------------------------------

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

        stim_tfr = grand_tfr_by_channel[
            "stim"
        ][ch].copy()

        no_stim_tfr = grand_tfr_by_channel[
            "no-stim"
        ][ch].copy()

        # Difference uses non-baseline corrected TFRs.
        diff = no_stim_tfr.copy()

        diff.data = (
            stim_tfr.data
            - no_stim_tfr.data
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
            stim.data
            - no_stim.data
        ) / (
            stim.data
            + no_stim.data
            + np.finfo(float).eps
        )

        grand_tfr_ratio[ch] = ratio

    subject_counts = {
        ch: len(subjects_by_channel[ch])
        for ch in OCCIPITAL_CHANNELS
    }

    # ==========================================================
    # Channel-specific TFR plots
    # ==========================================================

    # ----------------------------------------------------------
    # No stimulation
    # ----------------------------------------------------------

    fig_tfr_no = plot_three_channel_tfrs(
        grand_tfr_by_channel["no-stim"],
        OCCIPITAL_CHANNELS,
        subject_counts,
        (
            "Grand-average cue-locked TFR - "
            "no stimulation - combined attention"
        ),
        baseline=BASELINE,
        mode="percent",
        vlim=(-0.75, 0.75),
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_grand_average_TFR_no_stim.png",
    )

    fig_tfr_no.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_tfr_no,
        fname,
        "Grand-average cue-locked TFR - no stimulation",
        (
            "Each subject is averaged first and the subject-level TFRs "
            "are then grand-averaged. Cue onset = 0 s. "
            "Left- and right-attention trials are combined ('both'). "
            "Baseline: -0.3 to -0.1 s, percent change. "
            "Frequency range: 2-31.5 Hz."
        ),
        "TFR",
    )


    # ----------------------------------------------------------
    # Stimulation
    # ----------------------------------------------------------

    fig_tfr_stim = plot_three_channel_tfrs(
        grand_tfr_by_channel["stim"],
        OCCIPITAL_CHANNELS,
        subject_counts,
        (
            "Grand-average cue-locked TFR - "
            "stimulation - combined attention"
        ),
        baseline=BASELINE,
        mode="percent",
        vlim=(-0.75, 0.75),
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_grand_average_TFR_stim.png",
    )

    fig_tfr_stim.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_tfr_stim,
        fname,
        "Grand-average cue-locked TFR - stimulation",
        (
            "Each subject is averaged first and the subject-level TFRs "
            "are then grand-averaged. Cue onset = 0 s. "
            "Left- and right-attention trials are combined ('both'). "
            "Baseline: -0.3 to -0.1 s, percent change. "
            "Frequency range: 2-31.5 Hz."
        ),
        "TFR",
    )


    # ----------------------------------------------------------
    # Difference
    # ----------------------------------------------------------

    fig_tfr_diff = plot_three_channel_tfrs(
        grand_tfr_diff,
        OCCIPITAL_CHANNELS,
        subject_counts,
        (
            "Grand-average cue-locked TFR difference - "
            "stimulation minus no stimulation"
        ),
        baseline=None,
        mode=None,
        vlim=None,
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_grand_average_TFR_difference.png",
    )

    fig_tfr_diff.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_tfr_diff,
        fname,
        "Grand-average cue-locked TFR difference",
        (
            "Each subject is averaged first and then grand-averaged. "
            "Cue onset = 0 s. Left- and right-attention trials are "
            "combined ('both'). "
            "Difference = stimulation minus no-stimulation. "
            "No baseline correction was applied to the difference."
        ),
        "TFR",
    )


    # ----------------------------------------------------------
    # Ratio
    # ----------------------------------------------------------

    fig_tfr_ratio = plot_three_channel_tfrs(
        grand_tfr_ratio,
        OCCIPITAL_CHANNELS,
        subject_counts,
        (
            "Grand-average cue-locked TFR ratio - "
            "stimulation versus no stimulation"
        ),
        baseline=None,
        mode=None,
        vlim=None,
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_grand_average_TFR_ratio.png",
    )

    fig_tfr_ratio.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_tfr_ratio,
        fname,
        "Grand-average cue-locked TFR ratio",
        (
            "Cue onset = 0 s. Left- and right-attention trials are "
            "combined ('both'). "
            "Ratio = (stimulation - no-stimulation) / "
            "(stimulation + no-stimulation)."
            "No baseline correction was applied to the ratio."
        ),
        "TFR",
    )

    # ==========================================================
    # Combined posterior ROI grand-average TFR
    # ==========================================================

    roi_tfr_by_condition = {
        "no-stim": [],
        "stim": [],
    }

    roi_subjects = []
    roi_channel_summary = []


    for subject in subjects:

        available_channels = [
            ch
            for ch in OCCIPITAL_CHANNELS
            if (
                ch in subject_epochs[subject]["no-stim"].ch_names
                and
                ch in subject_epochs[subject]["stim"].ch_names
            )
        ]

        if not available_channels:
            continue

        roi_channel_summary.append(
            f"sub-{subject}: {', '.join(available_channels)}"
        )

        roi_subjects.append(subject)

        for stim_label in [
            "no-stim",
            "stim",
        ]:

            epochs = (
                subject_epochs[subject][stim_label]
                .copy()
            )

            available = [
                ch
                for ch in OCCIPITAL_CHANNELS
                if ch in epochs.ch_names
            ]

            epochs.pick(
                available
            )

            # Average the available posterior channels within each epoch.
            roi_data = epochs.get_data().mean(
                axis=1,
                keepdims=True,
            )

            roi_info = mne.create_info(
                ["posterior_ROI"],
                sfreq=epochs.info["sfreq"],
                ch_types=["eeg"],
            )

            roi_epochs = mne.EpochsArray(
                roi_data,
                roi_info,
                events=epochs.events.copy(),
                event_id=epochs.event_id.copy(),
                tmin=epochs.tmin,
            )

            # Calculate the subject-level ROI TFR.
            subject_roi_tfr = (
                roi_epochs.compute_tfr(
                    **TFR_PARAMS
                )
            )

            subject_roi_tfr.comment = (
                f"sub-{subject}, {stim_label}, "
                "posterior ROI, cue-locked, combined attention"
            )

            roi_tfr_by_condition[
                stim_label
            ].append(
                subject_roi_tfr
            )


    report.add_text(
        "Grand-average posterior ROI definition",
        (
            "The posterior ROI is calculated separately for each subject "
            "before the subject-level ROI TFRs are grand-averaged.\n\n"
            "Cue onset = 0 s. Left- and right-attention trials are "
            "combined ('both').\n\n"
            "Posterior channels available for each subject:\n"
            + "\n".join(roi_channel_summary)
            + "\n\n"
            "Each subject contributes equally to the final posterior ROI "
            "grand average."
        ),
        "Analysis details",
    )


    # ----------------------------------------------------------
    # Grand-average ROI TFR
    # ----------------------------------------------------------

    roi_grand_tfr = {}

    for stim_label in [
        "no-stim",
        "stim",
    ]:

        roi_grand_tfr[
            stim_label
        ] = mne.grand_average(
            roi_tfr_by_condition[
                stim_label
            ]
        )

        fname = op.join(
            GROUP_DERIV_DIR,
            f"group_{stim_label}_posterior-ROI_grand-tfr.h5",
        )

        roi_grand_tfr[
            stim_label
        ].save(
            fname,
            overwrite=True,
        )


    # ----------------------------------------------------------
    # ROI difference
    # ----------------------------------------------------------

    roi_stim_tfr = roi_grand_tfr[
        "stim"
    ].copy()

    roi_no_stim_tfr = roi_grand_tfr[
        "no-stim"
    ].copy()

    roi_grand_diff = roi_no_stim_tfr.copy()

    roi_grand_diff.data = (
        roi_stim_tfr.data
        - roi_no_stim_tfr.data
    )


    # ----------------------------------------------------------
    # ROI ratio
    # ----------------------------------------------------------

    roi_grand_ratio = roi_grand_tfr[
        "no-stim"
    ].copy()

    roi_grand_ratio.data = (
        roi_grand_tfr["stim"].data
        - roi_grand_tfr["no-stim"].data
    ) / (
        roi_grand_tfr["stim"].data
        + roi_grand_tfr["no-stim"].data
        + np.finfo(float).eps
    )

    # ==========================================================
    # Posterior ROI TFR plots
    # ==========================================================

    fig_roi_no = plot_single_roi_tfr(
        roi_grand_tfr["no-stim"],
        (
            "Grand-average cue-locked posterior ROI TFR - "
            "no stimulation"
        ),
        baseline=BASELINE,
        mode="percent",
        vlim=(-0.75, 0.75),
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_grand_average_TFR_posterior_ROI_no_stim.png",
    )

    fig_roi_no.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_roi_no,
        fname,
        "Grand-average posterior ROI TFR - no stimulation",
        (
            "Each subject contributes equally. Cue onset = 0 s. "
            "Left- and right-attention trials are combined ('both'). "
            "The posterior ROI is calculated within each subject from "
            "the available posterior channels before grand averaging. "
            "Baseline: -0.3 to -0.1 s, percent change."
        ),
        "TFR",
    )


    fig_roi_stim = plot_single_roi_tfr(
        roi_grand_tfr["stim"],
        (
            "Grand-average cue-locked posterior ROI TFR - "
            "stimulation"
        ),
        baseline=BASELINE,
        mode="percent",
        vlim=(-0.75, 0.75),
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_grand_average_TFR_posterior_ROI_stim.png",
    )

    fig_roi_stim.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_roi_stim,
        fname,
        "Grand-average posterior ROI TFR - stimulation",
        (
            "Each subject contributes equally. Cue onset = 0 s. "
            "Left- and right-attention trials are combined ('both'). "
            "The posterior ROI is calculated within each subject from "
            "the available posterior channels before grand averaging. "
            "Baseline: -0.3 to -0.1 s, percent change."
        ),
        "TFR",
    )


    fig_roi_diff = plot_single_roi_tfr(
        roi_grand_diff,
        (
            "Grand-average cue-locked posterior ROI TFR - "
            "difference: stim - no-stim"
        ),
        baseline=None,
        mode=None,
        vlim=None,
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_grand_average_TFR_posterior_ROI_difference.png",
    )

    fig_roi_diff.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_roi_diff,
        fname,
        "Grand-average posterior ROI TFR - difference",
        (
            "Cue onset = 0 s. Left- and right-attention trials are "
            "combined ('both'). Difference = baseline-corrected "
            "stimulation minus no-stimulation. "
            "Baseline: -0.3 to -0.1 s, percent change."
        ),
        "TFR",
    )


    fig_roi_ratio = plot_single_roi_tfr(
        roi_grand_ratio,
        (
            "Grand-average cue-locked posterior ROI TFR - "
            "ratio"
        ),
        baseline=None,
        mode=None,
        vlim=None,
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_grand_average_TFR_posterior_ROI_ratio.png",
    )

    fig_roi_ratio.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_roi_ratio,
        fname,
        "Grand-average posterior ROI TFR - ratio",
        (
            "Cue onset = 0 s. Left- and right-attention trials are "
            "combined ('both'). Ratio = (stimulation - no-stimulation) / "
            "(stimulation + no-stimulation)."
        ),
        "TFR",
    )

    add_analysis_notes_section(
        report,
        prompt_text="Analysis notes",
    )

    print(
        f"\nGroup report completed:\n"
        f"{report.pdf_fname}"
    )

    print(
        f"Report manifest:\n"
        f"{report.manifest_fname}"
    )


if __name__ == "__main__":

    args = parse_args()

    subjects = [
        str(s).removeprefix("sub-")
        for s in args.subjects
    ]

    print(
        "\nSubjects included in G02:"
        f"\n  {', '.join('sub-' + s for s in subjects)}\n"
    )

    build_grand_average_report(
        subjects
    )
