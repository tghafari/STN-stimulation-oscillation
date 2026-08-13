#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==============================================================
G01. Group analysis using concatenated epochs
==============================================================

This script performs descriptive group-level ERP and TFR analyses
using concatenated cleaned epochs from all included participants.

Unlike the grand-average analysis in G02, all retained trials are
concatenated before averaging. Therefore, participants with more
retained trials contribute more strongly to the group estimate.

Posterior-channel handling
--------------------------
The group analysis uses the posterior channels:

    PO3, PO4, POz

A channel is concatenated only across subjects who actually have that
channel available after subject-level cleaning.

For example, if POz is missing in sub-119:

    PO3 -> sub-115, sub-116, sub-118, sub-119
    PO4 -> sub-115, sub-116, sub-118, sub-119
    POz -> sub-115, sub-116, sub-118

No interpolation is performed in this group analysis.

TFR outputs
-----------
For each condition the report contains:

    1. No-stimulation TFR
       PO3 | PO4 | POz

    2. Stimulation TFR
       PO3 | PO4 | POz

    3. Difference TFR
       stim - no-stim

    4. Ratio TFR
       (stim - no-stim) / (stim + no-stim)

A second set of four TFR figures is calculated for a combined
posterior ROI. For each participant, the available posterior channels
are averaged within each epoch before participants are concatenated.
This allows a subject such as sub-119 to contribute using PO3 and PO4
even if POz is unavailable.

The report also records which participants contributed to each
posterior channel and allows analysis notes to be added at the end.

No PAF, MI, or statistical testing is performed here.

written by Tara Ghafari
==============================================================
"""

from __future__ import annotations
import argparse
import matplotlib.pyplot as plt

import os.path as op
from pathlib import Path

import mne
import numpy as np

from group_utils import (
    add_analysis_notes_section,
    add_subject_summary,
    ensure_dir,
    make_report,
    plot_compare_evokeds_by_channel,
    read_interpolation_summary,
    read_subject_epochs,
    save_subject_list,
)


# ==============================================================
# Configuration
# ==============================================================

PROJECT_ROOT = (
    "/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/"
    "BEAR_outage/STN-in-PD"
)

BIDS_ROOT = op.join(
    PROJECT_ROOT,
    "data",
    "BIDS",
)

GROUP_REPORT_DIR = op.join(
    PROJECT_ROOT,
    "derivatives",
    "reports",
    "group",
    "concatenated_epochs",
)

GROUP_DERIV_DIR = op.join(
    BIDS_ROOT,
    "derivatives",
    "group",
    "concatenated_epochs",
)

SUBJECT_LIST_PATH = op.join(
    PROJECT_ROOT,
    "derivatives",
    "reports",
    "group",
    "subjects_for_group_analysis.json",
)

REPORT_NAME = "group_concatenated_epochs"
REPORT_TITLE = "Concatenated epochs across subjects"

OCCIPITAL_CHANNELS = [
    "PO3",
    "PO4",
    "POz",
]

SESSION = "01"
TASK = "SpAtt"
RUN = "01"


# ==============================================================
# TFR parameters
# ==============================================================

BASELINE = (
    -0.3,
    -0.1,
)

FREQS = np.arange(
    2,
    32,
    0.5,
)

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


# ==============================================================
# Local plotting helpers
# ==============================================================

def plot_three_channel_tfrs(
    tfr_dict,
    channels,
    subject_counts,
    title,
    baseline=None,
    mode=None,
    vlim=None,
):
    """
    Plot one TFR for each posterior channel in a single row.

    Parameters
    ----------
    tfr_dict : dict
        Dictionary mapping channel name to AverageTFR.

    channels : list of str
        Posterior channels to plot.

    subject_counts : dict
        Number of subjects contributing to each channel.

    title : str
        Overall figure title.

    baseline : tuple or None
        Baseline interval.

    mode : str or None
        Baseline correction mode.

    vlim : tuple or None
        Color limits for the TFR plots.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """

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
            cmap="RdBu_r",
        )

        # Only give MNE vlim when actual limits were specified.
        # Passing vlim=None causes an error in the current MNE version.
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
    """
    Plot one combined posterior-ROI TFR.

    Parameters
    ----------
    tfr : mne.time_frequency.AverageTFR
        TFR containing the combined posterior ROI.

    title : str
        Figure title.

    baseline : tuple or None
        Baseline interval.

    mode : str or None
        Baseline correction mode.

    vlim : tuple or None
        Color limits.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """

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
        cmap="RdBu_r",
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
            "Run G01 concatenated-epochs group analysis "
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

# ==============================================================
# Main group analysis
# ==============================================================

def build_concat_epoch_report(subjects):
    """
    Run the concatenated-epoch group analysis.

    Parameters
    ----------
    subjects : list of str
        Subjects included in the analysis.

    Returns
    -------
    None
    """

    ensure_dir(
        GROUP_REPORT_DIR
    )

    ensure_dir(
        GROUP_DERIV_DIR
    )

    # ----------------------------------------------------------
    # Create report
    # ----------------------------------------------------------

    report = make_report(
        GROUP_REPORT_DIR,
        REPORT_NAME,
    )

    report.add_text(
        "Group analysis summary",
        (
            f"Report: {REPORT_TITLE}\n"
            f"Subjects analysed: "
            f"{', '.join('sub-' + s for s in subjects)}\n"
            f"Number of subjects: {len(subjects)}"
        ),
        "Group analysis",
    )

    add_subject_summary(
        report,
        subjects,
    )

    # ----------------------------------------------------------
    # Save subject list
    # ----------------------------------------------------------

    save_subject_list(
        SUBJECT_LIST_PATH,
        subjects,
    )

    # ----------------------------------------------------------
    # Load all subject epochs
    # ----------------------------------------------------------

    subject_epochs = {}

    for subject in subjects:

        subject_epochs[subject] = read_subject_epochs(
            BIDS_ROOT,
            subject,
        )

    # ----------------------------------------------------------
    # Determine subjects contributing to each posterior channel
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

        channel_summary.append(
            text
        )

    report.add_text(
        "Subjects contributing to each posterior channel",
        "\n\n".join(channel_summary),
        "Group analysis",
    )

    report.add_text(
        "Analysis details",
        (
            "ERP\n"
            "• ERP is calculated from cue-locked epochs.\n"
            "• Cue onset = 0 s.\n"
            "• Epoch window: -0.5 to 1.6 s.\n"
            "• ERP is low-pass filtered at 30 Hz for plotting.\n"
            "• ERP baseline correction: -0.1 to 0 s.\n\n"

            "TFR\n"
            "• TFRs are calculated from cue-locked epochs.\n"
            "• Left- and right-attention trials are combined ('both').\n"
            "• Epoch window: -0.5 to 1.6 s.\n"
            "• Frequency range: 2–31.5 Hz.\n"
            "• Frequencies are spaced in 0.5 Hz steps.\n"
            "• Multitaper method.\n"
            "• Number of cycles increases with frequency "
            "(n_cycles = frequency / 2).\n"
            "• Time-bandwidth = 2.\n"
            "• Decimation factor = 2.\n"
            "• Baseline for stimulation/no-stimulation TFR plots: "
            "-0.3 to -0.1 s, percent change.\n"
            "• Difference TFR: no-stimulation minus stimulation "
            "no baseline correction.\n"
            "• Ratio TFR: (no-stimulation - stimulation) / "
            "(no-stimulation + stimulation).\n"
            "no baseline correction.\n\n"

            "Posterior-channel TFRs\n"
            "• PO3, PO4 and POz are plotted separately.\n"
            "• A channel is included only for subjects in whom that channel "
            "was available after subject-level cleaning.\n\n"

            "Combined posterior ROI TFR\n"
            "• For each subject, the available posterior channels "
            "(PO3, PO4, POz) are averaged within each epoch.\n"
            "• The resulting subject-level posterior ROI epochs are then "
            "concatenated across subjects.\n"
            "• This allows participants with a missing posterior channel "
            "to contribute using their remaining posterior channels."
        ),
        "Analysis details",
    )

    subject_counts = {
        ch: len(subjects_by_channel[ch])
        for ch in OCCIPITAL_CHANNELS
    }

    # ==========================================================
    # CHANNEL-SPECIFIC CONCATENATED EPOCHS
    # ==========================================================

    concat_epochs_by_channel = {
        "no-stim": {},
        "stim": {},
    }

    for stim_label in [
        "no-stim",
        "stim",
    ]:

        for ch in OCCIPITAL_CHANNELS:

            epochs_list = []

            for subject in subjects_by_channel[ch]:

                epochs = (
                    subject_epochs[subject][stim_label]
                    .copy()
                    .pick([ch])
                )

                epochs_list.append(
                    epochs
                )

            if not epochs_list:

                raise RuntimeError(
                    f"No subjects available for "
                    f"{ch}, {stim_label}."
                )

            concat_epochs_by_channel[stim_label][ch] = (
                mne.concatenate_epochs(
                    epochs_list,
                )
            )

            concat_fname = op.join(
                GROUP_DERIV_DIR,
                f"group_{stim_label}_{ch}_concat-epo.fif",
            )

            concat_epochs_by_channel[
                stim_label
            ][ch].save(
                concat_fname,
                overwrite=True,
            )

            print(
                f"{stim_label} {ch}: "
                f"{len(epochs_list)} subjects, "
                f"{len(concat_epochs_by_channel[stim_label][ch])} epochs"
            )

    # ==========================================================
    # ERP
    # ==========================================================

    evoked_by_channel = {
        "no-stim": {},
        "stim": {},
    }

    for stim_label in [
        "no-stim",
        "stim",
    ]:

        for ch in OCCIPITAL_CHANNELS:

            evoked = (
                concat_epochs_by_channel[
                    stim_label
                ][ch]
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

            evoked.apply_baseline(
                baseline=(-0.1, 0),
            )

            evoked_by_channel[
                stim_label
            ][ch] = evoked

            evoked_fname = op.join(
                GROUP_DERIV_DIR,
                f"group_{stim_label}_{ch}_concat-ave.fif",
            )

            mne.write_evokeds(
                evoked_fname,
                evoked,
                overwrite=True,
            )

    # Plot ERP using channel-specific concatenations.
    # Channels with fewer contributing subjects remain explicitly
    # documented in the report.
    fig_erp, axes = plt.subplots(
        len(OCCIPITAL_CHANNELS),
        1,
        figsize=(
            10,
            4 * len(OCCIPITAL_CHANNELS),
        ),
        constrained_layout=True,
    )

    if len(OCCIPITAL_CHANNELS) == 1:
        axes = [axes]

    for ax, ch in zip(
        axes,
        OCCIPITAL_CHANNELS,
    ):

        mne.viz.plot_compare_evokeds(
            {
                "No stimulation": evoked_by_channel[
                    "no-stim"
                ][ch],
                "Stimulation": evoked_by_channel[
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

        ax.set_title(
            f"{ch} (n={subject_counts[ch]} subjects)\n"
            f"Cue onset = 0 s"
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

    fig_erp.suptitle(
        "Cue-locked ERP from concatenated epochs across subjects "
        "(cue onset = 0 s)",
        fontsize=14,
    )

    erp_fig_fname = op.join(
        GROUP_REPORT_DIR,
        "group_concatenated_epochs_ERP.png",
    )

    fig_erp.savefig(
        erp_fig_fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_erp,
        erp_fig_fname,
        "Cue-locked ERP from concatenated epochs across subjects",
        (
            "Cue-locked ERP; cue onset = 0 s. "
            "ERP is calculated separately for PO3, PO4 and POz, "
            "using only subjects with that channel available."
        ),
        "ERP",
    )

    # ==========================================================
    # TFR: CHANNEL-SPECIFIC
    # ==========================================================

    tfr_by_channel = {
        "no-stim": {},
        "stim": {},
    }

    for stim_label in [
        "no-stim",
        "stim",
    ]:

        for ch in OCCIPITAL_CHANNELS:

            print(
                f"Computing TFR: "
                f"{stim_label}, {ch}"
            )

            tfr_by_channel[
                stim_label
            ][ch] = (
                concat_epochs_by_channel[
                    stim_label
                ][ch]
                .compute_tfr(
                    **TFR_PARAMS
                )
            )

            tfr_fname = op.join(
                GROUP_DERIV_DIR,
                f"group_{stim_label}_{ch}_concat-tfr.h5",
            )

            tfr_by_channel[
                stim_label
            ][ch].save(
                tfr_fname,
                overwrite=True,
            )

    # ----------------------------------------------------------
    # Baseline-corrected condition TFRs
    #
    # These are ONLY used for the individual stim/no-stim plots.
    # The original TFRs in tfr_by_channel remain unchanged.
    # ----------------------------------------------------------

    tfr_stim_baselined = {}
    tfr_no_stim_baselined = {}

    for ch in OCCIPITAL_CHANNELS:

        tfr_stim_baselined[ch] = (
            tfr_by_channel["stim"][ch].copy()
        )

        tfr_no_stim_baselined[ch] = (
            tfr_by_channel["no-stim"][ch].copy()
        )

        tfr_stim_baselined[ch].apply_baseline(
            baseline=BASELINE,
            mode="percent",
        )

        tfr_no_stim_baselined[ch].apply_baseline(
            baseline=BASELINE,
            mode="percent",
        )


    # ----------------------------------------------------------
    # Difference and ratio
    #
    # IMPORTANT:
    # Both use the ORIGINAL, NON-BASELINE-CORRECTED TFRs.
    #
    # Difference:
    #     stim - no-stim
    #
    # Ratio:
    #     (stim - no-stim) / (stim + no-stim)
    # ----------------------------------------------------------

    tfr_diff = {}
    tfr_ratio = {}

    for ch in OCCIPITAL_CHANNELS:

        stim = tfr_by_channel[
            "stim"
        ][ch]

        no_stim = tfr_by_channel[
            "no-stim"
        ][ch]

        # Difference: RAW stim - RAW no-stim
        diff = stim.copy()

        diff.data = (
            stim.data
            - no_stim.data
        )

        tfr_diff[ch] = diff

        # Ratio: RAW (stim - no-stim) / (stim + no-stim)
        ratio = stim.copy()

        ratio.data = (
            stim.data
            - no_stim.data
        ) / (
            stim.data
            + no_stim.data
            + np.finfo(float).eps
        )

        tfr_ratio[ch] = ratio

    # ----------------------------------------------------------
    # Shared symmetric colour scale for difference and ratio
    #
    # Positive = stimulation > no-stimulation
    # Negative = stimulation < no-stimulation
    # ----------------------------------------------------------

    all_diff_values = np.concatenate([
        tfr_diff[ch].data.ravel()
        for ch in OCCIPITAL_CHANNELS
    ])

    all_ratio_values = np.concatenate([
        tfr_ratio[ch].data.ravel()
        for ch in OCCIPITAL_CHANNELS
    ])

    shared_vmax = max(
        np.nanmax(np.abs(all_diff_values)),
        np.nanmax(np.abs(all_ratio_values)),
    )

    shared_vlim = (
        -shared_vmax,
        shared_vmax,
    )

    # ----------------------------------------------------------
    # Channel-specific TFR figure: no stimulation
    # ----------------------------------------------------------

    fig_tfr_no = plot_three_channel_tfrs(
        tfr_no_stim_baselined,
        OCCIPITAL_CHANNELS,
        subject_counts,
        (
            "Concatenated epochs across subjects - "
            "Cue-locked TFR - no stimulation - combined attention"
        ),
        baseline=BASELINE,
        mode="percent",
        vlim=(-0.75, 0.75),
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_concatenated_epochs_TFR_no_stim.png",
    )

    fig_tfr_no.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_tfr_no,
        fname,
        "Concatenated epochs across subjects - no stimulation TFR",
        (    
            "Cue-locked TFR with cue onset = 0 s. "
            "Left- and right-attention trials are combined ('both'). "
            "Baseline: -0.3 to -0.1 s, percent change."
        ),
        "TFR",
    )

    # ----------------------------------------------------------
    # Channel-specific TFR figure: stimulation
    # ----------------------------------------------------------

    fig_tfr_stim = plot_three_channel_tfrs(
        tfr_stim_baselined,
        OCCIPITAL_CHANNELS,
        subject_counts,
        (
            "Concatenated epochs across subjects - "
            "Cue-locked TFR - stimulation - combined attention"
        ),
        baseline=BASELINE,
        mode="percent",
        vlim=(-0.75, 0.75),
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_concatenated_epochs_TFR_stim.png",
    )

    fig_tfr_stim.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_tfr_stim,
        fname,
        "Concatenated epochs across subjects - stimulation TFR",
        (
            "Cue-locked TFR with cue onset = 0 s. "
            "Left- and right-attention trials are combined ('both'). "
            "Baseline: -0.3 to -0.1 s, percent change."
        ),
        "TFR",
    )

    # ----------------------------------------------------------
    # Channel-specific TFR figure: difference
    # ----------------------------------------------------------

    fig_diff = plot_three_channel_tfrs(
        tfr_diff,
        OCCIPITAL_CHANNELS,
        subject_counts,
        (
            "Concatenated epochs across subjects - "
            "Cue-locked TFR difference - no stimulation minus stimulation"
        ),
        baseline=None,
        mode=None,
        vlim=shared_vlim,
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_concatenated_epochs_TFR_difference.png",
    )

    fig_diff.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_diff,
        fname,
        "Concatenated epochs across subjects - TFR difference",
        (
            "Cue-locked TFR difference with cue onset = 0 s. "
            "Left- and right-attention trials are combined. "
            "Difference = no-stimulation minus stimulation after "
            "No baseline correction was applied."
        ),
        "TFR",
    )

    # ----------------------------------------------------------
    # Channel-specific TFR figure: ratio
    # ----------------------------------------------------------

    fig_ratio = plot_three_channel_tfrs(
        tfr_ratio,
        OCCIPITAL_CHANNELS,
        subject_counts,
        (
            "Concatenated epochs across subjects - "
            "Cue-locked TFR ratio - no stimulation versus stimulation"
        ),
        baseline=None,
        mode=None,
        vlim=shared_vlim,
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_concatenated_epochs_TFR_ratio.png",
    )

    fig_ratio.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_ratio,
        fname,
        "Concatenated epochs across subjects - TFR ratio",
        (
            "Cue-locked TFR ratio with cue onset = 0 s. "
            "Left- and right-attention trials are combined. "
            "Ratio = (no-stimulation - stimulation) / "
            "(no-stimulation + stimulation)."
            "No baseline correction was applied."
        ),
        "TFR",
    )

    # ==========================================================
    # POSTERIOR ROI
    # ==========================================================
    #
    # For each subject, average all available posterior channels
    # within each epoch.
    #
    # Left- and right-attention trials are combined because the input
    # epochs are cue-locked epochs containing both cue directions.
    # Cue onset = 0 s.
    # ==========================================================

    roi_epochs_by_condition = {
        "no-stim": [],
        "stim": [],
    }

    roi_subjects = []
    roi_channel_summary = []


    for subject in subjects:

        subject_has_roi = False

        # Determine which posterior channels are available in this subject.
        available_channels = [
            ch
            for ch in OCCIPITAL_CHANNELS
            if ch in subject_epochs[subject]["no-stim"].ch_names
            and ch in subject_epochs[subject]["stim"].ch_names
        ]

        if not available_channels:
            raise RuntimeError(
                f"sub-{subject} has no posterior channels available "
                "for both stimulation conditions."
            )

        roi_channel_summary.append(
            f"sub-{subject}: {', '.join(available_channels)}"
        )

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

            if not available:
                raise RuntimeError(
                    f"sub-{subject} has no posterior channels available "
                    f"for {stim_label}."
                )

            epochs.pick(
                available
            )

            # ------------------------------------------------------
            # Average available posterior channels within each epoch.
            # ------------------------------------------------------

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

            roi_epochs_by_condition[
                stim_label
            ].append(
                roi_epochs
            )

            subject_has_roi = True

        if subject_has_roi:
            roi_subjects.append(
                subject
            )


    # ----------------------------------------------------------
    # Report ROI definition
    # ----------------------------------------------------------

    report.add_text(
        "Posterior ROI definition",
        (
            "The posterior ROI is calculated from cue-locked epochs "
            "(cue onset = 0 s). Left- and right-attention trials are "
            "combined ('both').\n\n"
            "For each subject, the available posterior channels "
            "(PO3, PO4, POz) are averaged within each epoch before "
            "the epochs are concatenated across subjects.\n\n"
            "Posterior channels available for each subject:\n"
            + "\n".join(roi_channel_summary)
            + "\n\n"
            "This means that participants with a rejected/missing "
            "posterior channel can still contribute to the ROI using "
            "their remaining posterior channels."
        ),
        "Analysis details",
    )


    # ==========================================================
    # Concatenate posterior ROI epochs
    # ==========================================================

    roi_concat = {}

    for stim_label in [
        "no-stim",
        "stim",
    ]:

        roi_concat[
            stim_label
        ] = mne.concatenate_epochs(
            roi_epochs_by_condition[
                stim_label
            ]
        )

        fname = op.join(
            GROUP_DERIV_DIR,
            f"group_{stim_label}_posterior-ROI_concat-epo.fif",
        )

        roi_concat[
            stim_label
        ].save(
            fname,
            overwrite=True,
        )


    # ==========================================================
    # ROI TFR
    # ==========================================================
    #
    # TFR is calculated from cue-locked epochs containing the
    # combined left- and right-attention conditions.
    #
    # Frequency range: 2-31.5 Hz
    # Multitaper
    # n_cycles = frequency / 2
    # time-bandwidth = 2
    # decimation = 2
    # ==========================================================

    roi_tfr = {}

    for stim_label in [
        "no-stim",
        "stim",
    ]:

        roi_tfr[
            stim_label
        ] = roi_concat[
            stim_label
        ].compute_tfr(
            **TFR_PARAMS
        )

        fname = op.join(
            GROUP_DERIV_DIR,
            f"group_{stim_label}_posterior-ROI_concat-tfr.h5",
        )

        roi_tfr[
            stim_label
        ].save(
            fname,
            overwrite=True,
        )


    # ==========================================================
    # ROI difference
    # ==========================================================
    #
    # Both TFRs are not baseline corrected.
    #
    # Difference:
    #
    #     stim - no-stim
    # ==========================================================

    roi_stim = roi_tfr[
        "stim"
    ].copy()

    roi_no_stim = roi_tfr[
        "no-stim"
    ].copy()

    roi_diff = roi_no_stim.copy()

    roi_diff.data = (
        roi_stim.data
        - roi_no_stim.data
    )


    # ==========================================================
    # ROI ratio
    # ==========================================================
    #
    # Ratio:
    #
    #     (stim - no-stim) / (stim + no-stim)
    #
    # This is calculated from the original TFR values rather than
    # the baseline-corrected values.
    # ==========================================================

    roi_ratio = roi_tfr[
        "no-stim"
    ].copy()

    roi_ratio.data = (
        roi_tfr["stim"].data
        - roi_tfr["no-stim"].data
    ) / (
        roi_tfr["stim"].data
        + roi_tfr["no-stim"].data
        + np.finfo(float).eps
    )

    # ----------------------------------------------------------
    # Shared symmetric colour scale for ROI difference and ratio
    # ----------------------------------------------------------

    roi_diff_vmax = np.nanmax(
        np.abs(roi_diff.data)
    )

    roi_ratio_vmax = np.nanmax(
        np.abs(roi_ratio.data)
    )

    roi_shared_vmax = max(
        roi_diff_vmax,
        roi_ratio_vmax,
    )

    roi_shared_vlim = (
        -roi_shared_vmax,
        roi_shared_vmax,
    )

    # ==========================================================
    # ROI no-stimulation plot
    # ==========================================================

    fig_roi_no = plot_single_roi_tfr(
        roi_tfr["no-stim"],
        (
            "Cue-locked posterior ROI TFR - "
            "no stimulation (combined attention)"
        ),
        baseline=BASELINE,
        mode="percent",
        vlim=(-0.75, 0.75),
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_concatenated_epochs_TFR_posterior_ROI_no_stim.png",
    )

    fig_roi_no.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_roi_no,
        fname,
        "Cue-locked posterior ROI TFR - no stimulation",
        (
            "Cue onset = 0 s. Left- and right-attention trials are "
            "combined ('both'). The posterior ROI is the mean of the "
            "available posterior channels within each subject before "
            "epoch concatenation. "
            "TFR baseline: -0.3 to -0.1 s, percent change. "
            "Frequency range: 2-31.5 Hz."
        ),
        "TFR",
    )


    # ==========================================================
    # ROI stimulation plot
    # ==========================================================

    fig_roi_stim = plot_single_roi_tfr(
        roi_tfr["stim"],
        (
            "Cue-locked posterior ROI TFR - "
            "stimulation (combined attention)"
        ),
        baseline=BASELINE,
        mode="percent",
        vlim=(-0.75, 0.75),
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_concatenated_epochs_TFR_posterior_ROI_stim.png",
    )

    fig_roi_stim.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_roi_stim,
        fname,
        "Cue-locked posterior ROI TFR - stimulation",
        (
            "Cue onset = 0 s. Left- and right-attention trials are "
            "combined ('both'). The posterior ROI is the mean of the "
            "available posterior channels within each subject before "
            "epoch concatenation. "
            "TFR baseline: -0.3 to -0.1 s, percent change. "
            "Frequency range: 2-31.5 Hz."
        ),
        "TFR",
    )


    # ==========================================================
    # ROI difference plot
    # ==========================================================

    fig_roi_diff = plot_single_roi_tfr(
        roi_diff,
        (
            "Cue-locked posterior ROI TFR - "
            "difference (stim - no-stim)"
        ),
        baseline=None,
        mode=None,
        vlim=roi_shared_vlim,
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_concatenated_epochs_TFR_posterior_ROI_difference.png",
    )

    fig_roi_diff.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_roi_diff,
        fname,
        "Cue-locked posterior ROI TFR - difference",
        (
            "Cue onset = 0 s. Left- and right-attention trials are "
            "combined ('both'). Difference = stimulation minus "
            "no-stimulation. No baseline correction. "
            "Frequency range: 2-31.5 Hz."
        ),
        "TFR",
    )


    # ==========================================================
    # ROI ratio plot
    # ==========================================================

    fig_roi_ratio = plot_single_roi_tfr(
        roi_ratio,
        (
            "Cue-locked posterior ROI TFR - "
            "ratio (stim - no-stim)"
        ),
        baseline=None,
        mode=None,
        vlim=roi_shared_vlim,
    )

    fname = op.join(
        GROUP_REPORT_DIR,
        "group_concatenated_epochs_TFR_posterior_ROI_ratio.png",
    )

    fig_roi_ratio.savefig(
        fname,
        dpi=180,
        bbox_inches="tight",
    )

    report.add_figure(
        fig_roi_ratio,
        fname,
        "Cue-locked posterior ROI TFR - ratio",
        (
            "Cue onset = 0 s. Left- and right-attention trials are "
            "combined ('both'). "
            "Ratio = (stim - no-stim) / (stim + no-stim). "
            "Frequency range: 2-31.5 Hz."
        ),
        "TFR",
    )
    # ==========================================================
    # Analysis notes
    # ==========================================================

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


# ==============================================================
# Run
# ==============================================================

if __name__ == "__main__":

    args = parse_args()

    subjects = [
        str(s).removeprefix("sub-")
        for s in args.subjects
    ]

    print(
        "\nSubjects included in G01:"
        f"\n  {', '.join('sub-' + s for s in subjects)}\n"
    )

    build_concat_epoch_report(
        subjects
    )
