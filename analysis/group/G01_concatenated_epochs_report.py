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
       no-stim - stim

    4. Ratio TFR
       (no-stim - stim) / (no-stim + stim)

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
import matplotlib.pyplot as plt

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

DEFAULT_SUBJECTS = [
    "115",
    "116",
    "118",
    "119",
]

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

    tfr.plot(
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
        vlim=vlim,
    )

    ax.set_title(title)

    return fig


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
    # Document previous interpolation
    # ----------------------------------------------------------

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
        "Subject-level channel handling",
        "\n".join(analysis_data_lines),
        "Group analysis",
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
            f"{ch} "
            f"(n={subject_counts[ch]} subjects)"
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

    fig_erp.suptitle(
        "ERP from concatenated epochs across subjects",
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
        "Concatenated epochs across subjects - ERP",
        (
            "ERP computed separately for each posterior channel. "
            "The number of contributing subjects is shown for each channel."
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
    # Difference and ratio
    # ----------------------------------------------------------

    tfr_diff = {}
    tfr_ratio = {}

    for ch in OCCIPITAL_CHANNELS:

        stim_bc = tfr_by_channel[
            "stim"
        ][ch].copy()

        no_stim_bc = tfr_by_channel[
            "no-stim"
        ][ch].copy()

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

        tfr_diff[ch] = diff

        no_stim = tfr_by_channel[
            "no-stim"
        ][ch]

        stim = tfr_by_channel[
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

        tfr_ratio[ch] = ratio

    # ----------------------------------------------------------
    # Channel-specific TFR figure: no stimulation
    # ----------------------------------------------------------

    fig_tfr_no = plot_three_channel_tfrs(
        tfr_by_channel["no-stim"],
        OCCIPITAL_CHANNELS,
        subject_counts,
        (
            "Concatenated epochs across subjects - "
            "no stimulation TFR"
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
            "Three posterior channels shown separately. "
            "Each panel displays the subjects contributing to that channel."
        ),
        "TFR",
    )

    # ----------------------------------------------------------
    # Channel-specific TFR figure: stimulation
    # ----------------------------------------------------------

    fig_tfr_stim = plot_three_channel_tfrs(
        tfr_by_channel["stim"],
        OCCIPITAL_CHANNELS,
        subject_counts,
        (
            "Concatenated epochs across subjects - "
            "stimulation TFR"
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
            "Three posterior channels shown separately."
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
            "TFR difference: no-stim - stim"
        ),
        baseline=None,
        mode=None,
        vlim=None,
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
        "Difference = no-stim - stim.",
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
            "TFR ratio"
        ),
        baseline=None,
        mode=None,
        vlim=None,
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
            "Ratio = "
            "(no-stim - stim) / "
            "(no-stim + stim)."
        ),
        "TFR",
    )

    # ==========================================================
    # POSTERIOR ROI
    # ==========================================================
    #
    # For each subject, average all available posterior channels
    # within each epoch. This means:
    #
    #     sub-115 -> PO3 + PO4 + POz
    #     sub-116 -> PO3 + PO4 + POz
    #     sub-118 -> PO3 + PO4 + POz
    #     sub-119 -> PO3 + PO4
    #
    # The resulting subject-level ROI epochs are then concatenated.
    # ==========================================================

    roi_epochs_by_condition = {
        "no-stim": [],
        "stim": [],
    }

    roi_subjects = []

    for subject in subjects:

        subject_has_roi = False

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
                    f"sub-{subject} has no posterior "
                    f"channels available for {stim_label}."
                )

            epochs.pick(
                available
            )

            # Average available posterior channels within each epoch.
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

    report.add_text(
        "Posterior ROI definition",
        (
            "For the combined posterior ROI, the available posterior "
            "channels were averaged within each epoch for each subject "
            "before concatenation.\n\n"
            + "\n".join(
                f"sub-{subject}: "
                + ", ".join(
                    ch
                    for ch in OCCIPITAL_CHANNELS
                    if ch in subject_epochs[subject]["no-stim"].ch_names
                )
                for subject in roi_subjects
            )
        ),
        "Group analysis",
    )

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

    # ----------------------------------------------------------
    # ROI TFR
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # ROI difference
    # ----------------------------------------------------------

    roi_stim_bc = roi_tfr[
        "stim"
    ].copy()

    roi_no_stim_bc = roi_tfr[
        "no-stim"
    ].copy()

    roi_stim_bc.apply_baseline(
        baseline=BASELINE,
        mode="percent",
    )

    roi_no_stim_bc.apply_baseline(
        baseline=BASELINE,
        mode="percent",
    )

    roi_diff = roi_no_stim_bc.copy()

    roi_diff.data = (
        roi_no_stim_bc.data
        - roi_stim_bc.data
    )

    # ----------------------------------------------------------
    # ROI ratio
    # ----------------------------------------------------------

    roi_ratio = roi_tfr[
        "no-stim"
    ].copy()

    roi_ratio.data = (
        roi_tfr["no-stim"].data
        - roi_tfr["stim"].data
    ) / (
        roi_tfr["no-stim"].data
        + roi_tfr["stim"].data
        + np.finfo(float).eps
    )

    # ----------------------------------------------------------
    # ROI no-stim plot
    # ----------------------------------------------------------

    fig_roi_no = plot_single_roi_tfr(
        roi_tfr["no-stim"],
        (
            "Posterior ROI TFR from concatenated epochs - "
            "no stimulation"
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
        "Posterior ROI TFR - no stimulation",
        (
            "Posterior ROI is the mean of the available posterior "
            "channels within each subject."
        ),
        "TFR",
    )

    # ----------------------------------------------------------
    # ROI stim plot
    # ----------------------------------------------------------

    fig_roi_stim = plot_single_roi_tfr(
        roi_tfr["stim"],
        (
            "Posterior ROI TFR from concatenated epochs - "
            "stimulation"
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
        "Posterior ROI TFR - stimulation",
        (
            "Posterior ROI is the mean of the available posterior "
            "channels within each subject."
        ),
        "TFR",
    )

    # ----------------------------------------------------------
    # ROI difference plot
    # ----------------------------------------------------------

    fig_roi_diff = plot_single_roi_tfr(
        roi_diff,
        (
            "Posterior ROI TFR from concatenated epochs - "
            "difference: no-stim - stim"
        ),
        baseline=None,
        mode=None,
        vlim=None,
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
        "Posterior ROI TFR - difference",
        "Difference = no-stim - stim.",
        "TFR",
    )

    # ----------------------------------------------------------
    # ROI ratio plot
    # ----------------------------------------------------------

    fig_roi_ratio = plot_single_roi_tfr(
        roi_ratio,
        (
            "Posterior ROI TFR from concatenated epochs - "
            "ratio"
        ),
        baseline=None,
        mode=None,
        vlim=None,
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
        "Posterior ROI TFR - ratio",
        (
            "Ratio = "
            "(no-stim - stim) / "
            "(no-stim + stim)."
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

    subjects = (
        load_subject_list(
            SUBJECT_LIST_PATH
        )
        if Path(
            SUBJECT_LIST_PATH
        ).exists()
        else DEFAULT_SUBJECTS
    )

    build_concat_epoch_report(
        subjects
    )

