"""
==============================================================
Group analysis utilities
==============================================================

Shared helper functions for the group-level EEG analyses:

    G01_concatenated_epochs_report.py
        Concatenates cleaned epochs across subjects before
        calculating group ERP and TFR.

    G02_grand_average_report.py
        Calculates subject-level averages first and then
        combines subjects using mne.grand_average().

This module provides functions to:
    - load and save the group subject list
    - locate subject derivative folders
    - load cleaned epochs, evoked responses, and TFRs
    - handle missing posterior channels
    - create standard ERP and TFR figures
    - create persistent PDF reports
    - add the included subjects and analysis notes to reports

Expected inputs are outputs from the completed subject-level pipeline
stored under:

    <BIDS_ROOT>/derivatives/sub-<subject>/

Group analyses currently focus on posterior channels PO3, PO4, and POz.
A channel may be absent if it was rejected during subject-level cleaning.

This module contains shared utilities only; the actual group analyses
are performed by G01 and G02.

written by Tara Ghafari
tara.ghafari@gmail.com
==============================================================
"""


from __future__ import annotations

import json
import os
import os.path as op
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.time_frequency import read_tfrs

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
UTILS_DIR = REPO_ROOT / "analysis" / "utils"

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))


def ensure_dir(path: str | Path) -> str:
    path = str(path)
    os.makedirs(path, exist_ok=True)
    return path


def load_subject_list(subject_list_path: str) -> List[str]:
    path = Path(subject_list_path)
    if not path.exists():
        raise FileNotFoundError(f"Subject list not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data.get("subjects", [])
    return [str(s).removeprefix("sub-") for s in subjects]


def save_subject_list(subject_list_path: str, subjects: Sequence[str]) -> None:
    path = Path(subject_list_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"subjects": [f"sub-{str(s).removeprefix('sub-')}" for s in subjects]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def subject_deriv_folder(bids_root: str, subject: str) -> str:
    return op.join(bids_root, "derivatives", f"sub-{subject}")


def read_subject_epochs(bids_root: str, subject: str) -> dict:
    """
    Load cue-locked epochs for group analysis.

    If a participant had a rejected posterior channel that was
    interpolated for group analysis, the file with the suffix
    '-epo-cue-group.fif' is used.

    Otherwise, the normal subject-level '-epo-cue.fif' file is used.

    Returns
    -------
    dict
        Keys:
            'no-stim'
            'stim'

        Values:
            MNE Epochs objects.
    """

    deriv_folder = subject_deriv_folder(
        bids_root,
        subject,
    )

    base = (
        f"sub-{subject}"
        "_ses-01"
        "_task-SpAtt"
        "_run-01"
        "_eeg"
    )
    out = {}

    for stim_label in ["no-stim", "stim"]:
        # First look for the group-ready version.
        group_fname = op.join(
            deriv_folder,
            f"{base}_{stim_label}_epo-cue-group.fif",
        )
        # If it does not exist, use the normal cleaned epochs.
        normal_fname = op.join(
            deriv_folder,
            f"{base}_{stim_label}_epo-cue.fif",
        )
        if op.exists(group_fname):
            fname = group_fname
            print(
                f"sub-{subject} {stim_label}: "
                f"using interpolated group-analysis epochs"
            )
        elif op.exists(normal_fname):
            fname = normal_fname
            print(
                f"sub-{subject} {stim_label}: "
                f"using standard cleaned epochs"
            )
        else:
            raise FileNotFoundError(
                f"Neither group nor standard epochs were found for "
                f"sub-{subject} {stim_label}:\n"
                f"  {group_fname}\n"
                f"  {normal_fname}"
            )
        out[stim_label] = mne.read_epochs(
            fname,
            preload=True,
            verbose=True,
        )

    return out


def read_subject_evokeds(bids_root: str, subject: str) -> dict:
    deriv_folder = subject_deriv_folder(bids_root, subject)
    base = f"sub-{subject}_ses-01_task-SpAtt_run-01_eeg"
    out = {}
    for stim_label in ["no-stim", "stim"]:
        cue_fname = op.join(deriv_folder, f"{base}_{stim_label}_evo-cue.fif")
        grating_fname = op.join(deriv_folder, f"{base}_{stim_label}_evo-grating.fif")
        if not op.exists(cue_fname):
            raise FileNotFoundError(f"Missing evoked file: {cue_fname}")
        if not op.exists(grating_fname):
            raise FileNotFoundError(f"Missing evoked file: {grating_fname}")
        out[(stim_label, "cue")] = mne.read_evokeds(cue_fname, condition=0, verbose=True)
        out[(stim_label, "grating")] = mne.read_evokeds(grating_fname, condition=0, verbose=True)
    return out


def read_subject_tfrs(bids_root: str, subject: str) -> dict:
    deriv_folder = subject_deriv_folder(bids_root, subject)
    base = f"sub-{subject}_ses-01_task-SpAtt_run-01_eeg"
    out = {}
    for stim_label in ["no-stim", "stim"]:
        for cue in ["both", "right", "left"]:
            fname = op.join(deriv_folder, f"{base}_{cue}_{stim_label}_tfr.h5")
            if not op.exists(fname):
                raise FileNotFoundError(f"Missing TFR file: {fname}")
            out[(stim_label, cue)] = read_tfrs(fname)[0]
    return out

def read_interpolation_summary(bids_root: str, subject: str) -> list[str]:
    """
    Read the list of posterior channels interpolated for group analysis.

    Parameters
    ----------
    bids_root : str
        Root of the BIDS directory.

    subject : str
        Subject number without the 'sub-' prefix.

    Returns
    -------
    list of str
        Names of channels interpolated for this participant.
        Returns an empty list if no interpolation record exists.
    """

    fname = op.join(
        bids_root,
        "derivatives",
        f"sub-{subject}",
        "qc",
        f"sub-{subject}_group_interpolation.json",
    )

    if not op.exists(fname):
        return []

    with open(fname, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("interpolated_channels", [])

def make_report(report_folder: str, report_name: str):
    from pdf_report import ParticipantPDF  # your persistent helper

    report_folder = ensure_dir(report_folder)
    return ParticipantPDF(report_folder, report_name)


def add_subject_summary(report, subjects: Sequence[str], title: str = "Subjects included") -> None:
    text = "\n".join([f"sub-{s}" for s in subjects]) if subjects else "None"
    report.add_text(title, text, "Group analysis")


def add_analysis_notes_section(report, prompt_text: str = "Analysis notes") -> None:
    notes = input("\nEnter analysis notes (press Enter to skip): ").strip()
    if notes:
        report.add_text(prompt_text, notes, "Group analysis")


def subset_present_channels(ch_names: Sequence[str], wanted: Sequence[str]) -> List[str]:
    return [ch for ch in wanted if ch in ch_names]


def plot_compare_evokeds_by_channel(evoked_dict: dict, channels: Sequence[str], title: str):
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 4 * len(channels)), constrained_layout=True)
    if len(channels) == 1:
        axes = [axes]
    for ax, ch in zip(axes, channels):
        mne.viz.plot_compare_evokeds(
            evoked_dict,
            picks=ch,
            combine="mean",
            axes=ax,
            show=False,
            ci=False,
            truncate_xaxis=False,
            truncate_yaxis=False,
        )
        ax.axvline(0, color="k", linestyle="--", linewidth=1)
        ax.set_title(f"{title} - {ch}")
    return fig


def plot_tfr_channel_grid(tfr, channels: Sequence[str], title: str, baseline=None, mode=None):
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 4 * len(channels)), constrained_layout=True)
    if len(channels) == 1:
        axes = [axes]
    for ax, ch in zip(axes, channels):
        tfr.plot(
            picks=ch,
            tmin=-0.5,
            tmax=1.5,
            baseline=baseline,
            mode=mode,
            axes=ax,
            show=False,
            colorbar=True,
        )
        ax.set_title(f"{title} - {ch}")
    return fig