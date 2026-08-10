# -*- coding: utf-8 -*-
"""
===============================================
Single-subject STN EEG pipeline

This script runs the full single-participant workflow based on your analysis folder:
    1. read BrainVision EEG
    2. create / update BIDS
    3. write a participant PDF report
    4. segment stimulation on and stimulation off
    5. plot PSDs for QC before filtering
    6. low-pass / high-pass filter the segmented data
    7. epoch cue-locked trials
    8. show the epoch time series so bad trials can be removed manually
    9. compute epoch PSD and record bad-channel QC
   10. compute cue-locked evoked responses (ERP)
   11. compute posterior-channel TFRs (TFR)
   12. save all outputs and figures into the participant report

The script is designed for one participant at a time.
Manual rejection is intentionally interactive: the epoch browser opens so
bad epochs can be removed by inspection of the time series.

written to mirror the structure and style of the existing analysis scripts
===============================================
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as op
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids

# -----------------------------------------------------------------------------
# Paths: edit these defaults if needed.
# -----------------------------------------------------------------------------
PROJECT_ROOT_DEFAULT = "/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/STN-in-PD"
GITHUB_ROOT_DEFAULT = "/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/GitHub/STN-stimulation-oscillation"
DATA_ROOT_DEFAULT = op.join(PROJECT_ROOT_DEFAULT, "data", "data-organised")
BIDS_ROOT_DEFAULT = op.join(PROJECT_ROOT_DEFAULT, "data", "BIDS")

# -----------------------------------------------------------------------------
# Analysis defaults.
# -----------------------------------------------------------------------------
SESSION_DEFAULT = "01"
TASK_DEFAULT = "SpAtt"
RUN_DEFAULT = "01"
MODALITY = "eeg"
EOG_RENAME = {"T8": "vEOG1", "FT10": "vEOG2", "T7": "hEOG1", "FT9": "hEOG2"}
EOG_TYPES = {"vEOG1": "eog", "vEOG2": "eog", "hEOG1": "eog", "hEOG2": "eog"}
ALWAYS_BAD_CHANNELS = ["TP9", "TP10"]
PICKED_POSTERIOR_CHANNELS = ["PO3", "PO4", "POz"]
ERP_POSTERIOR_CHANNELS = ["PO3", "PO4", "POz"]

EVENT_DICT = {
    "cue_onset_right": 1,
    "cue_onset_left": 2,
    "trial_onset": 3,
    "stim_onset": 4,
    "catch_onset": 5,
    "dot_onset_right": 6,
    "dot_onset_left": 7,
    "response_press_onset": 8,
    "block_onset": 20,
    "block_end": 21,
    "experiment_end": 30,
    "new_stim_segment": 99999,
}

# This is the crop table used for segmentation.
# Update / extend this if you add more subjects.
STIMULATION_CROPPED_TIME: Dict[str, Dict[str, List[float]]] = {
    "sub-107": {"no-stim": [15, 974], "stim": [1000, 1845]},
    "sub-108": {"no-stim": [8, 890], "stim": [930, 1882]},
    "sub-110": {"no-stim": [905, 1711], "stim": [0, 840]},
    "sub-102": {"no-stim": [0, 965], "stim": [1490, 2230]},
    "sub-101": {"no-stim": [0, 360, 515, 865], "stim": [1144, 1900]},
    "sub-112": {"no-stim": [878, 1289, 1870, 2280], "stim": [244, 650, 2708, 3117]},
    "sub-103": {"no-stim": [785, 1113, 1380, 1715], "stim": [72, 476, 1862, 2182]},
    "sub-104": {"no-stim": [1100, 1412, 1946, 2269], "stim": [9, 772]},
    "sub-105": {"no-stim": [4326, 5103], "stim": [112, 597, 850, 1327]},
    "sub-113": {"no-stim": [144, 507, 883, 1233], "stim": [1895, 2255, 2306, 2668]},
    "sub-115": {"no-stim": [1523, 1960, 2353, 2750], "stim": [30, 620, 768, 1235]},
}

# -----------------------------------------------------------------------------
# PDF helper import.
# -----------------------------------------------------------------------------

def _setup_import_path(github_root: str) -> None:
    if github_root not in sys.path:
        sys.path.insert(0, github_root)
    utils_dir = os.path.join(github_root, "analysis", "utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)


def _load_report_helper(github_root: str):
    _setup_import_path(github_root)
    from pdf_report import ParticipantPDF, impedance_text  # type: ignore

    return ParticipantPDF, impedance_text


# -----------------------------------------------------------------------------
# Utility functions.
# -----------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> str:
    path = str(path)
    os.makedirs(path, exist_ok=True)
    return path


def pretty_list(values: Sequence[object]) -> str:
    items = [str(v) for v in values]
    return ", ".join(items) if items else "None"


def save_fig(fig, path: str) -> str:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def crop_table_to_text(crop_table: Dict[str, List[float]]) -> str:
    lines = []
    for label in ["no-stim", "stim"]:
        times = crop_table[label]
        lines.append(f"{label}: {pretty_list(times)}")
    return "\n".join(lines)


def parse_crop_times(text_value: str) -> List[float]:
    cleaned = text_value.replace(";", ",").replace(" ", ",")
    parts = [p for p in cleaned.split(",") if p.strip()]
    times = [float(p) for p in parts]
    if len(times) not in (2, 4):
        raise ValueError("Please enter either 2 numbers (one segment) or 4 numbers (two segments).")
    if any(times[i] >= times[i + 1] for i in range(0, len(times), 2)):
        raise ValueError("Each crop pair must satisfy start < end.")
    return times


def prompt_for_crop_table(subject: str, bids_path: BIDSPath, default_crop_table: Dict[str, List[float]], bids_root: str, report, fig_folder: str) -> Dict[str, List[float]]:
    """Open the BIDS raw data and let the user update crop times interactively."""
    raw = read_raw_bids(bids_path=bids_path, verbose=True, extra_params={"preload": True})
    print("\nOpen the raw viewer now. Close it when you are done inspecting stim on / stim off periods.")
    raw.plot()
    input("After closing the raw viewer, press Enter to enter crop times...")

    crop_table = {
        "no-stim": list(default_crop_table["no-stim"]),
        "stim": list(default_crop_table["stim"]),
    }
    print("\nCurrent crop table:")
    print(crop_table_to_text(crop_table))

    for label in ["no-stim", "stim"]:
        current = crop_table[label]
        response = input(
            f"Enter new crop times for {label} (2 or 4 numbers, comma/space-separated)\n"
            f"Press Enter to keep current [{pretty_list(current)}]: "
        ).strip()
        if response:
            crop_table[label] = parse_crop_times(response)

    print("\nUpdated crop table:")
    print(crop_table_to_text(crop_table))

    crop_dir = ensure_dir(op.join(bids_root, "derivatives", f"sub-{subject}", "qc"))
    crop_json = op.join(crop_dir, f"sub-{subject}_crop_table.json")
    with open(crop_json, "w", encoding="utf-8") as f:
        json.dump(crop_table, f, indent=2)

    add_report_text(
        report,
        "Updated crop table",
        f"Crop times were reviewed interactively and saved to: {crop_json}\n{crop_table_to_text(crop_table)}",
        "Stimulation segmentation",
    )
    return crop_table


def get_stim_sequence(subject: str) -> List[str] | None:
    stim_sequence = {
        "sub-01": ["no_stim-left rec", "no_stim-right rec", "Right stim- no rec", "Left stim- no rec"],
        "sub-02": ["no_stim-left rec", "no_stim-right rec", "Left stim- no rec", "Right stim- no rec"],
        "sub-05": ["Left stim- no rec", "Right stim- no rec", "no_stim-left rec", "no_stim-right rec"],
        "sub-107": ["no_stim-right rec", "no_stim-left rec", "Right stim- no rec", "Left stim- no rec"],
        "sub-108": ["no_stim-right rec", "no_stim-left rec", "left stim- no rec", "right stim- no rec"],
        "sub-110": ["Right stim- no rec", "Left stim- no rec", "no_stim-no rec", "no_stim-no rec"],
        "sub-102": ["no_stim-left rec", "no_stim-right rec", "Left stim- no rec", "Right stim- no rec"],
        "sub-101": ["no_stim-left rec", "no_stim-right-rec", "Right stim- no rec", "Left stim- no rec"],
        "sub-111": ["Left stim- no rec", "Right stim- no rec", "no_stim-right rec", "no_stim-left rec"],
        "sub-112": ["Left stim- no rec", "no_stim-right rec", "no_stim-left rec", "Right stim- no rec"],
        "sub-103": ["Right stim- no rec", "no_stim-left rec", "no_stim-right rec", "Left stim- no rec"],
        "sub-104": ["Right stim- no rec", "Left stim- no rec", "no_stim-left rec", "no_stim-right rec"],
        "sub-105": ["Left stim- no rec", "Right stim- no rec", "no_stim-left rec", "no_stim-right rec"],
        "sub-113": ["no_stim-left rec", "no_stim-right rec", "Right stim- no rec", "Left stim- no rec"],
        "sub-114": ["no_stim-left rec", "no_stim-right rec", "Left stim- no rec", "Right stim- no rec"],
        "sub-115": ["Right stim- no rec", "no_stim-left rec", "no_stim-right rec", "Left stim- no rec"],
        "sub-116": ["Right stim- no rec", "Left stim- no rec", "no_stim-left rec", "no_stim-right rec"],
        "sub-117": ["Left stim- no rec", "Right stim- no rec", "no_stim-left rec", "no_stim-right rec"],
        "sub-118": ["Left stim- no rec", "no_stim-left rec", "no_stim-right rec", "Right stim- no rec"],
        "sub-119": ["no_stim-right rec", "no_stim-left rec", "Right stim- no rec", "Left stim- no rec"],
        "sub-120": ["no_stim-right rec", "no_stim-left rec", "Left stim- no rec", "Right stim- no rec"],
        "sub-121": ["Right stim- no rec", "no_stim-right rec", "no_stim-left rec", "Left stim- no rec"],
        "sub-122": ["Right stim- no rec", "Left stim- no rec", "no_stim-right rec", "no_stim-left rec"],
        "sub-123": ["Left stim- no rec", "Right stim- no rec", "no_stim-right rec", "no_stim-left rec"],
    }
    return stim_sequence.get(subject)


def get_bad_channel_reasons(raw: mne.io.BaseRaw):
    """Use PyPREP when available; fall back to an empty suggestion set."""
    try:
        from pyprep.find_noisy_channels import NoisyChannels
    except Exception as exc:
        print(f"PyPREP not available: {exc}")
        return {}

    eeg = raw.copy().pick("eeg")
    if eeg.get_montage() is None:
        eeg.set_montage("standard_1020", on_missing="warn")

    noisy = NoisyChannels(eeg, random_state=42)
    detectors = [
        ("deviation", noisy.find_bad_by_deviation),
        ("high-frequency noise", noisy.find_bad_by_hfnoise),
        ("correlation", noisy.find_bad_by_correlation),
        ("RANSAC", noisy.find_bad_by_ransac),
    ]
    reasons: Dict[str, List[str]] = {}
    for reason, detector in detectors:
        try:
            detector()
        except Exception as exc:
            print(f"PyPREP {reason} detector skipped: {exc}")

    attr_map = {
        "bad_by_deviation": "deviation",
        "bad_by_hf_noise": "high-frequency noise",
        "bad_by_correlation": "correlation",
        "bad_by_ransac": "RANSAC",
        "bad_by_nan": "NaN/flat data",
        "bad_by_SNR": "poor signal-to-noise ratio",
    }
    for attr, reason in attr_map.items():
        for ch in getattr(noisy, attr, []) or []:
            reasons.setdefault(str(ch), []).append(reason)
    for ch in noisy.get_bads():
        reasons.setdefault(str(ch), []).append("PyPREP overall noisy-channel decision")
    return {ch: sorted(set(vals)) for ch, vals in reasons.items()}


def add_report_text(report, title: str, text: str, section: str) -> None:
    report.add_text(title, text, section)


# -----------------------------------------------------------------------------
# Step 1: BrainVision -> BIDS.
# -----------------------------------------------------------------------------

def step_p01_bids_conversion(
    subject: str,
    session: str,
    task: str,
    run: str,
    project_root: str,
    data_root: str,
    bids_root: str,
    report,
    fig_folder: str,
):
    base_fpath = op.join(data_root, f"sub-{subject}", f"ses-{session}", MODALITY)
    brainvision_basename = f"AO{subject[1:]}"
    events_fname = op.join(base_fpath, f"sub-{subject}_ses-{session}_task-{task}_run-{run}_{MODALITY}-eve.fif")
    annotated_raw_fname = op.join(base_fpath, f"sub-{subject}_ses-{session}_task-{task}_run-{run}_{MODALITY}.fif")
    beh_fig_fname = op.join(project_root, "derivatives", "figures", f"sub-{subject}-beh-performance.png")

    if subject == "110":
        vhdr_fnames = [
            op.join(base_fpath, brainvision_basename + "_blocks1-2.vhdr"),
            op.join(base_fpath, brainvision_basename + "_blocks3-8.vhdr"),
        ]
        raw = mne.concatenate_raws([mne.io.read_raw_brainvision(f, preload=True) for f in vhdr_fnames])
    elif subject == "111":
        vhdr_fnames = [
            op.join(base_fpath, brainvision_basename + "_stimright.vhdr"),
            op.join(base_fpath, brainvision_basename + "_nostimright.vhdr"),
            op.join(base_fpath, brainvision_basename + "_nostimleft.vhdr"),
        ]
        raw = mne.concatenate_raws([mne.io.read_raw_brainvision(f, preload=True) for f in vhdr_fnames])
    else:
        vhdr_fnames = [op.join(base_fpath, brainvision_basename + ".vhdr")]
        raw = mne.io.read_raw_brainvision(vhdr_fnames[0], eog=("HEOGL", "HEOGR", "VEOGb"), preload=True)

    raw.rename_channels({k: v for k, v in EOG_RENAME.items() if k in raw.ch_names})
    raw.set_channel_types({k: v for k, v in EOG_TYPES.items() if k in raw.ch_names})
    raw.info["bads"].extend([ch for ch in ALWAYS_BAD_CHANNELS if ch in raw.ch_names])

    events, _ = mne.events_from_annotations(raw, event_id="auto")
    mapping = EVENT_DICT.copy()
    annotations_from_events = mne.annotations_from_events(
        events=events,
        event_desc=mapping,
        sfreq=raw.info["sfreq"],
        orig_time=raw.info["meas_date"],
    )
    raw.set_annotations(annotations_from_events)

    mne.write_events(events_fname, events, overwrite=True)
    raw.save(annotated_raw_fname, overwrite=True)

    _, events_id = mne.events_from_annotations(raw, event_id=EVENT_DICT)
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        datatype="eeg",
        task=task,
        run=run,
        root=bids_root,
    )
    raw.set_annotations(None)
    write_raw_bids(
        raw,
        bids_path,
        events=events_fname,
        event_id=events_id,
        overwrite=True,
        allow_preload=True,
        format="BrainVision",
    )

    fig_events = mne.viz.plot_events(
        events,
        sfreq=raw.info["sfreq"],
        first_samp=raw.first_samp,
        event_id=events_id,
        show=False,
    )
    report.add_figure(
        fig_events,
        op.join(fig_folder, f"P01_sub-{subject}_events_timeline.png"),
        "Events timeline",
        "Events read from BrainVision and written to BIDS.",
        "Quality control",
    )

    events_bids_path = bids_path.copy().update(suffix="events", extension=".tsv")
    events_file = pd.read_csv(events_bids_path, sep="\t")
    event_onsets = events_file[["onset", "value", "trial_type"]]

    durations_onset = ["cue", "catch", "stim", "dot", "response_press", "trial"]
    direction_onset = ["cue_onset", "dot_onset"]
    events_dict: Dict[str, np.ndarray] = {}

    for dur in durations_onset:
        events_dict[dur + "_onset"] = event_onsets.loc[
            event_onsets["trial_type"].str.contains(f"{dur}_onset"), "onset"
        ].to_numpy()

    for dirs in direction_onset:
        events_dict[dirs + "_right"] = event_onsets.loc[
            event_onsets["trial_type"].str.contains(f"{dirs}_right"), "onset"
        ].to_numpy()
        events_dict[dirs + "_left"] = event_onsets.loc[
            event_onsets["trial_type"].str.contains(f"{dirs}_left"), "onset"
        ].to_numpy()

    numbers_dict = {}
    for numbers in ["cue_onset_right", "cue_onset_left", "dot_onset_right", "dot_onset_left", "response_press_onset"]:
        numbers_dict[numbers] = events_dict[numbers].size

    eve_fig, ax = plt.subplots()
    bars = ax.bar(range(len(numbers_dict)), list(numbers_dict.values()))
    plt.xticks(range(len(numbers_dict)), list(numbers_dict.keys()), rotation=45)
    ax.bar_label(bars)
    report.add_figure(
        eve_fig,
        op.join(fig_folder, f"P01_sub-{subject}_event_counts.png"),
        "Number of events",
        "Total number of events.",
        "Quality control",
    )

    if subject == "115":
        add_report_text(
            report,
            "Sub-115 stimulation sequence note",
            "For sub-115, the actual stimulation sequence is stim on, stim on, stim off, stim off, which differs from the GitHub stimulation table.",
            "Quality control",
        )

    impedance_info = impedance_text(raw=raw, vhdr_path=vhdr_fnames)
    add_report_text(report, "Channel impedances", impedance_info, "Quality control")

    if op.exists(beh_fig_fname):
        report.add_image(
            beh_fig_fname,
            "Reaction time and behavioural performance",
            "Behaviour figure generated separately.",
            "Quality control",
        )
    else:
        add_report_text(report, "Behaviour figure", f"Figure not found yet: {beh_fig_fname}", "Quality control")

    add_report_text(
        report,
        "BIDS conversion",
        f"BIDS data written to: {bids_path}\nSampling frequency: {raw.info['sfreq']} Hz\nChannels marked bad at conversion: {pretty_list(raw.info['bads'])}",
        "Quality control",
    )

    return bids_path, vhdr_fnames


# -----------------------------------------------------------------------------
# Step 2: segment stim on / off.
# -----------------------------------------------------------------------------

def step_p02_segmenting_stim(
    subject: str,
    bids_path: BIDSPath,
    project_root: str,
    bids_root: str,
    report,
    fig_folder: str,
    crop_table: Dict[str, List[float]],
) -> Dict[str, str]:
    deriv_folder = op.join(bids_root, "derivatives", f"sub-{subject}")
    ensure_dir(deriv_folder)

    raw = read_raw_bids(bids_path=bids_path, verbose=True, extra_params={"preload": True})

    out_files = {}
    for label in ["no-stim", "stim"]:
        times = crop_table[label]
        pieces = [raw.copy().crop(tmin=times[0], tmax=times[1])]
        if len(times) == 4:
            pieces.append(raw.copy().crop(tmin=times[2], tmax=times[3]))
        segment = pieces[0] if len(pieces) == 1 else mne.concatenate_raws(pieces)

        fmax = min(200, segment.info["sfreq"] / 2 - 0.1)
        fig_psd = segment.compute_psd(fmin=0.1, fmax=fmax).plot(show=False)
        report.add_figure(
            fig_psd,
            op.join(fig_folder, f"P02_{label}_PSD_before_filter.png"),
            f"{label} PSD before filtering",
            f"Used to check whether a peak near 130 Hz is present. Kept ranges: {times}",
            "Stimulation segmentation",
        )

        segment.filter(l_freq=0.1, h_freq=100.0)
        output = op.join(deriv_folder, bids_path.basename + f"_{label}_raw.fif")
        segment.save(output, overwrite=True)
        add_report_text(
            report,
            f"{label} segment saved",
            f"Kept ranges: {times}\nFiltered 0.1-100 Hz\nOutput: {output}",
            "Stimulation segmentation",
        )
        out_files[label] = output

    return out_files


# -----------------------------------------------------------------------------
# Step 3: epoching, bad channels, manual rejection.
# -----------------------------------------------------------------------------

def step_p03_epoching(
    subject: str,
    bids_path: BIDSPath,
    segmented_files: Dict[str, str],
    bids_root: str,
    report,
    fig_folder: str,
) -> Dict[str, str]:
    deriv_folder = op.join(bids_root, "derivatives", f"sub-{subject}")

    segment_data = {}
    all_bad_channels = set(ALWAYS_BAD_CHANNELS)

    for label in ["no-stim", "stim"]:
        raw = mne.io.read_raw_fif(segmented_files[label], preload=True)

        reasons = get_bad_channel_reasons(raw)
        print(f"PyPREP suggested bad channels for {label}: {sorted(reasons)}")
        print(json.dumps(reasons, indent=2))

        # PSD browser for manual channel inspection.
        raw.copy().pick("eeg").compute_psd(fmin=0.1, fmax=150).plot()
        user = input("Additional bad channels, separated by spaces, or press return: ").strip().split()
        manual_reason = input("Optional manual reason for these additional channels: ").strip()

        for ch in user:
            reasons.setdefault(str(ch), []).append(manual_reason or "manually identified during QC")

        segment_bad_channels = set(str(ch) for ch in raw.info["bads"])
        segment_bad_channels.update(str(ch) for ch in reasons.keys())
        all_bad_channels.update(segment_bad_channels)

        segment_data[label] = {"raw": raw, "reasons": reasons}

    common_bads = sorted(all_bad_channels)
    out_epochs = {}

    for label in ["no-stim", "stim"]:
        raw = segment_data[label]["raw"]
        reasons = segment_data[label]["reasons"]
        bads_to_remove = [ch for ch in common_bads if ch in raw.ch_names]

        raw.info["bads"] = bads_to_remove
        raw.drop_channels(bads_to_remove)

        reason_text = "\n".join(
            f"{str(ch)}: {', '.join(map(str, reason_list))}"
            for ch, reason_list in sorted(reasons.items())
        ) or "No additional noisy channels detected."

        add_report_text(
            report,
            f"{label}: bad-channel reasons",
            f"Epochs -0.5 to 1.6 s\nReasons:\n{reason_text}",
            "Epoching and channel quality",
        )

        events, events_id = mne.events_from_annotations(raw, event_id=EVENT_DICT)
        cue_id = {k: events_id[k] for k in ["cue_onset_right", "cue_onset_left"] if k in events_id}

        epochs = mne.Epochs(
            raw,
            events,
            cue_id,
            tmin=-0.5,
            tmax=1.6,
            baseline=None,
            detrend=1,
            proj=True,
            picks="all",
            reject=None,
            reject_by_annotation=False,
            preload=True,
            event_repeated="merge",
        )

        n_fft = min(int(2 * epochs.info["sfreq"]), len(epochs.times))
        fig_psd = epochs.compute_psd(fmin=0.1, fmax=100, method="welch", n_fft=n_fft).plot(show=False)
        report.add_figure(
            fig_psd,
            op.join(fig_folder, f"P03_{label}_epoch_PSD.png"),
            f"{label}: PSD of cue epochs",
            "Epochs -0.5 to 1.6 s, cue onset = 0 s.",
            "Epoching and channel quality",
        )

        missing = [ch for ch in PICKED_POSTERIOR_CHANNELS if ch not in epochs.ch_names]
        if missing:
            raise RuntimeError(f"Posterior QC channels missing: {missing}")

        # Manual bad-epoch rejection: this opens the time series browser.
        n_before = len(epochs)
        epochs.plot(
            picks=PICKED_POSTERIOR_CHANNELS,
            n_channels=len(PICKED_POSTERIOR_CHANNELS),
            block=True,
            title=f"{label}: manually reject trials using only {PICKED_POSTERIOR_CHANNELS}",
        )
        n_after = len(epochs)

        output_fname = op.join(deriv_folder, bids_path.basename + f"_{label}_epo-cue.fif")
        epochs.save(output_fname, overwrite=True)

        add_report_text(
            report,
            f"{label}: manual posterior trial rejection",
            f"Channels shown: {pretty_list(PICKED_POSTERIOR_CHANNELS)}\nEpochs before: {n_before}\nEpochs retained: {n_after}\nEpochs rejected: {n_before - n_after}\nOutput: {output_fname}",
            "Epoching and channel quality",
        )
        out_epochs[label] = output_fname

    return out_epochs


# -----------------------------------------------------------------------------
# Step 4: ERP.
# -----------------------------------------------------------------------------

def step_a01_erp(
    subject: str,
    bids_path: BIDSPath,
    cleaned_epochs: Dict[str, str],
    project_root: str,
    report,
    fig_folder: str,
):
    evokeds = {"cue": {}, "grating": {}}

    def make_evoked(epochs, tmin, tmax, baseline, shift=None):
        evoked = epochs.average(method="mean").filter(l_freq=None, h_freq=30)
        evoked = evoked.copy().crop(tmin=tmin, tmax=tmax)
        evoked.apply_baseline(baseline)
        if shift is not None:
            evoked = evoked.copy().shift_time(shift, relative=True)
        return evoked

    def add_compare_fig(evoked_dict, fname, title, caption, picks, xlim):
        fig = mne.viz.plot_compare_evokeds(
            evoked_dict,
            picks=picks,
            combine="mean",
            show=False,
            ci=False,
            truncate_xaxis=False,
            truncate_yaxis=False,
        )
        if isinstance(fig, list):
            fig = fig[0]
        fig.axes[0].axvline(0, color="k", linestyle="--", linewidth=1)
        fig.axes[0].set_xlim(*xlim)
        report.add_figure(fig, fname, title, caption, "Evoked responses")

    deriv_folder = op.join(project_root, "data", "BIDS", "derivatives", f"sub-{subject}")
    for label in ["no-stim", "stim"]:
        epochs = mne.read_epochs(cleaned_epochs[label], preload=True)
        epochs = epochs[["cue_onset_right", "cue_onset_left"]]

        evokeds["cue"][label] = make_evoked(epochs, tmin=-0.1, tmax=0.5, baseline=(-0.1, 0))
        evokeds["grating"][label] = make_evoked(epochs, tmin=1.1, tmax=1.6, baseline=(1.1, 1.2), shift=-1.2)

        mne.write_evokeds(op.join(deriv_folder, bids_path.basename + f"_{label}_evo-cue.fif"), evokeds["cue"][label], overwrite=True)
        mne.write_evokeds(op.join(deriv_folder, bids_path.basename + f"_{label}_evo-grating.fif"), evokeds["grating"][label], overwrite=True)

    add_compare_fig(
        {"no stimulation": evokeds["cue"]["no-stim"], "stimulation": evokeds["cue"]["stim"]},
        op.join(fig_folder, "A01_stim_no_stim_evoked_cue_comparison.png"),
        "Cue-locked evoked comparison: stimulation vs no stimulation",
        "Cue onset at 0 s; window -0.1 to 0.5 s; baseline -0.1 to 0 s; three posterior channels averaged.",
        ERP_POSTERIOR_CHANNELS,
        (-0.1, 0.5),
    )

    add_compare_fig(
        {"no stimulation": evokeds["grating"]["no-stim"], "stimulation": evokeds["grating"]["stim"]},
        op.join(fig_folder, "A01_stim_no_stim_evoked_grating_comparison.png"),
        "Grating-locked evoked comparison: stimulation vs no stimulation",
        "Grating onset at 0 s; original window 1.1 to 1.6 s shifted by -1.2 s; baseline 1.1 to 1.2 s; three posterior channels averaged.",
        ERP_POSTERIOR_CHANNELS,
        (-0.1, 0.4),
    )

    fig_cue_channels, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, ch in zip(axes, ERP_POSTERIOR_CHANNELS):
        mne.viz.plot_compare_evokeds(
            {"no stimulation": evokeds["cue"]["no-stim"], "stimulation": evokeds["cue"]["stim"]},
            picks=ch,
            combine=None,
            axes=ax,
            show=False,
            ci=False,
            truncate_xaxis=False,
            truncate_yaxis=False,
        )
        ax.set_title(f"Cue-locked: {ch}")
        ax.axvline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlim(-0.1, 0.5)
    report.add_figure(
        fig_cue_channels,
        op.join(fig_folder, "A01_stim_no_stim_evoked_cue_by_channel.png"),
        "Cue-locked evoked responses by posterior channel",
        "Cue onset at 0 s; window -0.1 to 0.5 s; baseline -0.1 to 0 s; stimulation and no stimulation compared separately for each channel.",
        "Evoked responses",
    )

    fig_grating_channels, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, ch in zip(axes, ERP_POSTERIOR_CHANNELS):
        mne.viz.plot_compare_evokeds(
            {"no stimulation": evokeds["grating"]["no-stim"], "stimulation": evokeds["grating"]["stim"]},
            picks=ch,
            combine=None,
            axes=ax,
            show=False,
            ci=False,
            truncate_xaxis=False,
            truncate_yaxis=False,
        )
        ax.set_title(f"Grating-locked: {ch}")
        ax.axvline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlim(-0.1, 0.4)
    report.add_figure(
        fig_grating_channels,
        op.join(fig_folder, "A01_stim_no_stim_evoked_grating_by_channel.png"),
        "Grating-locked evoked responses by posterior channel",
        "Grating onset at 0 s; original window 1.1 to 1.6 s shifted by -1.2 s; baseline 1.1 to 1.2 s; stimulation and no stimulation compared separately for each channel.",
        "Evoked responses",
    )


# -----------------------------------------------------------------------------
# Step 5: TFR.
# -----------------------------------------------------------------------------

def step_a02_tfr(
    subject: str,
    bids_path: BIDSPath,
    cleaned_epochs: Dict[str, str],
    project_root: str,
    report,
    fig_folder: str,
):
    deriv_folder = op.join(project_root, "data", "BIDS", "derivatives", f"sub-{subject}")
    freqs = np.arange(2, 31, 1)
    n_cycles = freqs / 2
    baseline = (-0.3, -0.1)

    tfrs_raw = {}

    for label in ["no-stim", "stim"]:
        epochs = mne.read_epochs(cleaned_epochs[label], preload=True)
        missing = [ch for ch in PICKED_POSTERIOR_CHANNELS if ch not in epochs.ch_names]
        if missing:
            raise RuntimeError(f"Missing posterior channels: {missing}")
        epochs = epochs[["cue_onset_right", "cue_onset_left"]].copy().pick(PICKED_POSTERIOR_CHANNELS)

        tfr_raw = epochs.compute_tfr(
            method="multitaper",
            freqs=freqs,
            n_cycles=n_cycles,
            time_bandwidth=2.0,
            use_fft=True,
            return_itc=False,
            average=True,
            decim=2,
            n_jobs=4,
        )
        tfrs_raw[label] = tfr_raw

        tfr_plot = tfr_raw.copy()
        tfr_plot.apply_baseline(baseline=baseline, mode="percent")

        out = op.join(deriv_folder, bids_path.basename + f"_both_{label}_tfr.h5")
        tfr_raw.save(out, overwrite=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for ax, ch in zip(axes, PICKED_POSTERIOR_CHANNELS):
            tfr_plot.plot(picks=ch, tmin=-0.3, tmax=1.4, baseline=None, mode=None, axes=ax, show=False, colorbar=True)
            ax.set_title(f"{label}: {ch}")
        report.add_figure(
            fig,
            op.join(fig_folder, f"A02_{label}_three_channel_TFR.png"),
            f"{label}: combined attention-left/right TFR",
            f"Three posterior channels only; percent baseline {baseline}.",
            "Time-frequency analysis",
        )

    # Difference and ratio are computed from the raw TFR data without baseline correction.
    difference = tfrs_raw["stim"].copy()
    difference.data = tfrs_raw["stim"].data - tfrs_raw["no-stim"].data

    fig_diff, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, ch in zip(axes, PICKED_POSTERIOR_CHANNELS):
        difference.plot(picks=ch, tmin=-0.3, tmax=1.4, baseline=None, mode=None, axes=ax, show=False, colorbar=True)
        ax.set_title(f"Stim - no-stim: {ch}")
    report.add_figure(
        fig_diff,
        op.join(fig_folder, "A02_stim_minus_no_stim_TFR.png"),
        "TFR difference: stimulation minus no stimulation",
        "Difference computed from unbaselined TFR data and shown separately for each posterior channel.",
        "Time-frequency analysis",
    )

    ratio = tfrs_raw["stim"].copy()
    denom = tfrs_raw["stim"].data + tfrs_raw["no-stim"].data
    eps = np.finfo(float).eps
    ratio.data = (tfrs_raw["stim"].data - tfrs_raw["no-stim"].data) / (denom + eps)

    fig_ratio, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, ch in zip(axes, PICKED_POSTERIOR_CHANNELS):
        ratio.plot(picks=ch, tmin=-0.3, tmax=1.4, baseline=None, mode=None, axes=ax, show=False, colorbar=True)
        ax.set_title(f"Ratio: {ch}")
    report.add_figure(
        fig_ratio,
        op.join(fig_folder, "A02_stim_ratio_no_stim_TFR.png"),
        "TFR ratio: (stimulation - no stimulation) / (stimulation + no stimulation)",
        "Ratio computed from unbaselined TFR data and shown separately for each posterior channel.",
        "Time-frequency analysis",
    )


# -----------------------------------------------------------------------------
# Main entry point.
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Single-subject STN EEG pipeline")
    parser.add_argument("--subject", required=True, help="Participant label without the sub- prefix, e.g. 115")
    parser.add_argument("--session", default=SESSION_DEFAULT)
    parser.add_argument("--task", default=TASK_DEFAULT)
    parser.add_argument("--run", default=RUN_DEFAULT)
    parser.add_argument("--project-root", default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--github-root", default=GITHUB_ROOT_DEFAULT)
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--bids-root", default=BIDS_ROOT_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ParticipantPDF, impedance_text = _load_report_helper(args.github_root)

    fig_folder = ensure_dir(op.join(args.project_root, "derivatives", "figures", f"sub-{args.subject}"))
    report_folder = ensure_dir(op.join(args.project_root, "derivatives", "reports", f"sub-{args.subject}"))
    report = ParticipantPDF(report_folder, args.subject)

    report.add_text(
        "Pipeline summary",
        "This participant report was generated from BrainVision raw data, converted to BIDS, segmented into stimulation on and stimulation off, epoched around cue onset, manually cleaned using the epoch browser, and then analyzed at the sensor level with ERP and TFR.",
        "Quality control",
    )

    bids_path, vhdr_fnames = step_p01_bids_conversion(
        subject=args.subject,
        session=args.session,
        task=args.task,
        run=args.run,
        project_root=args.project_root,
        data_root=args.data_root,
        bids_root=args.bids_root,
        report=report,
        fig_folder=fig_folder,
    )

    subject_key = f"sub-{args.subject}"
    if subject_key not in STIMULATION_CROPPED_TIME:
        raise KeyError(f"No stimulation crop table found for {subject_key}")

    crop_table = prompt_for_crop_table(
        subject=args.subject,
        bids_path=bids_path,
        default_crop_table=STIMULATION_CROPPED_TIME[subject_key],
        bids_root=args.bids_root,
        report=report,
        fig_folder=fig_folder,
    )

    segmented_files = step_p02_segmenting_stim(
        subject=args.subject,
        bids_path=bids_path,
        project_root=args.project_root,
        bids_root=args.bids_root,
        report=report,
        fig_folder=fig_folder,
        crop_table=crop_table,
    )

    cleaned_epochs = step_p03_epoching(
        subject=args.subject,
        bids_path=bids_path,
        segmented_files=segmented_files,
        bids_root=args.bids_root,
        report=report,
        fig_folder=fig_folder,
    )

    step_a01_erp(
        subject=args.subject,
        bids_path=bids_path,
        cleaned_epochs=cleaned_epochs,
        project_root=args.project_root,
        report=report,
        fig_folder=fig_folder,
    )

    step_a02_tfr(
        subject=args.subject,
        bids_path=bids_path,
        cleaned_epochs=cleaned_epochs,
        project_root=args.project_root,
        report=report,
        fig_folder=fig_folder,
    )

    add_report_text(
        report,
        "Impedance summary",
        impedance_text(raw=mne.io.read_raw_brainvision(vhdr_fnames[0], preload=False), vhdr_path=vhdr_fnames),
        "Quality control",
    )

    print(f"Updated PDF: {report.pdf_fname}")


if __name__ == "__main__":
    main()
