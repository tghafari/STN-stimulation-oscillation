# -*- coding: utf-8 -*-
"""
Regenerate group-analysis epochs for sub-119.

POz was removed from the original saved subject-level epochs, so it
cannot be interpolated from those files. This script goes back to the
already-segmented raw data, recreates the cue epochs, interpolates POz,
and saves group-analysis-ready epochs.

No P01 or P02 rerun is required.
"""

import json
import os
import os.path as op

import mne
from mne_bids import BIDSPath


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

subject = "119"
session = "01"
task = "SpAtt"
run = "01"

project_root = (
    "/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/"
    "BEAR_outage/STN-in-PD"
)

bids_root = op.join(project_root, "data", "BIDS")

deriv_folder = op.join(
    bids_root,
    "derivatives",
    f"sub-{subject}",
)

bids_path = BIDSPath(
    subject=subject,
    session=session,
    task=task,
    run=run,
    root=bids_root,
    datatype="eeg",
    suffix="eeg",
)

posterior_channels = ["PO3", "PO4", "POz"]

event_dict = {
    "cue_onset_right": 1,
    "cue_onset_left": 2,
}


# ------------------------------------------------------------
# Process both stimulation conditions
# ------------------------------------------------------------

for label in ["no-stim", "stim"]:

    raw_fname = op.join(
        deriv_folder,
        bids_path.basename + f"_{label}_raw.fif",
    )

    print(f"\nReading segmented raw data:")
    print(raw_fname)

    raw = mne.io.read_raw_fif(
        raw_fname,
        preload=True,
    )

    # --------------------------------------------------------
    # Check that POz is available in the segmented raw data
    # --------------------------------------------------------

    missing = [
        ch
        for ch in posterior_channels
        if ch not in raw.ch_names
    ]

    if missing:
        raise RuntimeError(
            f"These posterior channels are missing from "
            f"the segmented raw data for sub-{subject} {label}: "
            f"{missing}"
        )

    print(
        f"{label}: posterior channels available: "
        f"{posterior_channels}"
    )

    # --------------------------------------------------------
    # Make sure standard-1020 positions are available
    # --------------------------------------------------------

    if raw.get_montage() is None:

        raw.set_montage(
            "standard_1020",
            on_missing="warn",
        )

    # --------------------------------------------------------
    # Mark POz as bad for interpolation
    # --------------------------------------------------------

    raw.info["bads"] = ["POz"]

    # --------------------------------------------------------
    # Create cue epochs
    # --------------------------------------------------------

    events, events_id = mne.events_from_annotations(
        raw,
        event_id=event_dict,
    )

    cue_id = {
        key: events_id[key]
        for key in [
            "cue_onset_right",
            "cue_onset_left",
        ]
        if key in events_id
    }

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

    print(
        f"{label}: created {len(epochs)} epochs."
    )

    # --------------------------------------------------------
    # Interpolate POz
    # --------------------------------------------------------

    epochs.interpolate_bads(
        reset_bads=True,
        mode="accurate",
    )

    print(
        f"{label}: POz interpolated."
    )

    # --------------------------------------------------------
    # Check that all three posterior channels now exist
    # --------------------------------------------------------

    missing_after = [
        ch
        for ch in posterior_channels
        if ch not in epochs.ch_names
    ]

    if missing_after:
        raise RuntimeError(
            f"Posterior channels still missing after interpolation: "
            f"{missing_after}"
        )

    # --------------------------------------------------------
    # Save group-analysis epochs
    # --------------------------------------------------------

    output_fname = op.join(
        deriv_folder,
        bids_path.basename
        + f"_{label}_epo-cue-group.fif",
    )

    epochs.save(
        output_fname,
        overwrite=True,
    )

    print(
        f"Saved:\n{output_fname}"
    )


# ------------------------------------------------------------
# Save interpolation record
# ------------------------------------------------------------

qc_folder = op.join(
    deriv_folder,
    "qc",
)

os.makedirs(
    qc_folder,
    exist_ok=True,
)

interpolation_fname = op.join(
    qc_folder,
    f"sub-{subject}_group_interpolation.json",
)

interpolation_summary = {
    "subject": subject,
    "interpolated_channels": ["POz"],
}

with open(
    interpolation_fname,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        interpolation_summary,
        f,
        indent=2,
    )

print(
    f"\nSaved interpolation record:\n"
    f"{interpolation_fname}"
)

print("\nFinished.")