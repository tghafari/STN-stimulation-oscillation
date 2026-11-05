"""Shared project paths, reference settings and EEG derivative filenames.

The preprocessing pipeline writes reference-labelled FIF files. P01 uses
`desc-noref` for unreferenced EEG and `desc-avgref` for average reference.
Later processing stages retain this reference label in their filenames.

The saved P01 QC JSON records the chosen rereferencing method. Command-line
`--rereference` takes precedence when supplied, matching the existing
pipeline behaviour. Mac is the default project platform.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mne_bids import BIDSPath

BLUEBEAR_PROJECT_ROOT = Path(
    "/rds/projects/j/jenseno-avtemporal-attention/"
    "Projects/subcortical-structures/STN-in-PD"
)
MAC_PROJECT_ROOT = Path(
    "/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/"
    "BEAR_outage/STN-in-PD"
)

CONDITIONS = ("no-stim", "stim")
POSTERIOR_SCREEN_CHANNELS = ("PO3", "PO4", "POz")


def resolve_project_root(platform="mac", project_root=None):
    """Resolve an explicit project directory or a known platform default."""
    if project_root:
        return Path(project_root).expanduser().resolve()
    if platform == "bluebear":
        return BLUEBEAR_PROJECT_ROOT
    if platform == "mac":
        return MAC_PROJECT_ROOT
    raise ValueError("platform must be 'mac' or 'bluebear'")


def bids_root(root):
    """Return the BIDS dataset root under the project directory."""
    return root / "data" / "BIDS"


def subject_deriv_dir(root, subject):
    """Create and return the participant's BIDS derivatives directory."""
    output = bids_root(root) / "derivatives" / f"sub-{subject}"
    output.mkdir(parents=True, exist_ok=True)
    return output


def qc_dir(root, subject):
    """Create and return the participant's all-channel QC directory."""
    output = subject_deriv_dir(root, subject) / "qc_all_channels"
    output.mkdir(parents=True, exist_ok=True)
    return output


def base_bids_path(root, subject, session, task, run):
    """Construct the MNE-BIDS path for one EEG recording."""
    return BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        root=bids_root(root),
        datatype="eeg",
        suffix="eeg",
    )


def reference_desc(rereference="avg"):
    """Translate a rereference setting into the saved filename label."""
    return "noref" if rereference in (None, "none") else "avgref"


def participant_rereference(root, subject):
    """Find the reference setting from CLI or the participant's P01 QC JSON.

    The CLI takes precedence, followed by the existing audit file. If
    neither specifies a reference, preserve the pipeline's default
    average-reference setting.
    """
    arguments = sys.argv[1:]
    if "--rereference" in arguments:
        index = arguments.index("--rereference")
        value = (
            arguments[index + 1]
            if index + 1 < len(arguments)
            else "avg"
        )
        return None if value == "none" else "avg"

    audit = qc_dir(root, subject) / "P01_pyprep_segment_epoch_reref.json"
    if audit.exists():
        try:
            return json.loads(
                audit.read_text(encoding="utf-8")
            ).get("rereference")
        except Exception:
            # Preserve the historical fallback when an audit is unreadable.
            pass
    return "avg"


def reference_state_desc(root, subject):
    """Return the reference label currently used for participant files."""
    return reference_desc(participant_rereference(root, subject))


def stage_path(root, subject, session, task, run, condition, desc, kind):
    """Return the expected FIF path for a particular processing stage.

    For EEG-derived rereference, ICA, cleaning and ERP stages, prepend
    the reference label. Other stage descriptions are left unchanged.
    """
    bids_path = base_bids_path(root, subject, session, task, run)
    if desc in {"reref", "ica", "clean", "erp"}:
        reference = reference_state_desc(root, subject)
        desc = reference if desc == "reref" else f"{reference}-{desc}"
    filename = f"{bids_path.basename}_{condition}_desc-{desc}_{kind}.fif"
    return subject_deriv_dir(root, subject) / filename


def crop_table_path():
    """Return the shared, participant-specific stimulation crop-time JSON."""
    return (
        Path(__file__).resolve().parents[1]
        / "stimulation_cropped_time.json"
    )
