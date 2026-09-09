"""Shared configuration helpers for the all-channel EEG preprocessing pipeline.

The purpose of this module is to keep paths and filenames in one visible place.
Nothing here performs preprocessing.
"""
from __future__ import annotations

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


def resolve_project_root(platform: str, project_root: str | None = None) -> Path:
    """Return the project root, preferring an explicit command-line path."""
    if project_root:
        return Path(project_root).expanduser().resolve()
    if platform == "bluebear":
        return BLUEBEAR_PROJECT_ROOT
    if platform == "mac":
        return MAC_PROJECT_ROOT
    raise ValueError("platform must be 'mac' or 'bluebear'")


def bids_root(project_root: Path) -> Path:
    return project_root / "data" / "BIDS"


def subject_deriv_dir(project_root: Path, subject: str) -> Path:
    out = bids_root(project_root) / "derivatives" / f"sub-{subject}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def qc_dir(project_root: Path, subject: str) -> Path:
    out = subject_deriv_dir(project_root, subject) / "qc_all_channels"
    out.mkdir(parents=True, exist_ok=True)
    return out


def base_bids_path(project_root: Path, subject: str, session: str, task: str, run: str) -> BIDSPath:
    """BIDS path used only to generate consistent filenames."""
    return BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        root=bids_root(project_root),
        datatype="eeg",
        suffix="eeg",
    )


def segmented_raw_path(project_root: Path, subject: str, session: str, task: str, run: str, condition: str) -> Path:
    """Input produced by the existing stimulation-segmentation script P02."""
    bp = base_bids_path(project_root, subject, session, task, run)
    return subject_deriv_dir(project_root, subject) / f"{bp.basename}_{condition}_raw.fif"


def stage_path(project_root: Path, subject: str, session: str, task: str, run: str, condition: str, desc: str, kind: str) -> Path:
    """Create an explicit derivative filename for an intermediate stage."""
    bp = base_bids_path(project_root, subject, session, task, run)
    return subject_deriv_dir(project_root, subject) / f"{bp.basename}_{condition}_desc-{desc}_{kind}.fif"
