"""Shared paths and filenames for the all-channel EEG preprocessing pipeline."""
from __future__ import annotations
from pathlib import Path
from mne_bids import BIDSPath
BLUEBEAR_PROJECT_ROOT=Path('/rds/projects/j/jenseno-avtemporal-attention/Projects/subcortical-structures/STN-in-PD')
MAC_PROJECT_ROOT=Path('/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/STN-in-PD')
CONDITIONS=('no-stim','stim'); POSTERIOR_SCREEN_CHANNELS=('PO3','PO4','POz')
def resolve_project_root(platform,project_root=None):
    if project_root:return Path(project_root).expanduser().resolve()
    if platform=='bluebear':return BLUEBEAR_PROJECT_ROOT
    if platform=='mac':return MAC_PROJECT_ROOT
    raise ValueError("platform must be 'mac' or 'bluebear'")
def bids_root(root):return root/'data'/'BIDS'
def subject_deriv_dir(root,subject):
    out=bids_root(root)/'derivatives'/f'sub-{subject}'; out.mkdir(parents=True,exist_ok=True); return out
def qc_dir(root,subject):
    out=subject_deriv_dir(root,subject)/'qc_all_channels'; out.mkdir(parents=True,exist_ok=True); return out
def base_bids_path(root,subject,session,task,run):return BIDSPath(subject=subject,session=session,task=task,run=run,root=bids_root(root),datatype='eeg',suffix='eeg')
def stage_path(root,subject,session,task,run,condition,desc,kind):
    bp=base_bids_path(root,subject,session,task,run); return subject_deriv_dir(root,subject)/f'{bp.basename}_{condition}_desc-{desc}_{kind}.fif'
def crop_table_path():return Path(__file__).resolve().parents[1]/'stimulation_cropped_time.json'
