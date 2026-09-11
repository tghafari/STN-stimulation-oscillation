"""Shared paths and filenames for the all-channel EEG preprocessing pipeline.

Reference-aware naming
----------------------
P01 epoch files explicitly encode the reference state:
  desc-noref_epo.fif   : no rereferencing
  desc-avgref_epo.fif  : average EEG reference
Later stages preserve it, e.g. desc-noref-ica_epo.fif and desc-avgref-clean_epo.fif.
Average reference is the pipeline default; --rereference none disables it in P01.
Downstream scripts read the reference state from the P01 audit JSON.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
from mne_bids import BIDSPath
BLUEBEAR_PROJECT_ROOT=Path('/rds/projects/j/jenseno-avtemporal-attention/Projects/subcortical-structures/STN-in-PD')
MAC_PROJECT_ROOT=Path('/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/STN-in-PD')
CONDITIONS=('no-stim','stim');POSTERIOR_SCREEN_CHANNELS=('PO3','PO4','POz')
def resolve_project_root(platform='mac',project_root=None):
 if project_root:return Path(project_root).expanduser().resolve()
 if platform=='bluebear':return BLUEBEAR_PROJECT_ROOT
 if platform=='mac':return MAC_PROJECT_ROOT
 raise ValueError("platform must be 'mac' or 'bluebear'")
def bids_root(root):return root/'data'/'BIDS'
def subject_deriv_dir(root,subject):
 out=bids_root(root)/'derivatives'/f'sub-{subject}';out.mkdir(parents=True,exist_ok=True);return out
def qc_dir(root,subject):
 out=subject_deriv_dir(root,subject)/'qc_all_channels';out.mkdir(parents=True,exist_ok=True);return out
def base_bids_path(root,subject,session,task,run):return BIDSPath(subject=subject,session=session,task=task,run=run,root=bids_root(root),datatype='eeg',suffix='eeg')
def reference_desc(rereference='avg'):return 'noref' if rereference in (None,'none') else 'avgref'
def participant_rereference(root,subject):
 """Return 'avg' or None from current P01 command line or the saved P01 audit."""
 argv=sys.argv[1:]
 if '--rereference' in argv:
  i=argv.index('--rereference');value=argv[i+1] if i+1<len(argv) else 'avg';return None if value=='none' else 'avg'
 audit=qc_dir(root,subject)/'P01_pyprep_segment_epoch_reref.json'
 if audit.exists():
  try:return json.loads(audit.read_text(encoding='utf-8')).get('rereference')
  except Exception:pass
 # Before P01 has written its audit, the pipeline default is average reference.
 return 'avg'
def reference_state_desc(root,subject):return reference_desc(participant_rereference(root,subject))
def stage_path(root,subject,session,task,run,condition,desc,kind):
 """Return a stage file path, preserving EEG reference state in EEG-derived files."""
 bp=base_bids_path(root,subject,session,task,run)
 if desc in {'reref','ica','clean','erp'}:
  ref=reference_state_desc(root,subject);desc=ref if desc=='reref' else f'{ref}-{desc}'
 return subject_deriv_dir(root,subject)/f'{bp.basename}_{condition}_desc-{desc}_{kind}.fif'
def crop_table_path():return Path(__file__).resolve().parents[1]/'stimulation_cropped_time.json'
