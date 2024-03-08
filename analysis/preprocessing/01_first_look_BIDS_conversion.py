"""
===============================================
01_first_look_and_BIDS_conversion
    in this code opens .eeg file
    and plot raw eeg data.
    then converts raw objevt to bids
    format

written by Tara Ghafari
t.ghafari@bham.ac.uk
==============================================  
"""

# Import relevant Python modules
import os.path as op
import os

import mne
from mne_bids import (BIDSPath, write_raw_bids, read_raw_bids)
import matplotlib.pyplot as plt

# fill these out
subj_code = '01_ly'  # subject code assigned to by Benchi's group
subj_name = 'Liuyu'  # name on eeg file
platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
pilot = True  # is it pilot data or real data?

if platform == 'bluebear':
    rds_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/jenseno-avtemporal-attention'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
if pilot:
    data_root = op.join(project_root, 'Data/pilot-data/AO')
else:
    data_root = op.join(project_root, 'Data/real-data')

base_fname = op.join(data_root, subj_code, f'{subj_code}_EEG', f'{subj_name}_ao1_new')
eeg_fname = base_fname + '.eeg'
vhdr_fname = base_fname + '.vhdr'
events_fname = base_fname + '-eve.fif'

# BIDS settings
bids_root = op.join(project_root, 'Data', 'BIDS')
subject = '01'
session = '01'
task = 'SpAtt'
run = '01'

# Read raw file in BrainVision (.vhdr, .vmrk, .eeg) format
raw = mne.io.read_raw_brainvision(vhdr_fname, preload=False)

# Plot raw to take a look
raw.plot(title="raw") 

# Read events from raw object
events, events_id = mne.events_from_annotations(raw, event_id='auto')
mne.write_events(events_fname, events, overwrite=True)

# Convert to BIDS
bids_path = BIDSPath(subject=subject, session=session,
                     task=task, run=run, root=bids_root)
write_raw_bids(raw, bids_path, events_data=events_fname, 
               event_id=events_id, overwrite=True)