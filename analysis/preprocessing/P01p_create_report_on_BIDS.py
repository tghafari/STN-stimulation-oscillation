"""
===============================================
01p_create report from bids
    This code will create a report from the BIDS raw
    instead of converting to BIDS from raw and then
    create a report.
    
    This is used for those files that are already
    in BIDS format but don't yet have a report.

written by Tara Ghafari
t.ghafari@bham.ac.uk
==============================================  
"""


import os.path as op
import os
import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt

# BIDS settings: fill these out 
subject = '115'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
deriv_suffix = 'ann'
extension = '.fif'

platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
muscle_reject = False  # rejecting muscle artefact?

if platform == 'bluebear':
    rds_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    camcan_dir = '/rds/projects/q/quinna-camcan/dataman/data_information'
elif platform == 'mac':
    rds_dir = '/Volumes/jenseno-avtemporal-attention'
    camcan_dir = '/Volumes/quinna-camcan/dataman/data_information'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
bids_root = op.join(project_root, 'data', 'BIDS')

# for bear outage
bids_root = '/Users/taraghafari/Desktop/BEAR_outage/STN-in-PD/data/BIDS'

# Specify specific file names
bids_path = BIDSPath(subject=subject, 
                     session=session,
                     task=task, 
                     run=run, 
                     root=bids_root, 
                     datatype ='eeg', 
                     suffix=eeg_suffix)

deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results
if not op.exists(deriv_folder):
    os.makedirs(deriv_folder)
deriv_fname = op.join(deriv_folder, bids_path.basename + '_' + deriv_suffix + extension)  # prone to change if annotation worked for eeg brainvision

# Read the already-existing BIDS raw
raw = read_raw_bids(bids_path=bids_path, 
                    verbose=False, 
                    extra_params={'preload':True})
raw.load_data()

# Work on a copy for any modifications
raw_report = raw.copy()

# Get events from the BIDS raw annotations
events, _ = mne.events_from_annotations(raw_report, event_id='auto')

mapping = {
    1: 'cue_onset_right',
    2: 'cue_onset_left',
    3: 'trial_onset',
    4: 'stim_onset',
    5: 'catch_onset',
    6: 'dot_onset_right',
    7: 'dot_onset_left',
    8: 'response_press_onset',
    20: 'block_onset',
    21: 'block_end',
    30: 'experiment_end',
    99999: 'new_stim_segment',
}

annotations_from_events = mne.annotations_from_events(
    events=events,
    event_desc=mapping,
    sfreq=raw_report.info["sfreq"],
    orig_time=raw_report.info["meas_date"],
)
raw_report.set_annotations(annotations_from_events)

event_dict = {
    'cue_onset_right': 1,
    'cue_onset_left': 2,
    'trial_onset': 3,
    'stim_onset': 4,
    'catch_onset': 5,
    'dot_onset_right': 6,
    'dot_onset_left': 7,
    'response_press_onset': 8,
    'block_onset': 20,
    'block_end': 21,
    'experiment_end': 30,
    'new_stim_segment': 99999,
}
_, events_id = mne.events_from_annotations(raw_report, event_id=event_dict)

# Read the existing BIDS events.tsv
events_bids_path = bids_path.copy().update(suffix='events', extension='.tsv')
events_file = pd.read_csv(events_bids_path, sep='\t')
event_onsets = events_file[['onset', 'value', 'trial_type']]

numbers_dict = {}
for key in ['cue_onset_right', 'cue_onset_left', 'dot_onset_right', 'dot_onset_left', 'response_press_onset']:
    numbers_dict[key] = event_onsets.loc[
        event_onsets['trial_type'].str.contains(key),
        'onset'
    ].to_numpy().size

eve_fig, ax = plt.subplots()
bars = ax.bar(range(len(numbers_dict)), list(numbers_dict.values()))
plt.xticks(range(len(numbers_dict)), list(numbers_dict.keys()), rotation=45)
ax.bar_label(bars)
plt.show()

# Build report from copies only
report_root = op.join(project_root, 'derivatives/reports')
report_folder = op.join(report_root, f'sub-{subject}')
os.makedirs(report_folder, exist_ok=True)

report_fname = op.join(report_folder, f'sub-{subject}_report.hdf5')
html_report_fname = op.join(report_folder, f'sub-{subject}_report.html')

report = mne.Report(title=f'Subject {subject}')
report.add_figure(eve_fig,
                  title='Number of events',
                  caption='number of events in total',
                  tags=('eve'))
report.add_events(events=events,
                  event_id=events_id,
                  tags=('eve'),
                  title='events from annotations',
                  sfreq=raw_report.info['sfreq'])

raw_for_report = raw_report.copy()
raw_for_report.filter(0.1, 100)
report.add_raw(raw=raw_for_report,
               title='raw not referenced with bad channels',
               psd=True,
               butterfly=False,
               tags=('raw'))

report.save(report_fname, overwrite=False)
report.save(html_report_fname, overwrite=False, open_browser=True)