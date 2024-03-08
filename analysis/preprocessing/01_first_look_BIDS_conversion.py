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
import pandas as pd

import mne
from mne_bids import (BIDSPath, write_raw_bids, read_raw_bids)
import matplotlib.pyplot as plt

# fill these out
subj_code = '01_ly'  # subject code assigned to by Benchi's group
subj_name = 'Liuyu'  # name on eeg file
platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
pilot = True  # is it pilot data or real data?
rprt = True

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

# BIDS events
events_suffix = 'events'  
events_extension = '.tsv'

# Read raw file in BrainVision (.vhdr, .vmrk, .eeg) format
raw = mne.io.read_raw_brainvision(vhdr_fname, eog=('HEOGL', 'HEOGR', 'VEOGb'), preload=False)

# Plot raw to take a look
raw.plot(title="raw") 

# Read events from raw object
events, events_id = mne.events_from_annotations(raw, event_id='auto')
mne.write_events(events_fname, events, overwrite=True)

# Convert to BIDS
bids_path = BIDSPath(subject=subject, session=session, datatype ='eeg',
                     task=task, run=run, root=bids_root)
write_raw_bids(raw, bids_path, events_data=events_fname, 
               event_id=events_id, overwrite=True)

# Plot triggers from bids .tsv file
events_bids_path = bids_path.copy().update(suffix=events_suffix,
                                            extension=events_extension)
events_file = pd.read_csv(events_bids_path, sep='\t')
event_onsets = events_file[['onset', 'value', 'trial_type']]

# Plot all events
event_onsets.plot(kind='scatter', x='onset', y='trial_type')
plt.xlabel('onset(sec)')
plt.ylabel('event type')
plt.show()

if rprt:
    report_root = op.join(project_root, r'results/reports')  # RDS folder for reports
    if not op.exists(op.join(report_root , 'sub-' + subject)):
        os.makedirs(op.join(report_root , 'sub-' + subject))
    report_folder = op.join(report_root , 'sub-' + subject)
    report_fname = op.join(report_folder, f'sub-{subject}_events.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub_{subject}_events.html')

    report = mne.Report(title=f'sub-{subject}')
    report.add_events(events=events, title='events from "events"', sfreq=raw.info['sfreq'])
    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
        

""" this section needs to be edited with new stim names from Benchi
# Check durations using triggers
durations_onset = ['cue', 'catch', 'stim', 'dot', 'response_press']
durations_offset = ['cue'] #, 'stim', 'dot']  # stim and dot are removed in actual data collection
direction_onset = ['cue_onset', 'dot_onset']
events_dict = {}

for dur in durations_onset:    
    events_dict[dur + "_onset"] = event_onsets.loc[event_onsets['trial_type'].str.contains(f'{dur}_onset'),
                                                'onset'].to_numpy()
for dur in durations_offset:   
    events_dict[dur + "_offset"] = event_onsets.loc[event_onsets['trial_type'].str.contains(f'{dur}_offset'),
                                                'onset'].to_numpy()
for dirs in direction_onset:
    events_dict[dirs + "_right"] = event_onsets.loc[event_onsets['trial_type'].str.contains(f'{dirs}_right'),
                                                'onset'].to_numpy()
    events_dict[dirs + "_left"] = event_onsets.loc[event_onsets['trial_type'].str.contains(f'{dirs}_left'),
                                                'onset'].to_numpy()

# compare number of trials with stimuli and responses
numbers_dict = {}
for numbers in  ['cue_onset_right', 'cue_onset_left', 'dot_onset_right', 'dot_onset_left', 
                        'response_press_onset']:
    numbers_dict[numbers] = events_dict[numbers].size
    
fig, ax = plt.subplots()
bars = ax.bar(range(len(numbers_dict)), list(numbers_dict.values()))
plt.xticks(range(len(numbers_dict)), list(numbers_dict.keys()), rotation=45)
ax.bar_label(bars)
plt.show()

# Check duration of cue presentation  
events_dict['cue_duration'] = events_dict['cue_offset'] - events_dict['cue_onset']
events_dict['stim_to_dot_duration'] = events_dict['dot_onset'] - events_dict['stim_onset']
events_dict['RT'] = events_dict['response_press_onset'] - events_dict['dot_onset'] 
# cheat line to add extra elements for easier calculation of RTs:
    #events_dict['dot_onset'] = np.insert(events_dict['dot_onset'], index_onset_should_be_added_before, onset)

# Plot all durations
for dur in ['cue_duration', 'stim_to_dot_duration', 'RT']:
    fig, ax = plt.subplots()
    plt.hist(events_dict[dur])
    plt.title(dur)
    plt.xlabel('time in sec')
    plt.ylabel('number of events')
    plt.show()
"""

