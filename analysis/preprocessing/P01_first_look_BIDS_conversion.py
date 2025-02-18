"""
===============================================
01_first_look_and_BIDS_conversion
    1. this code opens .eeg file
    and plot raw eeg data.
    2. the user then rejects bad channels from
    raw.plot() and the psd plots
    3. reads the events from annotations of 
    brainvision data.
    4. corrects the annotation and event_ids 
    5. adds annotations to the raw 
    7. saves as .fif
    8. then converts the raw data to bids
    9. It also plots triggers and RT to quality
    check the data.

    note that this code will save two eeg files,
    one .fif in the original folder and one .fif 
    bids in the bids folder.

written by Tara Ghafari
t.ghafari@bham.ac.uk
==============================================  
"""

import os.path as op
import os
import pandas as pd

import mne
from mne_bids import (BIDSPath, write_raw_bids, read_raw_bids)
import matplotlib.pyplot as plt
from copy import deepcopy

# # Fill these out - for older subjects- before 105
# subj_code = 'sub05'  # subject code assigned to by Benchi's group- only for subjects before 5 (inc)
# base_fname = '5.15_5_AO'  # the name of the eeg file for 01_ly to sub05 should be manually copied here

# Stimulation sequence
"""copy the stim sequence for each participant from here: 
https://github.com/tghafari/STN-stimulation-oscillation/wiki/Stimulation-table"""
stim_sequence = {'sub-01':["no_stim-left rec", "no_stim-right rec", "Right stim- no rec", "Left stim- no rec"],  # stimulation on STN
                 'sub-02':["no_stim-left rec", "no_stim-right rec", "Left stim- no rec", "Right stim- no rec"],  # stimulation on STN
                 'sub-05':["Left stim- no rec", "Right stim- no rec", "no_stim-left rec", "no_stim-right rec"],  # stimulation on STN
                 'sub-107':["no_stim-right rec", "no_stim-left rec", "Right stim- no rec", "Left stim- no rec"],  # stimulation on STN
                 'sub-108':["no_stim-right rec", "no_stim-left rec", "left stim- no rec", "right stim- no rec"],  # stimulation on STN
                 'sub-110': ["Right stim- no rec", "Left stim- no rec", "no_stim-no rec", "no_stim-no rec"],  # no LFP recording, stimulation on VLM
                 'sub-102': ["no_stim-left rec", "no_stim-right rec", "Left stim- no rec", "Right stim- no rec"],
                 'sub-101': ["no_stim-left rec", "no_stim-right-rec", "Right stim- no rec", "Left stim- no rec"],
                 'sub-111': ["Left stim- no rec", "Right stim- no rec", "no_stim-right rec", "no_stim-left rec"],
                 'sub-112': ["Left stim- no rec", "no_stim-right rec", "no_stim-left rec", "Right stim- no rec"],
                 'sub-103': ["Right stim- no rec", "no_stim-left rec", "no_stim-right rec", "Left stim- no rec"],
                 } 
# BIDS settings
subject = '102'
brainVision_basename = f'{subject[-2:]}_ao'  # might need modification per subject

session = '01'
task = 'SpAtt'
run = '01'
modality = 'eeg'
extension = '.fif'

platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
sanity_test = False
eve_rprt = True
summary_rprt = True

if platform == 'bluebear':
    rds_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/jenseno-avtemporal-attention'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
# '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD'  # only for bear outage time
data_root = op.join(project_root, 'data/data-organised')
# '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/data/data-organised'  # only for bear outage time

base_fpath = op.join(data_root, f'sub-{subject}', f'ses-{session}', f'{modality}')  
base_fname = f'sub-{subject}_ses-{session}_task-{task}_run-{run}_{modality}'
eeg_fname = op.join(base_fpath, brainVision_basename + '.eeg')  
vhdr_fname = op.join(base_fpath, brainVision_basename + '.vhdr')
events_fname = op.join(base_fpath, base_fname + '-eve.fif')
annotated_raw_fname = op.join(base_fpath, base_fname + extension)
beh_fig_fname = op.join(project_root, 'derivatives/figures', f'sub-{subject}-beh-performance.png')  # where you save the matlab output of behavioural performance plots

# BIDS events
events_suffix = 'events'  
events_extension = '.tsv'
bids_root = op.join(project_root, 'data', 'BIDS')

# Read raw file in BrainVision (.vhdr, .vmrk, .eeg) format
if subject == '110':
    raw_fnames = [op.join(base_fpath, brainVision_basename + '_blocks1-2.vhdr'), 
                  op.join(base_fpath, brainVision_basename + '_blocks3-8.vhdr')]
    raw = mne.concatenate_raws([mne.io.read_raw_brainvision(f, preload=True) for f in raw_fnames])
elif subject == '111':
        raw_fnames = [op.join(base_fpath, brainVision_basename + '_stimright.vhdr'), 
                  op.join(base_fpath, brainVision_basename + '_nostimright.vhdr'),
                  op.join(base_fpath, brainVision_basename + '_nostimleft.vhdr')]
        raw = mne.concatenate_raws([mne.io.read_raw_brainvision(f, preload=True) for f in raw_fnames])
else:
    raw = mne.io.read_raw_brainvision(vhdr_fname, eog=('HEOGL', 'HEOGR', 'VEOGb'), preload=True)

# first thing first- find if you must crop useless data
# raw.plot()  # get an idea about the data, confirm stimulation order and annotate break spans with BAD

# Rename channels according to function
"""T8 and FT10 = vertical electro-oculogram (EOG), 
T7 and FT9 = horizontal EOG;
TP9 and TP10 = mastoids, 
Fz = on-line reference
57 channels on the head."""

raw.rename_channels({'T8':'vEOG1', 
                     'FT10':'vEOG2',
                     'T7':'hEOG1',
                     'FT9':'hEOG2'}
                     )
# Set both vEOG and hEOG as EOG channels
raw.set_channel_types({'vEOG1':'eog', 
                       'vEOG2':'eog',
                       'hEOG1':'eog', 
                       'hEOG2':'eog'}
                       )  
# Remove channels on mastoid
raw.info["bads"].extend(['TP9','TP10'])  

# Read events from raw object
events, _ = mne.events_from_annotations(raw, event_id='auto')
# Create Annotation object with correct labels
"""list of triggers https://github.com/tghafari/STN-stimulation-oscillation/blob/main/Instructions/triggers.md"""
mapping = {1:'cue_onset_right',
           2:'cue_onset_left',
           3:'trial_onset',
           4:'stim_onset',
           5:'catch_onset',
           6:'dot_onset_right',
           7:'dot_onset_left',
           8:'response_press_onset',
           20:'block_onset',
           21:'block_end',
           30:'experiment_end',
           #31: 'abort',  # participant 04_wmf has abort
           #10001:'new_stim_segment_maybe',  # sub01 has an extra trigger           
           99999:'new_stim_segment',
        }
annotations_from_events = mne.annotations_from_events(events=events,
                                                    event_desc=mapping,
                                                    sfreq=raw.info["sfreq"],
                                                    orig_time=raw.info["meas_date"],
                                                    )
raw.set_annotations(annotations_from_events)

# Write events in a separate file
mne.write_events(events_fname, events, overwrite=True)  
# Save a non-bids raw just in case 
"""Note that the event_id is incorrect here, use the event_id dict if needed"""
raw.save(annotated_raw_fname, overwrite=True) 

# Have to explicitly assign values to events for brainvision data
event_dict = {'cue_onset_right':1,
           'cue_onset_left':2,
           'trial_onset':3,
           'stim_onset':4,
           'catch_onset':5,
           'dot_onset_right':6,
           'dot_onset_left':7,
           'response_press_onset':8,
           'block_onset':20,
           'block_end':21,
           'experiment_end':30,  #sub02 does not have this
           #'abort':31,  # participant 04_wmf has abort
           #'new_stim_segment_maybe':255,  # sub102 has an extra trigger
           'new_stim_segment':99999, 
        }
_, events_id = mne.events_from_annotations(raw, event_id=event_dict)

# Convert to BIDS
bids_path = BIDSPath(subject=subject, 
                     session=session, 
                     datatype ='eeg',
                     task=task, 
                     run=run, 
                     root=bids_root)

# Write to BIDS format
raw.set_annotations(None)  # have to remove annotations to prevent duplicating when converting to BIDS
write_raw_bids(raw, 
               bids_path, 
               events=events_fname, 
               event_id=events_id, 
               overwrite=True, 
               allow_preload=True,
               format='BrainVision')

# Plot all events
fig = mne.viz.plot_events(events, 
                          sfreq=raw.info["sfreq"], 
                          first_samp=raw.first_samp, 
                          event_id=events_id)

# Plot triggers from bids .tsv file
events_bids_path = bids_path.copy().update(suffix=events_suffix,
                                            extension=events_extension)
events_file = pd.read_csv(events_bids_path, sep='\t')
event_onsets = events_file[['onset', 'value', 'trial_type']]        

# Check event durations
durations_onset = ['cue', 'catch', 'stim', 'dot', 'response_press','trial']
direction_onset = ['cue_onset', 'dot_onset']
events_dict = {}

for dur in durations_onset:    
    events_dict[dur + "_onset"] = event_onsets.loc[event_onsets['trial_type'].str.contains(f'{dur}_onset'),
                                                'onset'].to_numpy()

for dirs in direction_onset:
    events_dict[dirs + "_right"] = event_onsets.loc[event_onsets['trial_type'].str.contains(f'{dirs}_right'),
                                                'onset'].to_numpy()
    events_dict[dirs + "_left"] = event_onsets.loc[event_onsets['trial_type'].str.contains(f'{dirs}_left'),
                                                'onset'].to_numpy()

# Compare number of trials with stimuli and responses
numbers_dict = {}
for numbers in  ['cue_onset_right', 'cue_onset_left', 'dot_onset_right', 'dot_onset_left', 
                        'response_press_onset']:
    numbers_dict[numbers] = events_dict[numbers].size
    
eve_fig, ax = plt.subplots()
bars = ax.bar(range(len(numbers_dict)), list(numbers_dict.values()))
plt.xticks(range(len(numbers_dict)), list(numbers_dict.keys()), rotation=45)
ax.bar_label(bars)
plt.show()

if sanity_test:
    # Check duration of cue presentation  
    events_dict['stim_to_dot_duration'] = events_dict['dot_onset'] - events_dict['stim_onset']
    # Plot  durations
    events_dict["dur_cue_onset"] = events_dict['cue_onset'] - events_dict['trial_onset']
    fig, ax = plt.subplots()
    plt.hist(events_dict["dur_cue_onset"])
    plt.title("dur_cue_onset")
    plt.xlabel('time in sec')
    plt.ylabel('number of events')
    plt.show()


if summary_rprt:
    report_root = op.join(project_root, 'derivatives/reports')  # RDS folder for reports
   
    if not op.exists(op.join(report_root , 'sub-' + subject)):
        os.makedirs(op.join(report_root , 'sub-' + subject))
    report_folder = op.join(report_root , 'sub-' + subject)

    report_fname = op.join(report_folder, 
                        f'sub-{subject}_180225.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub-{subject}_180225.html')
    
    report = mne.Report(title=f'Subject {subject}')
    report.add_image(beh_fig_fname,
                    title='RT and performance',
                    caption='reaction time and behavioural performance',
                    tags=('beh'))
    report.add_events(events=events, 
                    event_id=events_id, 
                    tags=('eve'),
                    title='events from "events"', 
                    sfreq=raw.info['sfreq'])
    report.add_figure(eve_fig,
                        title='Number of events',
                        caption='number of events in total',
                        tags=('eve'))
    report.add_raw(raw=raw.filter(0.1, 100), title='raw not referenced with bad channels', 
                   psd=True, 
                   butterfly=False, 
                   tags=('raw'))
    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
