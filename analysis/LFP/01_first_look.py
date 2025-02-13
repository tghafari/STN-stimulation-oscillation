"""
===============================================
01_first_look_and_BIDS_conversion
    1. this code opens .eeg file
    and plot raw eeg data.
    2. reads the events from annotations of 
    brainvision data.
    3. corrects the annotation and event_ids 
    4. adds annotations to the raw 
    5. standardise montage and set reference
    6. saves as .fif
    7. then converts the raw data to bids
    8. It also plots triggers and RT to quality
    check the data.

    note that this code will save two eeg files,
    one .fif in the original folder and one .fif 
    bids in the bids folder.

written by Tara Ghafari
t.ghafari@bham.ac.uk
==============================================  

ToDos:
- avg RT is very long
- number of button presses = right dot, as if 
they only responded to right dot not left.
"""

# Import relevant Python modules
import os.path as op
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne_bids import BIDSPath, print_dir_tree, write_raw_bids
from mne_bids.stats import count_events


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
subject = '103'
session = '01'
task = 'SpAtt'
run = '01'
modality = 'lfp'
side = 'lfp left'

# BIDS events
events_suffix = 'events'  
events_extension = '.tsv'

platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
pilot = False  # is it pilot data or real data?
sanity_test = False
eve_rprt = True
summary_rprt = True

if platform == 'bluebear':
    rds_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/jenseno-avtemporal-attention-2'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
# '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD'  # only for bear outage time
data_root = op.join(project_root, 'data/data-organised')
# '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/data/data-organised'  # only for bear outage time

bids_root = op.join(project_root, 'data', 'BIDS')
base_fpath = op.join(data_root, f'sub-{subject}', f'ses-{session}', f'{modality}', f'{side}')  
base_fname = '1010P24295_2025_01_15_09_57_47_uv'

lfp_fname = op.join(base_fpath, base_fname + '.edf') 
events_fname = op.join(base_fpath, base_fname + '-eve.fif')
annotated_raw_fname = op.join(base_fpath, base_fname + '_ann.fif')

# Read raw file in BrainVision (.vhdr, .vmrk, .eeg) format
raw = mne.io.read_raw_edf(lfp_fname, preload=True)
raw.plot()  # first thing first

correct_sfreq = 1250  # note that the file headers incorrectly mention 1000 whereas the correct sfreq is 1250Hz

# Create a new Info object with the updated sampling frequency.
# Here we assume all channels are EEG channels. Adjust ch_types if needed.
new_info = mne.create_info(ch_names=raw.info['ch_names'], 
                           sfreq=correct_sfreq, 
                           ch_types='eeg')

# Create a new RawArray with the same data and the new info
new_raw = mne.io.RawArray(raw.get_data(), new_info)
# Copy annotations and set orig_time to None
new_annotations = raw.annotations.copy()
new_annotations.orig_time = None     # this doesn't work

# Set the annotations on the new raw object
new_raw.set_annotations(new_annotations)

# Remove the first few seconds while the LFP is warming up.
cropped_raw = new_raw.copy().crop(tmin=119, tmax=491)
cropped_raw.plot()

cropped_raw.info["line_freq"] = 50  # specify power line frequency as required by BIDS
# cropped_raw.info['sfreq'] = 1250

# Read events from raw object
"""Note that events_from_annotations messes up the 
event values. We still have to proceed with this method as
there's no other way to retrieve events from the
raw object as there are no stim channels."""
events_messed_up, _ = mne.events_from_annotations(cropped_raw, event_id='auto')

events_messed_up_unique = np.unique(events_messed_up[:,2])  # just to check how many unique events are in the raw- should be equal to the next line
events_messed_up_unique_len = len(events_messed_up)
annotations_unique = cropped_raw.annotations.count()  # both should have 10 unique events
annotations_unique_len = cropped_raw.annotations

# Create Annotation object with correct labels
"""use values in events variable. events_from_annotations assigns values from 1 to 
however many events are in the raw file in some order I don't understand.
(from mne: Map descriptions to unique integer values based on their sorted order.)
Make sure to plot and check which value corresponds to which event and define 
them in the mapping dictionary below."""

# mapping = {1:'cue_onset_right',
#            4:'cue_onset_left',
#            5:'trial_onset',
#            6:'stim_onset',
#            7:'catch_onset',
#            8:'dot_onset_right',
#            9:'dot_onset_left',
#            10:'response_press_onset',
#            2:'block_onset',
#            3:'block_end',         
#         }  % maybe for 108? or old subjects

mapping_103 = {1:'cue_onset_left',
           2:'cue_onset_right',
           3:'trial_onset',
           4:'stim_onset',
           5:'catch_onset',
           7:'dot_onset_right',
           6:'dot_onset_left',
           8:'response_press_onset',
           15:'block_onset',
           14:'block_end',         
        }  # right and left dot/cue might be the other way around
#annotations_from_events with mapping decreases events 217->170 and messes up the ids
annotations_from_events = mne.annotations_from_events(events=events_messed_up,
                                                    event_desc=mapping_103,
                                                    sfreq=cropped_raw.info["sfreq"],
                                                    orig_time=cropped_raw.info["meas_date"],
                                                    )
cropped_raw.set_annotations(annotations_from_events)

# Plot to test
cropped_raw.plot(title="raw") 
n_fft = int(cropped_raw.info['sfreq']*2)  # to ensure window size = 2
psd_fig = cropped_raw.copy().compute_psd(method='welch',
                                         n_fft=n_fft,  # default method is welch here (multitaper for epoch)
                                         n_overlap=int(n_fft/2),
                                         fmax=100).plot()  
                                                                                      
# Have to explicitly assign values to events for brainvision data
"""list of triggers https://github.com/tghafari/STN-stimulation-oscillation/blob/main/Instructions/triggers.md"""
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
            }
events, events_id = mne.events_from_annotations(cropped_raw, event_id=event_dict)
mne.write_events(events_fname, events, overwrite=True)  # write events in a separate file

# Save a non-bids raw just in case 
cropped_raw.save(annotated_raw_fname, overwrite=True)  # note that the event_id is incorrect here, use the event_id dict if needed

# Convert to BIDS
bids_path = BIDSPath(subject=subject, session=session,
                     task=task, run=run, root=bids_root, datatype='eeg',  # BIDS does not accept LFP
                     extension='.edf').mkdir(exist_ok=True)


# Write to BIDS format
cropped_raw.set_annotations(None)  # have to remove annotations to prevent duplicating when converting to BIDS
write_raw_bids(cropped_raw, 
               bids_path, 
               events=events_fname, 
               event_id=events_id, 
               overwrite=True, 
               allow_preload=True,
               format='EDF')

# Plot all events
fig = mne.viz.plot_events(events, sfreq=cropped_raw.info["sfreq"], 
                          first_samp=cropped_raw.first_samp, 
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


# here needs a for loop to read the tsv file line by line and checks for 
# all events between two trial_onsets
# calculate reaction time
# add event info + rt to a report called: sanity checks

if sanity_test:
    # Check duration of cue presentation  
    events_dict['stim_to_dot_duration'] = events_dict['dot_onset'] - events_dict['stim_onset']
    events_dict['RT'] = events_dict['response_press_onset'] - events_dict['dot_onset_right']  # participants 01 and 04 only responded to right stim -> do not execute this line
    # cheat line to add extra elements for easier calculation of RTs:
    # events_dict['dot_onset'] = np.insert(events_dict['dot_onset'], index_onset_should_be_added_before, onset)

    # Plot  durations
    # events_dict["dur_cue_onset"] = events_dict['cue_onset'] - events_dict['trial_onset']
    fig, ax = plt.subplots()
    plt.hist(events_dict["RT"])
    plt.title("RT")
    plt.xlabel('time in sec')
    plt.ylabel('number of events')
    plt.show()


if summary_rprt:
    report_root = op.join(project_root, 'derivatives/reports')  # RDS folder for reports
   
    if not op.exists(op.join(report_root , 'sub-' + subject)):
        os.makedirs(op.join(report_root , 'sub-' + subject))
    report_folder = op.join(report_root , 'sub-' + subject)

    report_fname = op.join(report_folder, 
                        f'sub-{subject}_LFP.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub-{subject}_LFP.html')
    
    report = mne.Report(title=f'Subject {subject}-LFP')
    if eve_rprt:
        report.add_events(events=events, 
                        event_id=events_id, 
                        tags=('eve'),
                        title='events from "events"', 
                        sfreq=cropped_raw.info['sfreq'])
    report.add_raw(raw=cropped_raw.filter(0.3, 100), title='raw with bad channels', 
                   psd=True, 
                   butterfly=False, 
                   tags=('raw'))
    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
