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
import pandas as pd

import mne
from mne_bids import (BIDSPath, write_raw_bids, read_raw_bids)
import matplotlib.pyplot as plt

# Fill these out
subj_code = 'sub05'  # subject code assigned to by Benchi's group
base_fname = '5.15_5_AO'  # the name of the eeg file for 01_ly to sub05 should be manually copied here

# Stimulation sequence
"""copy the stim sequence for each participant from here: 
https://github.com/tghafari/STN-stimulation-oscillation/wiki/Stimulation-table"""
stim_sequence = {'sub-01':["no_stim-left rec", "no_stim-right rec", "Right stim- no rec", "Left stim- no rec"],
                 'sub-02':["no_stim-left rec", "no_stim-right rec", "Left stim- no rec", "Right stim- no rec"],
                 'sub-05':["Left stim- no rec", "Right stim- no rec", "no_stim-left rec", "no_stim-right rec"]}  

# BIDS settings
subject = '05'
session = '01'
task = 'SpAtt'
run = '01'

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
    rds_dir = '/Volumes/jenseno-avtemporal-attention'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
if pilot:
    data_root = op.join(project_root, 'data/pilot-data')
else:
    data_root = op.join(project_root, 'data/real-data')

bids_root = op.join(project_root, 'data', 'BIDS')
base_fpath = op.join(data_root, subj_code, f'EEG_{subj_code}')  
eeg_fname = op.join(base_fpath, base_fname + '.eeg')  
vhdr_fname = op.join(base_fpath, base_fname + '.vhdr')
events_fname = op.join(base_fpath, base_fname + '-eve.fif')
annotated_raw_fname = op.join(base_fpath, base_fname + '_eeg.fif')

# Read raw file in BrainVision (.vhdr, .vmrk, .eeg) format
raw = mne.io.read_raw_brainvision(vhdr_fname, eog=('HEOGL', 'HEOGR', 'VEOGb'), preload=True)
raw.plot()  # first thing first

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
           #99999:'new_stim_segment',
        }
annotations_from_events = mne.annotations_from_events(events=events,
                                                    event_desc=mapping,
                                                    sfreq=raw.info["sfreq"],
                                                    orig_time=raw.info["meas_date"],
                                                    )
raw.set_annotations(annotations_from_events)

## Set Fz reference 
"""you will have to drop (Fz) for plotting if you add it to the ch_names"""
"""
# - not for now - we'l decide later if we want to do common average reference
raw.add_reference_channels(ref_channels=['Fz'])  # the reference channel is not by default in the channel list
raw.set_eeg_reference(ref_channels=['Fz'], projection=False, verbose=False)

# Preparing the brainvision data format to standard
montage = mne.channels.make_standard_montage("easycap-M1")
raw.set_montage(montage, verbose=False)
"""

mne.write_events(events_fname, events, overwrite=True)  # write events in a separate file

# Filter raw data (130Hz is stimulation frequency)
raw.filter(0.3,100)

# Plot to test
raw.plot(title="raw") 
n_fft = int(raw.info['sfreq']*2)  # to ensure window size = 2
psd_fig = raw.copy().compute_psd(n_fft=n_fft,  # default method is welch here (multitaper for epoch)
                                 n_overlap=int(n_fft/2),
                                 fmax=105).plot()  

#sub-05 first 1300seconds are before the task starts.
if subj_code == 'sub05':    
    raw.crop(tmin=1380)
                                                                                         
# Save a non-bids raw just in case 
raw.save(annotated_raw_fname, overwrite=True)  # note that the event_id is incorrect here, use the event_id dict if needed

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
           #'experiment_end':30,  #sub02 does not have this
           #'abort':31,  # participant 04_wmf has abort
           #'new_stim_segment_maybe':10001,  # sub01 has an extra trigger
           'new_stim_segment':99999, 
        }
_, events_id = mne.events_from_annotations(raw, event_id=event_dict)

# Convert to BIDS
bids_path = BIDSPath(subject=subject, session=session, datatype ='eeg',
                     task=task, run=run, root=bids_root)

# Write to BIDS format
raw.set_annotations(None)  # have to remove annotations to prevent duplicating when converting to BIDS
write_raw_bids(raw, bids_path, events_data=events_fname, 
               event_id=events_id, overwrite=True, allow_preload=True,
               format='BrainVision')

# Plot all events
fig = mne.viz.plot_events(events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=events_id)

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

if sanity_test:
    # Check duration of cue presentation  
    events_dict['stim_to_dot_duration'] = events_dict['dot_onset'] - events_dict['stim_onset']
    # events_dict['RT'] = events_dict['response_press_onset'] - events_dict['dot_onset']  # participants 01 and 04 only responded to right stim -> do not execute this line
    # cheat line to add extra elements for easier calculation of RTs:
    # events_dict['dot_onset'] = np.insert(events_dict['dot_onset'], index_onset_should_be_added_before, onset)

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
                        f'sub-{subject}_preproc_1.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub-{subject}_preproc_1.html')
    
    report = mne.Report(title=f'Subject {subject}')
    if eve_rprt:
        report.add_events(events=events, 
                        event_id=events_id, 
                        tags=('eve'),
                        title='events from "events"', 
                        sfreq=raw.info['sfreq'])
    report.add_raw(raw=raw.filter(0.3, 100), title='raw with bad channels', 
                   psd=True, 
                   butterfly=False, 
                   tags=('raw'))
    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
