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

# fill these out
subj_code = '01_ly'  # subject code assigned to by Benchi's group
subj_name = 'Liuyu'  # name on eeg file
platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
pilot = True  # is it pilot data or real data?
eve_rprt = True
summary_rprt = True

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
annotated_raw_fname = base_fname + '_eeg.fif'

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
raw = mne.io.read_raw_brainvision(vhdr_fname, eog=('HEOGL', 'HEOGR', 'VEOGb'), preload=True)

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
           99999:'new_stim_segment',
        }
annotations_from_events = mne.annotations_from_events(events=events,
                                                    event_desc=mapping,
                                                    sfreq=raw.info["sfreq"],
                                                    orig_time=raw.info["meas_date"],
                                                    )
raw.set_annotations(annotations_from_events)

## Set Fz reference - not for now - 
"""you will have to drop (Fz) for plotting if you add it to the ch_names"""
# we'l decide later if we want to do common average reference
#raw.add_reference_channels(ref_channels=['Fz'])  # the reference channel is not by default in the channel list
#raw.set_eeg_reference(ref_channels=['Fz'], projection=False, verbose=False)

# Preparing the brainvision data format to standard
montage = mne.channels.make_standard_montage("easycap-M1")
raw.set_montage(montage, verbose=False)

mne.write_events(events_fname, events, overwrite=True)  # write events in a separate file

# Filter raw data (130Hz is stimulation frequency)
raw.filter(0.3,100)

# Plot to test
raw.plot(title="raw") 
n_fft = int(raw.info['sfreq']*2)  # to ensure window size = 2
psd_fig = raw.copy().compute_psd(n_fft=n_fft,  # default method is welch here (multitaper for epoch)
                                 n_overlap=int(n_fft/2),
                                 fmax=105).plot()  
                                                                                         
# Save a non-bids raw just in case 
raw.save(annotated_raw_fname, overwrite=True)  # note that the event_id is incorrect here, use the event_id dict if needed

# Have to explicitly assign values to events 
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
           'experiment_end':30,
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
durations_onset = ['cue', 'catch', 'stim', 'dot', 'response_press']
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

# Check duration of cue presentation  
events_dict['stim_to_dot_duration'] = events_dict['dot_onset'] - events_dict['stim_onset']
events_dict['RT'] = events_dict['response_press_onset'] - events_dict['dot_onset']  # participant 01 only responded to right stim -> do not execute this line
# cheat line to add extra elements for easier calculation of RTs:
    #events_dict['dot_onset'] = np.insert(events_dict['dot_onset'], index_onset_should_be_added_before, onset)

# Plot all durations
for dur in ['RT']:
    fig, ax = plt.subplots()
    plt.hist(events_dict[dur])
    plt.title(dur)
    plt.xlabel('time in sec')
    plt.ylabel('number of events')
    plt.show()


if summary_rprt:
    report_root = op.join(project_root, 'results/reports')  # RDS folder for reports
   
    if not op.exists(op.join(report_root , 'sub-' + subject)):
        os.makedirs(op.join(report_root , 'sub-' + subject))
    report_folder = op.join(report_root , 'sub-' + subject)

    report_fname = op.join(report_folder, 
                        f'sub-{subject}_preproc_1.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub-{subject}_preproc_1.html')
    
    report = mne.open_report(report_fname)
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
