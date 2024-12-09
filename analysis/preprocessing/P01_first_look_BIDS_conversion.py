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
    6. standardise montage and set reference
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
                 } 
# BIDS settings
subject = '110'
brainVision_basename = f'ao_{subject[-2:]}'  # might need modification per subject

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
    rds_dir = '/Volumes/jenseno-avtemporal-attention-2'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
data_root = op.join(project_root, 'data/data-organised')

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
else:
    raw = mne.io.read_raw_brainvision(vhdr_fname, eog=('HEOGL', 'HEOGR', 'VEOGb'), preload=True)

# first thing first- Remove bad channels from raw scrolling and find if you must crop useless data
raw.plot()  # get an idea about the data, confirm stimulation order

# Here crop any extra segments at the beginning or end of the recording
raw.crop(tmin=290).plot()  # sub110

# Scrole one more time to remove any other bad channels after filtering
raw.copy().filter(l_freq=0.1, h_freq=100).plot()  


n_fft = int(raw.info['sfreq']*2)  # to ensure window size = 2sec
psd_fig = raw.copy().filter(0.3,100).compute_psd(n_fft=n_fft,  # default method is welch here (multitaper for epoch)
                                                n_overlap=int(n_fft/2),
                                                fmax=105).plot()  

bad_channels = True  # are there any more bad channels from psd?
# Mark bad channels before ICA
if bad_channels:
    original_bads = deepcopy(raw.info["bads"])
    print(f'these are original bads: {original_bads}')
    user_list = input('Are there any bad channels after rejecting bad epochs? name of channel, e.g. FT10 T9 (separate by space) or return.')
    bad_channels = user_list.split()
    raw.copy().pick(bad_channels).compute_psd().plot()  # double check bad channels
    if len(bad_channels) == 1:
        print('one bad channel removing')
        raw.info["bads"].append(bad_channels[0])  # add a single channel
    else:
        print(f'{len(bad_channels)} bad channels removing')
        raw.info["bads"].extend(bad_channels)  # add a list of channels - should there be more than one channel to drop

"""
list bad channels for all participants:
{
pilot_BIDS/sub-01_ses-01_run-01: ["FCz"],
pilot_BIDS/sub-02_ses-01_run-01: [],
BIDS/sub-01_ses-01_run-01: ["T7", "FT10"],
BIDS/sub-02_ses-01_run-01: ["TP10"],
BIDS/sub-05_ses-01_run-01: ["almost all channels look terrible in psd"],
BIDS/sub-107_ses-01_run-01: ["FT10"], #"all good!"
BIDS/sub-108_ses-01_run-01: ["FT9", "T8", "T7"],
BIDS/sub-110_ses-01_run-01: ["T8", "FT10", "FCz", "TP9"],
} """

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

#sub-05 first 1300seconds are before the task starts.
# if subj_code == 'sub05':    
#     raw.crop(tmin=1380)

# Set average reference 
# First, plot the channel layout
mne.viz.plot_sensors(raw.info, 
                     ch_type='all', 
                     show_names=True, 
                     ch_groups='position',
                     to_sphere=False,  # the sensor array appears similar as to looking downwards straight above the subject’s head.
                     linewidth=0,
                     )

channels_are_even = True
# If channels distrubited evenly, do average reference (eeglab resources)
if channels_are_even:
    raw_referenced = raw.copy()
    raw_referenced.set_eeg_reference(ref_channels="average")
    raw_referenced.plot(title='Average reference raw')

# if not evenly distributed, do Fz reference which = not referencing
else:
    raw_referenced = raw.copy()
    raw_referenced.add_reference_channels(ref_channels=['Fz']) # the reference channel is not by default in the channel list
    raw_referenced.set_eeg_reference(ref_channels=['Fz'], projection=False, verbose=False)
    raw_referenced.plot(title='Fz reference raw')

# Preparing the brainvision data format to standard
"""it is important to bring the montage to the standard space. Otherwise the 
ICA and PSDs look weird."""
# Only do this after Sirui sent the montage
# montage = mne.channels.make_standard_montage("easycap-M1") 
# raw_referenced.set_montage(montage, verbose=False)
# montage.plot()  # 2D

# Save a non-bids raw just in case 
raw_referenced.save(annotated_raw_fname, overwrite=True)  # note that the event_id is incorrect here, use the event_id dict if needed

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
           #'new_stim_segment_maybe':10001,  # sub01 has an extra trigger
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
raw_referenced.set_annotations(None)  # have to remove annotations to prevent duplicating when converting to BIDS
write_raw_bids(raw_referenced, 
               bids_path, 
               events=events_fname, 
               event_id=events_id, 
               overwrite=True, 
               allow_preload=True,
               format='BrainVision')

# Plot to test, filter raw data (130Hz is stimulation frequency)
psd_fig = raw_referenced.copy().filter(0.3,100).compute_psd(n_fft=n_fft,  # default method is welch here (multitaper for epoch)
                                                n_overlap=int(n_fft/2),
                                                fmax=105).plot()  

# Plot all events
fig = mne.viz.plot_events(events, sfreq=raw_referenced.info["sfreq"], first_samp=raw_referenced.first_samp, event_id=events_id)

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
    
eve_fig, ax = plt.subplots()
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
                        f'sub-{subject}_091224.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub-{subject}_091224.html')
    
    report = mne.Report(title=f'Subject {subject}')
    report.add_image(beh_fig_fname,
                    title='RT and performance',
                    caption='reaction time and behavioural performance',
                    tags=('beh'))
    report.add_events(events=events, 
                    event_id=events_id, 
                    tags=('eve'),
                    title='events from "events"', 
                    sfreq=raw_referenced.info['sfreq'])
    report.add_figure(eve_fig,
                        title='Number of events',
                        caption='number of events in total',
                        tags=('eve'))
    report.add_raw(raw=raw_referenced.filter(0.3, 100), title='raw referenced with bad channels marked', 
                   psd=True, 
                   butterfly=False, 
                   tags=('raw'))
    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
