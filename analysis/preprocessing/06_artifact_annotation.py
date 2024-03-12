# -*- coding: utf-8 -*-
"""
===============================================
06. Annotation of artifacts

This code will identify artifacts and then annotate
them for later use (eg., to reject).

This code does not work for BrainVision data yet.
(bug in muscle artifact in mne for non MEG data)

written by Tara Ghafari
adapted from flux pipeline
==============================================
"""

import os.path as op
import os
import numpy as np

import mne
from mne.preprocessing import annotate_muscle_zscore
from mne_bids import BIDSPath, read_raw_bids

# BIDS settings: fill these out 
subject = '01'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
deriv_suffix = 'ann'

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

project_root = r'/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD'  # this is to use for bear outage

# Specify specific file names
bids_root = op.join(project_root, 'Data', 'BIDS')
bids_path = BIDSPath(subject=subject, session=session,
                     task=task, run=run, root=bids_root, 
                     datatype ='eeg', suffix=eeg_suffix)
deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results
if not op.exists(deriv_folder):
    os.makedirs(deriv_folder)
deriv_fname = bids_path.basename + '_' + deriv_suffix

# Read raw data 
raw = read_raw_bids(bids_path=bids_path, verbose=False, 
                     extra_params={'preload':True})

# Preparing the brainvision data format to standard
montage = mne.channels.make_standard_montage("easycap-M1")
raw.set_montage(montage, verbose=False)

# Set common average reference
raw.set_eeg_reference("average", projection=False, verbose=False)

"""
# Identifying and annotating eye blinks using vEOG (EOG001)
raw.copy().pick_channels(ch_names=['EOG001','EOG002'   # vEOG, hEOG, EKG
                                       ,'ECG003']).plot()  # 'plot to make sure channel' 
                                                           # 'names are correct, rename otherwise'
eog_events = mne.preprocessing.find_eog_events(raw_sss, ch_name='EOG001')
onset = eog_events[:,0] / raw_sss.info['sfreq'] -.25 #'from flux pipline, but why?'
                                                     # 'blink onsets in seconds'
onset -= raw_sss.first_time  # first_time is apparently the time start time of the raw data
n_blinks = len(eog_events)  # length of the event file is the number of blinks in total
duration = np.repeat(.5, n_blinks)  # duration of each blink is assumed to be 500ms
description = ['blink'] * n_blinks
annotation_blink = mne.Annotations(onset, duration, description)
"""

# Identifying and annotating muscle artifacts
""" 
muscle artifacts are identified from the magnetometer data filtered and 
z-scored in filter_freq range
"""
threshold_muscle = 10
min_length_good = .2
filter_freq = [110,140]
annotation_muscle, scores_muscle = annotate_muscle_zscore(raw, 
                                                          ch_type='eeg', 
                                                          threshold=threshold_muscle, 
                                                          min_length_good=min_length_good, 
                                                          filter_freq=filter_freq)
annotation_muscle.onset -= raw.first_time  # align the artifact onsets to data onset
annotation_muscle._orig_time = None  # remove date and time from the annotation variable

# Include annotations in dataset and inspect
raw.set_annotations(annotation_blink + annotation_muscle)
raw.set_channel_types({'EOG001':'eog', 'EOG002':'eog', 'ECG003':'ecg'})  # set both vEOG and hEOG as EOG channels
eog_picks = mne.pick_types(raw.info, meg=False, eog=True)
scale = dict(eog=500e-6)
raw.plot(order=eog_picks, scalings=scale, start=50)

# Save the artifact annotated file
raw.save(deriv_fname, overwrite=True)













