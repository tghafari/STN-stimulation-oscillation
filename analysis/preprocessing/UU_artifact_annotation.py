# -*- coding: utf-8 -*-
"""
DOES NOT WORK!
===============================================
06. Annotation of artifacts

This code will identify artifacts and then annotate
them for later use (eg., to reject).

This code does not work for BrainVision data yet.
(bug (?) in muscle artifact in mne for non MEG data)

written by Tara Ghafari
adapted from flux pipeline
==============================================
Issues:
- subject 1 does not have eog
- annotating muscle raises this error:

In [40]: threshold_muscle = 10
    ...: min_length_good = .2
    ...: filter_freq = [110,140]
    ...: annotation_muscle, scores_muscle = annotate_muscle_zscore(raw_with_ref,
    ...: 
    ...:                                                           ch_type='eeg'
    ...: ,
    ...:                                                           threshold=thr
    ...: eshold_muscle,
    ...:                                                           min_length_go
    ...: od=min_length_good,
    ...:                                                           filter_freq=f
    ...: ilter_freq)
    ...: annotation_muscle.onset -= raw.first_time  # align the artifact onsets 
    ...: to data onset
    ...: annotation_muscle._orig_time = None  # remove date and time from the an
    ...: notation variable
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[40], line 4
      2 min_length_good = .2
      3 filter_freq = [110,140]
----> 4 annotation_muscle, scores_muscle = annotate_muscle_zscore(raw_with_ref, 
      5                                                           ch_type='eeg', 
      6                                                           threshold=threshold_muscle, 
      7                                                           min_length_good=min_length_good, 
      8                                                           filter_freq=filter_freq)
      9 annotation_muscle.onset -= raw.first_time  # align the artifact onsets to data onset
     10 annotation_muscle._orig_time = None  # remove date and time from the annotation variable

File <decorator-gen-470>:12, in annotate_muscle_zscore(raw, threshold, ch_type, min_length_good, filter_freq, n_jobs, verbose)

File ~/miniconda3/envs/mne/lib/python3.11/site-packages/mne/preprocessing/artifact_detection.py:106, in annotate_muscle_zscore(raw, threshold, ch_type, min_length_good, filter_freq, n_jobs, verbose)
    104 else:
    105     ch_type = {"meg": False, ch_type: True}
--> 106     raw_copy.pick(**ch_type)
    108 raw_copy.filter(
    109     filter_freq[0],
    110     filter_freq[1],
   (...)
    113     n_jobs=n_jobs,
    114 )
    115 raw_copy.apply_hilbert(envelope=True, n_jobs=n_jobs)

TypeError: pick() got an unexpected keyword argument 'meg'

"""

import os.path as op
import os
import numpy as np

import mne
from mne.preprocessing import annotate_muscle_zscore
from mne_bids import BIDSPath, read_raw_bids


# BIDS settings: fill these out 
subject = '107'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
deriv_suffix = 'ann'
extension = '.fif'

pilot = False  # is it pilot data or real data?
summary_rprt = True  # do you want to add evokeds figures to the summary report?
platform = 'bluebear'  # are you using 'bluebear', 'mac', or 'windows'?

if platform == 'bluebear':
    rds_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    camcan_dir = '/rds/projects/q/quinna-camcan/dataman/data_information'
elif platform == 'windows':
    rds_dir = 'Z:'
    camcan_dir = 'X:/dataman/data_information'
elif platform == 'mac':
    rds_dir = '/Volumes/jenseno-avtemporal-attention'
    camcan_dir = '/Volumes/quinna-camcan/dataman/data_information'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
if pilot:
    bids_root = op.join(project_root, 'data', 'pilot-BIDS')
else:
    bids_root = op.join(project_root, 'data', 'BIDS')

# Specify specific file names
bids_path = BIDSPath(subject=subject, session=session,
                     task=task, run=run, root=bids_root, 
                     datatype ='eeg', suffix=eeg_suffix)
deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results
if not op.exists(deriv_folder):
    os.makedirs(deriv_folder)
deriv_fname = op.join(deriv_folder, bids_path.basename + '_' + deriv_suffix + extension)  # prone to change if annotation worked for eeg brainvision

# Read raw data 
raw = read_raw_bids(bids_path=bids_path, verbose=False, 
                     extra_params={'preload':True})

"""
# Identifying and annotating eye blinks using vEOG (EOG001)
raw.copy().pick_channels(ch_names=['EOG001','EOG002'   # vEOG, hEOG, EKG
                                       ,'ECG003']).plot()  # 'plot to make sure channel' 
                                                        # 'names are correct, rename otherwise'

eog_events = mne.preprocessing.find_eog_events(raw)
onset = eog_events[:,0] / raw_sss.info['sfreq'] -.25 #'from flux pipline, but why?'
                                                     # 'blink onsets in seconds'
onset -= raw_sss.first_time  # first_time is apparently the time start time of the raw data
n_blinks = len(eog_events)  # length of the event file is the number of blinks in total
duration = np.repeat(.5, n_blinks)  # duration of each blink is assumed to be 500ms
description = ['blink'] * n_blinks
annotation_blink = mne.Annotations(onset, duration, description)


# Identifying and annotating muscle artifacts

muscle artifacts are identified from the magnetometer data filtered and 
z-scored in filter_freq range

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
raw.set_annotations(annotation_muscle)
raw.set_channel_types({'EOG001':'eog', 'EOG002':'eog', 'ECG003':'ecg'})  # set both vEOG and hEOG as EOG channels
eog_picks = mne.pick_types(raw.info, meg=False, eog=True)
scale = dict(eog=500e-6)
raw.plot(order=eog_picks, scalings=scale, start=50)

# Save the artifact annotated file
raw.save(deriv_fname, overwrite=True)
"""












