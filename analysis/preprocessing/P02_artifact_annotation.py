# -*- coding: utf-8 -*-
"""
===============================================
06. Annotation of artifacts

This code will identify artifacts and then annotate
them for later use (eg., to reject).


written by Tara Ghafari
adapted from flux pipeline
==============================================
Issues:

"""

import os.path as op
import os
import numpy as np
import matplotlib.pylab as plt

import mne
from mne.preprocessing import find_eog_events, annotate_muscle_zscore
from mne_bids import BIDSPath, read_raw_bids


# BIDS settings: fill these out 
subject = '110'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
deriv_suffix = 'ann'
extension = '.fif'

platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?

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
bids_root = op.join(project_root, 'data', 'BIDS')

# for bear outage
bids_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/BIDS'

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


# Identifying and annotating eye blinks using vEOG
eog_events = find_eog_events(raw)
onset = eog_events[:,0] / raw.info['sfreq'] -.25 #'from flux pipline and mne tutorial but why?'
n_blinks = len(eog_events)  # length of the event file is the number of blinks in total
duration = np.repeat(.5, n_blinks)  # duration of each blink is assumed to be 500ms
description = ['blink'] * n_blinks
orig_time = raw.info['meas_date']
annotation_blink = mne.Annotations(onset, duration, description, orig_time)

# Identifying and annotating muscle artefact
"""I prefer not to do muscle annotation, as stimulation influences this.
better do it manually after segmenting and epoching"""
threshold_muscle = 10
min_length_good = .2
filter_freq = [60,100]
annotation_muscle, scores_muscle = annotate_muscle_zscore(raw.copy().filter(0.1,100),  # remove stimulation frequency
                                                        ch_type='eeg',
                                                        threshold=threshold_muscle,
                                                        min_length_good=min_length_good,
                                                        filter_freq=filter_freq
                                                        )
# Plot muscle annotation zscores
_, ax = plt.subplots()
ax.plot(raw.times, scores_muscle)
ax.axhline(y=threshold_muscle, color='r')
ax.set(xlabel='time, (s)', ylabel='zscore', title='Muscle activity (threshold = %s)' % threshold_muscle)
plt.show()

# Include annotations in dataset and inspect
"""make sure to add event annotations here again, 
because set_annotations overwrites all annotations"""
annotations_event = raw.annotations 
raw.set_annotations(annotations_event + annotation_blink)

# Plot annotated dataset
raw.plot()

# Save the artifact annotated file
raw.save(deriv_fname, overwrite=True)













