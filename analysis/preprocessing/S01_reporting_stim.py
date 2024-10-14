# -*- coding: utf-8 -*-
"""
===============================================
S01. reporting stimulation epochs

This code will:

    1. read in the ica (from fif file)
    2. epoch again with reject_by_annotation=False
    3. plot raw_ica and manually note down the
    time points in which stimulation started
    or ended.
    4. crop the data into stimulation and 
    no-stimulation parts.
    5. plot psd for each part separately to
    double check if 130Hz peak exists or not.
    6. save the cropped data.

    Cropped data will then be loaded into
    08_epoching_SpAtt.py for further analyses.


written by Tara Ghafari
==============================================
ToDos:

questions?

"""

import os.path as op
import os
import numpy as np
import pandas as pd

import mne
from mne_bids import BIDSPath
import matplotlib.pyplot as plt
from autoreject import get_rejection_threshold


# BIDS settings: fill these out 
subject = '108'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
input_suffix = 'ica'
no_stim_suffix = 'no_stim'
stim_suffix = 'stim'
extension = '.fif'

pilot = False  # is it pilot data or real data?
summary_rprt = True  # do you want to add evokeds figures to the summary report?
platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
test_plot = False

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
    data_root = op.join(project_root, 'Data/pilot-data/AO')
else:
    data_root = op.join(project_root, 'Data/real-data')

# Specify specific file names
ROI_dir = op.join(project_root, 'results/lateralisation-indices')
bids_root = op.join(project_root, 'Data', 'BIDS')
bids_path = BIDSPath(subject=subject, session=session,
                     task=task, run=run, root=bids_root, 
                     datatype ='eeg', suffix=eeg_suffix)
deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results
input_fname = op.join(deriv_folder, bids_path.basename + '_' + input_suffix + extension)
no_stim_fname = op.join(deriv_folder, bids_path.basename + '_' + no_stim_suffix + extension)  
stim_fname = op.join(deriv_folder, bids_path.basename + '_' + stim_suffix + extension)  

# Read ica cleaned data
raw_ica = mne.io.read_raw_fif(input_fname, verbose=True, preload=True)

# Plot to find the points to crop
raw_ica.plot()

# Record the cropped time points for each subject
stimulation_cropped_time = {"sub-108_no-stim": [8, 890],
                            "sub-108_stim": [930, 1882]}

no_stim_segment = raw_ica.copy().crop(tmin=stimulation_cropped_time[f'sub-{subject}_no-stim'][0], 
                                      tmax=stimulation_cropped_time[f'sub-{subject}_no-stim'][1])
no_stim_segment.compute_psd().plot()  # double check and save if ok
no_stim_segment.save(no_stim_fname)

stim_segment = raw_ica.copy().crop(tmin=stimulation_cropped_time[f'sub-{subject}_stim'][0],
                                   tmax=stimulation_cropped_time[f'sub-{subject}_stim'][1])
stim_segment.compute_psd().plot() 
stim_segment.save(stim_fname)


if summary_rprt:
    report_root = op.join(project_root, 'results/reports')  # RDS folder for reports
   
    if not op.exists(op.join(report_root , 'sub-' + subject)):
        os.makedirs(op.join(report_root , 'sub-' + subject))
    report_folder = op.join(report_root , 'sub-' + subject)

    report_fname = op.join(report_folder, 
                        f'sub-{subject}_preproc_1.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub-{subject}_preproc_1.html')
    
    report = mne.Report(title=f'Subject {subject}')
    report.add_raw(raw=raw.filter(0.3, 100), title='raw with bad channels', 
                   psd=True, 
                   butterfly=False, 
                   tags=('raw'))
    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
