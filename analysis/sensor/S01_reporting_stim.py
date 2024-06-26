# -*- coding: utf-8 -*-
"""
===============================================
S01. reporting stimulation epochs

This code will:

    1. read in the ica (from fif file)
    2. epoch again with reject_by_annotation=False
    3. divide stim epochs from no stim epochs
    based on the psd (with or without 130Hz peak)
    4. plot psd
    5. plot mean psd on parieto occipital 
    channels
    6. plots TFR plot_topo
    7. plots TFR on representative channels
    8. MI topographically 


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
subject = '02'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
input_suffix = 'ica'
deriv_suffix = 'stm-epo'
extension = '.fif'

pilot = True  # is it pilot data or real data?
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
deriv_fname = op.join(deriv_folder, bids_path.basename + '_' + deriv_suffix + extension)  # prone to change if annotation worked for eeg brainvision

peak_alpha_fname = op.join(ROI_dir, f'sub-{subject}_peak_alpha.npz')  # 2 numpy arrays saved into an uncompressed file

# read annotated data
raw_ica = mne.io.read_raw_fif(input_fname, verbose=True, preload=True)
events, events_id = mne.events_from_annotations(raw_ica, event_id='auto')
              
# epoch ica data to make sure an entire block epoch is not rejected by annotation
epochs_block_onset_end = mne.Epochs(raw_ica, 
                    events, 
                    events_id,   # select only block_onset(3) and block_end(2) events                   
                    tmin=-0.7, 
                    tmax=1.7,
                    baseline=None, 
                    proj=True,  
                    picks='all', 
                    detrend=1, 
                    event_repeated='error',
                    reject=None,  # we'll reject after calculating the threshold
                    reject_by_annotation=False,
                    preload=True, 
                    verbose=True)

# separately plot psd for each block: which block is stim on which stim off?































if summary_rprt:
    report_root = op.join(project_root, 'results/reports')  # RDS folder for reports
   
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
