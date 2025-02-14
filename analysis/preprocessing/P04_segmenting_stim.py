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


# BIDS settings: fill these out 
subject = '107'
session = '01'
task = 'SpAtt'
run = '01'  # change this for subjects with two stim or two no-stim segments
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
input_suffix = 'ica'
no_stim_suffix = 'no-stim_ica'
stim_suffix = 'stim_ica'
extension = '.fif'

pilot = False  # is it pilot data or real data?
summary_rprt = True  # do you want to add evokeds figures to the summary report?
platform = 'mac'  # are you using 'bluebear' or 'mac'?
test_plot = False

if platform == 'bluebear':
    rds_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    camcan_dir = '/rds/projects/q/quinna-camcan/dataman/data_information'
elif platform == 'mac':
    rds_dir = '/Volumes/jenseno-avtemporal-attention'
    camcan_dir = '/Volumes/quinna-camcan/dataman/data_information'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')

# Specify specific file names
ROI_dir = op.join(project_root, 'results/lateralisation-indices')
bids_root = op.join(project_root, 'data', 'BIDS')

# for bear outage
# bids_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/data/BIDS'

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
stimulation_cropped_time = {"sub-107_no-stim": [15, 974],
                            "sub-107_stim": [1000, 1845],
                            "sub-108_no-stim": [8, 890],
                            "sub-108_stim": [930, 1882],
                            "sub-110_no-stim": [905, 1711],
                            "sub-110_stim": [0, 840],
                            "sub-102_no-stim": [0, 965],
                            "sub-102_stim": [1497, 2244],
                            "sub-101_no-stim": [0, 360, 515, 865],
                            "sub-101_stim": [1144, 1900],
                            "sub-112_no-stim": [878, 1289, 1870, 2280],
                            "sub-112_stim": [244, 650, 2708, 3117],
                            "sub-103_no-stim": [785, 1113, 1380, 1715],
                            "sub-103_stim": [72, 476, 1862, 2182],
                            }

# Crop and save segments separately
if subject in ['101', '112', '103']:  # there is some stimulation in the break between two no-stim blocks
    no_stim_fragements = [raw_ica.copy().crop(tmin=stimulation_cropped_time[f'sub-{subject}_no-stim'][0], 
                          tmax=stimulation_cropped_time[f'sub-{subject}_no-stim'][1]), 
                          raw_ica.copy().crop(tmin=stimulation_cropped_time[f'sub-{subject}_no-stim'][2], 
                          tmax=stimulation_cropped_time[f'sub-{subject}_no-stim'][3])]
    no_stim_segment = mne.concatenate_raws(no_stim_fragements)

else:
    no_stim_segment = raw_ica.copy().crop(tmin=stimulation_cropped_time[f'sub-{subject}_no-stim'][0], 
                                      tmax=stimulation_cropped_time[f'sub-{subject}_no-stim'][1])
    
fig_no_stim_psd = no_stim_segment.compute_psd(fmin=0.1, fmax=200).plot()  # double check and save if ok
no_stim_segment.save(no_stim_fname, overwrite=True)

if subject in ['112', '103']:  # there is some stimulation in the break between two stim blocks
    stim_fragements = [raw_ica.copy().crop(tmin=stimulation_cropped_time[f'sub-{subject}_stim'][0], 
                          tmax=stimulation_cropped_time[f'sub-{subject}_stim'][1]), 
                          raw_ica.copy().crop(tmin=stimulation_cropped_time[f'sub-{subject}_stim'][2], 
                          tmax=stimulation_cropped_time[f'sub-{subject}_stim'][3])]
    stim_segment = mne.concatenate_raws(stim_fragements)
else:

    stim_segment = raw_ica.copy().crop(tmin=stimulation_cropped_time[f'sub-{subject}_stim'][0],
                                   tmax=stimulation_cropped_time[f'sub-{subject}_stim'][1])
    
fig_stim_psd = stim_segment.compute_psd(fmin=0.1, fmax=200).plot() 
stim_segment.save(stim_fname, overwrite=True)


if summary_rprt:
    report_root = op.join(project_root, 'derivatives/reports')  
   # for bear outage
    # report_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/derivatives/reports' # only for bear outage time

    report_folder = op.join(report_root , 'sub-' + subject)

    report_fname = op.join(report_folder, 
                        f'sub-{subject}_070225.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub-{subject}_070225.html')
    
    report = mne.open_report(report_fname)
    report.add_figure(fig=fig_no_stim_psd, title='no stimulation psd',
                    caption='psd of no stimulation segment after ica', 
                    tags=('stim'),
                    section='stim'
                    ) 
    report.add_figure(fig=fig_stim_psd, title='stimulation psd',
                    caption='psd of stimulated segment after ica', 
                    tags=('stim'),
                    section='stim'
                    )     
    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
