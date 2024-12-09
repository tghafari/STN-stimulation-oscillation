# -*- coding: utf-8 -*-
"""
===============================================
07. Run and apply ICA

This code will run ICA to find occular and cardiac
artifacts: 1. decomposition, 2. manual identification,
3. project out

written by Tara Ghafari
adapted from flux pipeline
==============================================
ToDos:
    1) 
    2) 
    
Issues:
    1) read_raw_bids doesn't read annotated
    data
    
Questions:
    1) 

"""

import os.path as op
import os
import numpy as np

import mne
from mne.preprocessing import ICA
from copy import deepcopy
from mne_bids import BIDSPath, read_raw_bids

# BIDS settings: fill these out 
subject = '110'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
deriv_suffix = 'ica'
extension = '.fif'

summary_rprt = True  # do you want to add evokeds figures to the summary report?
platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?

if platform == 'bluebear':
    rds_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    camcan_dir = '/rds/projects/q/quinna-camcan/dataman/data_information'
elif platform == 'mac':
    rds_dir = '/Volumes/jenseno-avtemporal-attention-2'
    camcan_dir = '/Volumes/quinna-camcan/dataman/data_information'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
bids_root = op.join(project_root, 'data', 'BIDS')

# Specify specific file names
bids_path = BIDSPath(subject=subject, session=session,
                     task=task, run=run, root=bids_root, 
                     datatype ='eeg', suffix=eeg_suffix)
deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results
if not op.exists(deriv_folder):
    os.makedirs(deriv_folder)
deriv_fname = op.join(deriv_folder, bids_path.basename + '_' + deriv_suffix + extension)  # prone to change if annotation worked for eeg brainvision

# read annotated data
raw = read_raw_bids(bids_path=bids_path, verbose=False, 
                        extra_params={'preload':True})
raw.plot()  # just scan through the data to ensure quality measures have been made in the previous step

# Resample and filtering
"""
we down sample the data in order to make ICA run faster, 
highpass filter at 1Hz to remove slow drifts and lowpass 40Hz
because that's what we need
"""
raw_resmpld = deepcopy(raw).resample(200).filter(0.1, 40)

# Apply ICA and identify artifact components
ica = ICA(method='fastica', random_state=97, n_components=30, verbose=True)
ica.fit(raw_resmpld, verbose=True)
ica.plot_sources(raw_resmpld, title='ICA')
ica.plot_components()

ICA_rej_dic = {f'sub-{subject}_ses-{session}':[0, 1]} # manually selected bad ICs or from sub config file 
artifact_ICs = ICA_rej_dic[f'sub-{subject}_ses-{session}']
"""
list bad ICA components for all participants:
{
'pilot_BIDS/sub-01_ses-01_run-01': [2, 5],  # 2:blink, 5:saccades
'pilot_BIDS/sub-02_ses-01_run-01': [1, 4],  # 1:blink, 4:saccades
'BIDS/sub-01_ses-01_run-01': [0, 1, 2], # 0:blink, 1:saccades, 2:blink/saccades
'BIDS/sub-02_ses-01_run-01': [0, 1, 2, 3, 4], # 0:blink, 1:saccades, 2:blink/saccades, 3&4: empty
'BIDS/sub-05_ses-01_run-01': [0, 1, 8, 58, 59], # don't know-almost all look terrible
'BIDS/sub-107_ses-01_run-01': [28], # maybe eye movement?  
'BIDS/sub-108_ses-01_run-01': [1, 13], # don't know-almost all look terrible
'BIDS/sub-108_ses-01_run-01': [0, 1], # 0:blink, 4:saccades
} """

# Double check the manually selected artifactual ICs
""" Plot original data against reconstructed 
  signal excluding artifact ICs + Ic properties"""

for exc in np.arange(len(artifact_ICs)):
    ica.plot_overlay(raw_resmpld, exclude=[artifact_ICs[exc]], picks='eeg')  
  
ica.plot_overlay(raw_resmpld, exclude=artifact_ICs, picks='eeg')  # all
ica.plot_properties(raw_resmpld, picks=artifact_ICs)

# Exclude ICA components
ica.exclude = artifact_ICs
raw_ica = deepcopy(raw)
ica.apply(raw_ica)

# Save the ICA cleaned data
raw_ica.save(deriv_fname, overwrite=True)

# plot a few frontal channels before and after ICA
chs = ['C5', 'O2']
ch_idx = [raw.ch_names.index(ch) for ch in chs]
raw.plot(order=ch_idx, duration=5, title='before')
raw_ica.plot(order=ch_idx, duration=5, title='after')

# only add excluded components to the report
fig_ica = ica.plot_components(picks=artifact_ICs, title='removed components')

# Filter data for the report
if summary_rprt:
    report_root = op.join(project_root, 'derivatives/reports')  # RDS folder for reports
   
    if not op.exists(op.join(report_root , 'sub-' + subject)):
        os.makedirs(op.join(report_root , 'sub-' + subject))
    report_folder = op.join(report_root , 'sub-' + subject)

    report_fname = op.join(report_folder, 
                        f'sub-{subject}_091224.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub-{subject}_091224.html')
    
    report = mne.open_report(report_fname)
    report.add_figure(fig_ica, 
                      title="removed ICA components (filtered:0.1-40)",
                      caption="removed ICA components: eye movement(?)",
                      tags=('ica'), 
                      image_format="PNG")
    report.add_raw(raw=raw_ica.filter(0.3, 100), title='raw after ICA (avg reference)', 
                   psd=True, 
                   butterfly=False, 
                   tags=('ica'))
    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks


