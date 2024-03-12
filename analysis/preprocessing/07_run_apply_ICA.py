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
from mne_bids import BIDSPath, read_raw_bids

# BIDS settings: fill these out 
subject = '01'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
deriv_suffix = 'ica'

pilot = True  # is it pilot data or real data?
summary_rprt = True  # do you want to add evokeds figures to the summary report?
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
if pilot:
    data_root = op.join(project_root, 'Data/pilot-data/AO')
else:
    data_root = op.join(project_root, 'Data/real-data')

# Specify specific file names
bids_root = op.join(project_root, 'Data', 'BIDS')
bids_path = BIDSPath(subject=subject, session=session,
                     task=task, run=run, root=bids_root, 
                     datatype ='eeg', suffix=eeg_suffix)
deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results
if not op.exists(deriv_folder):
    os.makedirs(deriv_folder)
deriv_fname = bids_path.basename + '_' + deriv_suffix  # prone to change if annotation worked for eeg brainvision

# read annotated data + resample and filtering
"""
we down sample the data in order to make ICA run faster, 
highpass filter at 1Hz to remove slow drifts and lowpass 40Hz
because that's what we need
"""
raw_ann = read_raw_bids(bids_path=bids_path, verbose=False, 
                        extra_params={'preload':True})
raw_resmpld = raw_ann.copy().resample(200).filter(1, 40)

# Apply ICA and identify artifact components
ica = ICA(method='fastica', random_state=97, n_components=30, verbose=True)
ica.fit(raw_resmpld, verbose=True)
ica.plot_sources(raw_resmpld, title='ICA')
ica.plot_components()

ICA_rej_dic = {f'sub-{subject}_ses-{session}':[3,13]} # manually selected bad ICs or from sub config file 
artifact_ICs = ICA_rej_dic[f'sub-{subject}_ses-{session}']


# Double check the manually selected artifactual ICs
""" Plot original data against reconstructed 
  signal excluding artifact ICs + Ic properties"""

for exc in np.arange(len(artifact_ICs)):
    ica.plot_overlay(raw_resmpld, exclude=[artifact_ICs[exc]], picks='mag')  
  
ica.plot_overlay(raw_resmpld, exclude=artifact_ICs, picks='mag')  # all
ica.plot_properties(raw_resmpld, picks=artifact_ICs)

# Exclude ICA components
ica.exclude = artifact_ICs
raw_ica = raw_ann.copy()
ica.apply(raw_ica)

# Save the ICA cleaned data
raw_ica.save(deriv_fname, overwrite=True)

# plot a few frontal channels before and after ICA
chs = ['MEG0311', 'MEG0121', 'MEG1211', 'MEG1411', 'MEG0342', 'MEG1432']
ch_idx = [raw_ann.ch_names.index(ch) for ch in chs]
raw_ann.plot(order=ch_idx, duration=5, title='before')
raw_ica.plot(order=ch_idx, duration=5, title='after')

# only add excluded components to the report
fig_ica = ica.plot_components(picks=artifact_ICs, title='removed components')

# Filter data for the report
if summary_rprt:
    report_root = op.join(project_root, r'results/reports')  # RDS folder for reports
   
    if not op.exists(op.join(report_root , 'sub-' + subject)):
        os.makedirs(op.join(report_root , 'sub-' + subject))
    report_folder = op.join(report_root , 'sub-' + subject)

    report_fname = op.join(report_folder, 
                        f'sub-{subject}_preproc.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub-{subject}_preproc.html')
    
    report = mne.open_report(report_fname)
    report.add_figure(fig_ica, title="removed ICA components (eog, ecg)",
                      tags=('ica'), image_format="PNG")
    report.add_raw(raw=raw_ica.filter(0, 60), title='raw after ICA', 
                   psd=True, butterfly=False, tags=('ica'))
    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks


report.add_ica(ica, title='ICA components', inst=None)  # for until when we figure out what to do with annotation



