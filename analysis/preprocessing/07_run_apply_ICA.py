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
subject = '01'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
deriv_suffix = 'ica'
extension = '.fif'

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
deriv_fname = op.join(deriv_folder, bids_path.basename + '_' + deriv_suffix + extension)  # prone to change if annotation worked for eeg brainvision

# read annotated data
raw = read_raw_bids(bids_path=bids_path, verbose=False, 
                        extra_params={'preload':True})

# Plot to find bad channels and reject from data
raw.plot(title="raw") 
n_fft = int(raw.info['sfreq']*2)  # to ensure window size = 2
raw.copy().drop_channels(['Fz']).compute_psd(n_fft=n_fft,  # default method is welch here (multitaper for epoch)
                                             n_overlap=int(n_fft/2),  # drop reference is to adjust the scaling of the figure
                                             fmin=0.1, fmax=100).plot()  
                                                                                         
# Mark bad channels before ICA
original_bads = deepcopy(raw.info["bads"])
raw.info["bads"].append("FCz")  # add a single channel
# raw.info["bads"].extend(["EEG 051", "EEG 052"])  # add a list of channels - should there be more than one channel to drop

"""
list bad channels for all participants:
{
sub-01_ses-01_run-01: ['FCz'],
} """

# Resample and filtering
"""
we down sample the data in order to make ICA run faster, 
highpass filter at 1Hz to remove slow drifts and lowpass 40Hz
because that's what we need
"""
raw_resmpld = raw.copy().resample(200).filter(1, 40)

# Apply ICA and identify artifact components
ica = ICA(method='fastica', random_state=97, n_components=30, verbose=True)
ica.fit(raw_resmpld, verbose=True)
ica.plot_sources(raw_resmpld, title='ICA')
ica.plot_components()

ICA_rej_dic = {f'sub-{subject}_ses-{session}':[3,4]} # manually selected bad ICs or from sub config file 
artifact_ICs = ICA_rej_dic[f'sub-{subject}_ses-{session}']
"""
list bad ICA components for all participants:
{
'sub-01_ses-01_run-01': [3, 4],  # 3:saccades, 4:blinks
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
raw_ica = raw.copy()
ica.apply(raw_ica)

# Save the ICA cleaned data
raw_ica.save(deriv_fname, overwrite=True)

# plot a few frontal channels before and after ICA
chs = ['MEG0311', 'MEG0121', 'MEG1211', 'MEG1411', 'MEG0342', 'MEG1432']
ch_idx = [raw.ch_names.index(ch) for ch in chs]
raw.plot(order=ch_idx, duration=5, title='before')
raw_ica.plot(order=ch_idx, duration=5, title='after')

# only add excluded components to the report
fig_ica = ica.plot_components(picks=artifact_ICs, title='removed components')

# Filter data for the report
if summary_rprt:
    report_root = op.join(project_root, 'results/reports')  # RDS folder for reports
   
    if not op.exists(op.join(report_root , 'sub-' + subject)):
        os.makedirs(op.join(report_root , 'sub-' + subject))
    report_folder = op.join(report_root , 'sub-' + subject)

    report_fname = op.join(report_folder, 
                        f'sub-{subject}_preproc_2.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub-{subject}_preproc_2.html')
    
    report = mne.open_report(report_fname)
    report.add_figure(fig_ica, title="removed ICA components (saccade, blink)",
                      tags=('ica'), image_format="PNG")
    report.add_raw(raw=raw_ica.drop_channels(['Fz']).filter(0.3, 100), title='raw after ICA', 
                   psd=True, 
                   butterfly=False, 
                   tags=('ica'))
    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks




