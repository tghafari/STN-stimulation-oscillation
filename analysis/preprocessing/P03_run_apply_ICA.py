# -*- coding: utf-8 -*-
"""
===============================================
07. Run and apply ICA

This code will run ICA to find occular and cardiac
artifacts: 

    1. resampling and running ICA
    2. finding single channels that are associated
    with bad components
    3. reject those channels
    4. apply common reference again
    5. run ICA again
    6. project bad components out

written by Tara Ghafari
adapted from flux pipeline
==============================================

"""

import os.path as op
import os
import numpy as np

import mne
from mne.preprocessing import ICA
from copy import deepcopy
from mne_bids import BIDSPath, read_raw_bids

# BIDS settings: fill these out 
subject = '105'
session = '01'
task = 'SpAtt'
run = '01'
input_suffix = 'ann'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
deriv_suffix = 'ica'
extension = '.fif'

summary_rprt = True  # do you want to add evokeds figures to the summary report?
platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
montage = False  # 'standard' for setting standard montage, 'costume' for Sirui's montage, False for not doing anything with montage

if platform == 'bluebear':
    rds_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    camcan_dir = '/rds/projects/q/quinna-camcan/dataman/data_information'
elif platform == 'mac':
    rds_dir = '/Volumes/jenseno-avtemporal-attention'
    camcan_dir = '/Volumes/quinna-camcan/dataman/data_information'

# project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
# bids_root = op.join(project_root, 'data', 'BIDS')

# for bear outage
project_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD'  # only for bear outage time
bids_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/data/BIDS'

# Specify specific file names
bids_path = BIDSPath(subject=subject, session=session,
                     task=task, run=run, root=bids_root, 
                     datatype ='eeg', suffix=eeg_suffix)
deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results
if not op.exists(deriv_folder):
    os.makedirs(deriv_folder)

input_fname = op.join(deriv_folder, bids_path.basename + '_' + input_suffix + extension)
deriv_fname = op.join(deriv_folder, bids_path.basename + '_' + deriv_suffix + extension)  # prone to change if annotation worked for eeg brainvision
montage_fname = op.join(project_root, 'data', 'data-organised', 'new-64.bvef')

# Read annotated data
raw = mne.io.read_raw_fif(input_fname, verbose=True, preload=True)

########################## BAD CHANNEL REJECTION ######################################

## 1. Scroll and psd to remove any other bad channels after filtering
"""Mark bad channels on the plots"""
raw_filtered = raw.copy().filter(l_freq=1, h_freq=100)  # filter only for plotting now 
raw_filtered.plot()  # mark bad channels after filtering stimulation frequency
raw.info["bads"] = raw_filtered.info["bads"]  # add marked bad channels to raw

"""Reject channels that are different than others"""
n_fft = int(raw.info['sfreq']*2)  # to ensure window size = 2sec
raw_filtered.compute_psd(n_fft=n_fft,  # default method is welch here (multitaper for epoch)
                         n_overlap=int(n_fft/2),
                         fmax=105).plot()  
"""Channels marked bad in this stage:
 {'sub-110': 'Fp1 FCz',
 'sub-107':'C2 FC1 F3 AF7',
 'sub-102': 'FT7 F3',
 'sub-104': 'F2 F4 F3 AF7 C4 FC2 AF3 F8',  I didn't remove these!

}"""

## 2. Mark bad channels before ICA
original_bads = deepcopy(raw.info["bads"]) 
print(f'these are original bads: {original_bads}')
user_list = input('Any other bad channels from psd? name of channel, e.g. FT10 T9 (separate by space) or return.')
bad_channels = user_list.split()

if len(bad_channels) > 0:
    raw.copy().pick(bad_channels).compute_psd().plot()  # double check bad channels
    if len(bad_channels) == 1:
        print('one bad channel removing')
        raw.info["bads"].append(bad_channels[0])  # add a single channel
    else:
        print(f'{len(bad_channels)} bad channels removing')
        raw.info["bads"].extend(bad_channels)  # add a list of channels - should there be more than one channel to drop

if not montage:
    print('not setting standard or costume montage as default.')
elif montage == 'standard':  # set standard montage - easycap-M1
    """it is important to bring the montage to the standard space. Otherwise the 
    ICA and PSDs look weird."""
    raw.plot_sensors(show_names=True)
    montage = mne.channels.make_standard_montage("easycap-M1")
    montage.plot()  # 2D
    raw.set_montage(montage, verbose=False)
    raw.plot_sensors(show_names=True)

elif montage == 'costume':  # set costume montage from Sirui's layout
    raw.plot_sensors(show_names=True)
    montage = mne.channels.read_custom_montage(montage_fname)
    montage.plot()  # 2D
    raw.set_montage(montage, verbose=False)
    raw.plot_sensors(show_names=True)

## 3. Resample and filter for ICA
"""
we down sample the data in order to make ICA run faster, 
highpass filter at 1Hz to remove slow drifts and lowpass 40Hz
because that's what we need
"""
raw_resmpld = raw.copy().pick('eeg').resample(200).filter(1, 40) # l_freq=1 recommended by mne

# Apply ICA and identify bad channels
ica = ICA(method='fastica', random_state=97, n_components=30, verbose=True)
ica.fit(raw_resmpld, reject_by_annotation=True, verbose=True)
raw.plot_sensors(show_names=True)
ica.plot_components()

# Take another look at bad channels and remove
raw.plot() # Mark bad channels from ICA here on raw.plot() 'CP6 P1 C2 P6

"""
list bad channels for all participants:
bad ICA channels in last line
{
pilot_BIDS/sub-01_ses-01_run-01: ["FCz"],
pilot_BIDS/sub-02_ses-01_run-01: [],
BIDS/sub-01_ses-01_run-01: ["T7", "FT10"],
BIDS/sub-02_ses-01_run-01: ["TP10"],
BIDS/sub-05_ses-01_run-01: ["almost all channels look terrible in psd"],
BIDS/sub-107_ses-01_run-01: ["AFz", "C6", "AF4", "P3", "Pz", "F1"], 
BIDS/sub-108_ses-01_run-01: ["FT9", "T8", "T7"],
BIDS/sub-110_ses-01_run-01: ['TP9', 'TP10', 'Fp1', 'FCz', 
                            'AF4', 'Pz', 'F6', 'FT7'],
BIDS/sub-102_ses-01_run-01: ['TP9', 'TP10', 'Fp1', 'F7', 'Fp2', 'F8', 'TP7', 'PO7', 'F6', 'F5', 
                            'FC6', 'AF7', 'AF3', 'FT7', 'F3', 'C5'],
BIDS/sub-103_ses-01_run-01: ['TP9', 'TP10', 'P5', 'PO7', 'AF4', 'TP8', 'FC1',
                             'AFz','F6', 'P1', 'P6', 'F4, 'AF3']
BIDS/sub-101_ses-01_run-01: ['TP9', 'TP10', 'CP5', 'P7', 'TP7', 'P5', 'C5', 'P6', 'PO8', 'PO7', 'P8', 'F3', 
                            'FC6', 'F5', 'TP8']
BIDS/sub-112_ses-01_run-01: ['TP10', 'Fp1', 'CP6', 'FC5', 'AF8', 'Fp2', 'F5', 
                            'AF7', 'F8', 'AF3', 'FC4', 'F6', 'F3', 'Cz', 'CPz', 
                            'Pz', 'P1', 'FC6', 'FC1']
BIDS/sub-104_ses-01_run-01: ['TP9', 'TP10', 'Fp1', 'F7', 'CP6', 'FC5', 'CP5', 'FT7', 'F5', 'TP7',
                             'C5', 'C6', 'AF8', 'F6', 'Fp2', 'P7', 'PO7', 'F8',
                             'FC1' 'C3']
 BIDS/sub-105_ses-01_run-01: ['TP9', 'TP10', 'Fp1', 'P7', 'AF7', 'TP7', 'F8', 'FC6', 'FT8', 
                             'PO8', 'AF4', 'AFz', 'C5',
                             'AF3']                            

} """

del raw_resmpld, ica  # free up memory

## 5. Rereference to average
"""If channels distrubited evenly, do average reference (eeglab resources)
do not average reference before removing all the bad channels."""
raw.set_eeg_reference(ref_channels="average")

##################################### MAIN ICA ######################################
"""This is the ica that will be applied to the data. You can redo the previous steps
as many times as you want."""
# Run ica again after bad channel rejection
raw_resmpld = raw.copy().pick('eeg').resample(200).filter(0.5, 40) # l_freq=1 recommended by mne
ica = ICA(method='fastica', random_state=97, n_components=30, verbose=True)
ica.fit(raw_resmpld, reject_by_annotation=True, verbose=True)
ica.plot_sources(raw_resmpld, title='ICA')
ica.plot_components()

ICA_rej_dic = {f'sub-{subject}_ses-{session}':[3, 6]} # manually selected bad ICs or from sub config file 
artifact_ICs = ICA_rej_dic[f'sub-{subject}_ses-{session}'] #6
"""
list bad ICA components for all participants:
{
'pilot_BIDS/sub-01_ses-01_run-01': [2, 5],  # 2:blink, 5:saccades
'pilot_BIDS/sub-02_ses-01_run-01': [1, 4],  # 1:blink, 4:saccades
'BIDS/sub-01_ses-01_run-01': [0, 1, 2], # 0:blink, 1:saccades, 2:blink/saccades
'BIDS/sub-02_ses-01_run-01': [0, 1, 2, 3, 4], # 0:blink, 1:saccades, 2:blink/saccades, 3&4: empty
'BIDS/sub-05_ses-01_run-01': [0, 1, 8, 58, 59], # don't know-almost all look terrible
'BIDS/sub-107_ses-01_run-01': [0, 3, 5], saccade and blink  
'BIDS/sub-108_ses-01_run-01': [1, 13], # don't know-almost all look terrible
'BIDS/sub-110_ses-01_run-01': [1, 2], # 1:blink, 2:saccades
'BIDS/sub-102_ses-01_run-01': [0, 1], # 0:blink, 4:saccades  
'BIDS/sub-103_ses-01_run-01': [0, 1], # 0:blink, 4:saccades  
'BIDS/sub-101_ses-01_run-01': [0, 1, 3, 4, 5], # 0:saccade, 1,3,4,5:blink  !this participant has blinked too many times!
'BIDS/sub-112_ses-01_run-01': [0, 3, 4, 15, 19, 28, 29], # 0:blink, 4:saccades
'BIDS/sub-104_ses-01_run-01': [0, 4], # 0:blink, 4:saccades  
'BIDS/sub-104_ses-01_run-01': [3, 6], # 3,6: based on overlay plot, couldn't find blinks/saccades 


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

# only add excluded components to the report
fig_ica = ica.plot_components(picks=artifact_ICs, title='removed components')

# Filter data for the report
if summary_rprt:
    report_root = op.join(project_root, 'derivatives/reports')  # RDS folder for reports

    if not op.exists(op.join(report_root , 'sub-' + subject)):
        os.makedirs(op.join(report_root , 'sub-' + subject))
    report_folder = op.join(report_root , 'sub-' + subject)

    report_fname = op.join(report_folder, 
                        f'sub-{subject}_130325.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub-{subject}_130325.html')
    
    report = mne.open_report(report_fname)
    report.add_figure(fig_ica, 
                      title="removed ICA components (filtered:0.5-40)",
                      caption="removed ICA components: eye movement(?)",
                      tags=('ica'), 
                      image_format="PNG")
    report.add_raw(raw=raw_ica.filter(0.1, 100), title='raw after ICA (avg reference)', 
                   psd=True, 
                   butterfly=False, 
                   tags=('ica'))
    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks


