# -*- coding: utf-8 -*-
"""
===============================================
08. Epoching raw data based on conditions

This code will epoch continuous EEG based
on conditions that are annotated in the
data and generates  an HTML report about epochs.

written by Tara Ghafari
adapted from flux pipeline
==============================================
ToDos:
    1) which epochs to keep?
    
Questions:

"""

import os.path as op
import os
import numpy as np

import mne
from mne.preprocessing import ICA
from copy import deepcopy
from mne_bids import BIDSPath, read_raw_bids
from autoreject import get_rejection_threshold



# BIDS settings: fill these out 
subject = '01'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
input_suffix = 'ica'
deriv_suffix = 'epo'
extension = '.fif'

pilot = True  # is it pilot data or real data?
summary_rprt = False  # do you want to add evokeds figures to the summary report?
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
bids_root = op.join(project_root, 'Data', 'BIDS')
bids_path = BIDSPath(subject=subject, session=session,
                     task=task, run=run, root=bids_root, 
                     datatype ='eeg', suffix=eeg_suffix)
deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results
input_fname = op.join(deriv_folder, bids_path.basename + '_' + input_suffix + extension)
deriv_fname = op.join(deriv_folder, bids_path.basename + '_' + deriv_suffix + extension)  # prone to change if annotation worked for eeg brainvision

# read annotated data
raw = read_raw_bids(bids_path=bids_path, verbose=False, 
                        extra_params={'preload':True})

# Mark bad channels again- for whatever reason the bad channels are not annotated even though they are in the raw_ica from oy_run_apply_ICA
original_bads = deepcopy(raw.info["bads"])
raw.info["bads"].append("FCz")  # add a single channel
# raw.info["bads"].extend(["EEG 051", "EEG 052"])  # add a list of channels - should there be more than one channel to drop

events, events_id = mne.events_from_annotations(raw, event_id='auto')

# Make epochs (1.7 seconds on cue onset)
epochs = mne.Epochs(raw, 
                    events, 
                    events_id,   # select events_picks and events_picks_id                   
                    tmin=-0.5, 
                    tmax=1.7,
                    baseline=None, 
                    proj=True, 
                    picks='all', 
                    detrend=1, 
                    event_repeated='merge',
                    reject=None,  # we'll reject after calculating the threshold
                    reject_by_annotation=True,
                    preload=True, 
                    verbose=True)

reject = get_rejection_threshold(epochs, 
                                 decim=10)

# Drop bad epochs based on peak-to-peak magnitude
print(f"\n\n Numer of epochs BEFORE rejection: {len(epochs.events)} \n\n")
epochs.drop_bad(reject=reject)
print(f"\n\n Numer of epochs AFTER rejection: {len(epochs.events)} \n\n")

# Save the epoched data 
fig_bads = epochs.plot_drop_log()  # rejected epochs
epochs.save(deriv_fname, overwrite=True)

if test_plot:
    ############################### Check-up plots ################################
    # Plotting to check the raw epoch
    epochs['cue_onset_left'].plot(events=events, event_id=events_id, n_epochs=10)  # shows all the events in the epoched data that's based on 'cue_onset_left'
    epochs['cue_onset_right'].plot(events=events, event_id=events_id, n_epochs=10) 

    # plot amplitude on heads
    times_to_topomap = [-.1, .1, .8, 1.1]
    epochs['cue_onset_left'].average().plot_topomap(times_to_topomap)  # title='cue onset left (0 sec)'
    epochs['cue_onset_right'].average().plot_topomap(times_to_topomap)  # title='cue onset right (0 sec)'

    # Topo plot evoked responses
    evoked_obj_topo_plot = [epochs['cue_onset_left'].average(), epochs['cue_onset_right'].average()]
    mne.viz.plot_evoked_topo(evoked_obj_topo_plot, show=True)

    ###############################################################################

    # Plots the average of one epoch type - pick best sensors for report
    epochs['cue_onset_left'].average().copy().filter(1,60).plot()
    epochs['cue_onset_right'].average().copy().filter(1,60).plot()

if summary_rprt:

    report_root = op.join(project_root, 'results/reports')  
    if not op.exists(op.join(report_root , 'sub-' + subject)):
        os.makedirs(op.join(report_root , 'sub-' + subject))
    report_folder = op.join(report_root , 'sub-' + subject)

    report_fname = op.join(report_folder, 
                        f'sub-{subject}_preproc_2.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub-{subject}_preproc_2.html')

    report = mne.open_report(report_fname)

    report.add_figure(fig=fig_bads, title='dropped epochs',
                    caption='epochs dropped and reason', 
                    tags=('epo'),
                    section='epocheds'
                    )
    report.save(report_fname, overwrite=True, open_browser=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
