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
    
Issues/ contributions to community:

    
Questions:


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
input_suffix = 'ica'
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

input_fname = op.join(deriv_folder, bids_path.basename + '_' + input_suffix + extension)  # prone to change if annotation worked for eeg brainvision
deriv_fname = op.join(deriv_folder, bids_path.basename + '_' + deriv_suffix + extension) 

# read annotated data
raw = read_raw_bids(bids_path=bids_path, verbose=False, 
                     extra_params={'preload':True})  # read raw for events and event ids only

# read raw and events file
raw_ica = mne.io.read_raw_fif(input_fname, allow_maxshield=True,
                              verbose=True, preload=True)

events, events_id = mne.events_from_annotations(raw, event_id='auto')

# Set the peak-peak amplitude threshold for trial rejection.
""" subject to change based on data quality"""
reject = dict(grad=5000e-13,  # T/m (gradiometers)
              mag=5e-12,      # T (magnetometers)
              #eog=150e-6      # V (EOG channels)
              )

# Make epochs (2 seconds centered on stim onset)
metadata, _, _ = mne.epochs.make_metadata(
                events=events, event_id=events_id, 
                tmin=-1.5, tmax=1, 
                sfreq=raw_ica.info['sfreq'])

epochs = mne.Epochs(raw_ica, events, events_id,   # select events_picks and events_picks_id                   
                    metadata=metadata,            # if only interested in specific events (not all)
                    tmin=-0.8, tmax=1.2,
                    baseline=None, proj=True, picks='all', 
                    detrend=1, event_repeated='drop',
                    reject=reject, reject_by_annotation=True,
                    preload=True, verbose=True)

# Defie epochs we care about
conds_we_care_about = ["cue_onset_right", "cue_onset_left", "stim_onset", "response_press_onset"] # TODO:discuss with Ole
epochs.equalize_event_counts(conds_we_care_about)  # this operates in-place

# Save the epoched data 
epochs.save(deriv_fname, overwrite=True)

############################### Check-up plots ################################
# Plotting to check the raw epoch
epochs['cue_onset_left'].plot(events=events, event_id=events_id, n_epochs=10)  # shows all the events in the epoched data that's based on 'cue_onset_left'
epochs['cue_onset_right'].plot(events=events, event_id=events_id, n_epochs=10) 

# plot amplitude on heads
times_to_topomap = [-.1, .1, .8, 1.1]
epochs['cue_onset_left'].average(picks=['meg']).plot_topomap(times_to_topomap)  # title='cue onset left (0 sec)'
epochs['cue_onset_right'].average(picks=['meg']).plot_topomap(times_to_topomap)  # title='cue onset right (0 sec)'

# Topo plot evoked responses
evoked_obj_topo_plot = [epochs['cue_onset_left'].average(picks=['grad']), epochs['cue_onset_right'].average(picks=['grad'])]
mne.viz.plot_evoked_topo(evoked_obj_topo_plot, show=True)

fig_bads = epochs.plot_drop_log()  # rejected epochs
###############################################################################

# Plots the average of one epoch type - pick best sensors for report
epochs['cue_onset_left'].average(picks=['meg']).copy().filter(1,60).plot()
epochs['cue_onset_right'].average(picks=['meg']).copy().filter(1,60).plot()

# Plots to save
fig_right = epochs['cue_onset_right'].copy().filter(0.0,30).crop(-.1,1.2).plot_image(
    picks=['MEG1932'],vmin=-100,vmax=100)  # event related field image
fig_left = epochs['cue_onset_left'].copy().filter(0.0,30).crop(-.1,1.2).plot_image(
    picks=['MEG2332'],vmin=-100,vmax=100)  # event related field image

if summary_rprt:
    report_root = op.join(mTBI_root, r'results-outputs/mne-reports')  # RDS folder for reports
    
    if not op.exists(op.join(report_root , 'sub-' + subject, 'task-' + task)):
        os.makedirs(op.join(report_root , 'sub-' + subject, 'task-' + task))
    report_folder = op.join(report_root , 'sub-' + subject, 'task-' + task)

    report_fname = op.join(report_folder, 
        f'mneReport_sub-{subject}_{task}_1.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'report_preproc_{task}_1.html')

    report = mne.open_report(report_fname)

    report.add_figure(fig=fig_right, title='cue right',
                    caption='evoked response on one left sensor (MEG1943)', 
                    tags=('epo'),
                    section='epocheds'
                    )
    report.add_figure(fig=fig_left, title='cue left',
                    caption='evoked response on one right sensor (MEG2522)', 
                    tags=('epo'),
                    section='epocheds' 
                    )   
    report.save(report_fname, overwrite=True, open_browser=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
