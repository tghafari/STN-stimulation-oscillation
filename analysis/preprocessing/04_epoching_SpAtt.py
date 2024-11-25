# -*- coding: utf-8 -*-
"""
===============================================
08. Epoching raw data based on conditions

This code will epoch continuous EEG based
on conditions that are annotated in the
data and generates  an HTML report about epochs.

IMO, cleanest way is to epoch based on all events,
calculate reject threshold, find channel with most 
bad epochs, remove channel, calculate threshold 
again, apply on epochs.

written by Tara Ghafari
adapted from flux pipeline
==============================================
ToDos:

"""

import os.path as op
import os
import numpy as np

import mne
from mne.preprocessing import ICA
from copy import deepcopy
from mne_bids import BIDSPath
from autoreject import get_rejection_threshold

def segment_epoching(stim):
    if stim:
        input_fname = op.join(deriv_folder, bids_path.basename
                               + '_' + stim_suffix + '_' + input_suffix + extension)
    else:
        input_fname = op.join(deriv_folder, bids_path.basename
                               + '_' + no_stim_suffix + '_' + input_suffix + extension)

    # Read ica data
    segmented_ica = mne.io.read_raw_fif(input_fname, verbose=True, preload=True)
    segmented_ica.filter(l_freq=0.1, h_freq=100),  # get rid of stim frequency before epoching, otherwise too many bad channels

    print(f'double checking bad channels: {segmented_ica.info['bads']}')

    events, events_id = mne.events_from_annotations(segmented_ica, event_id='auto')

    # Make epochs (-0.7 t0 1.7 seconds on cue onset or all events)
    epochs = mne.Epochs(segmented_ica,
                        events, 
                        events_id,  # events_id for all events, events_id_to_consider for only cue onsets                  
                        tmin=-0.7, 
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
    return epochs, events, events_id
 
def finding_bad_channel(epochs):

    reject_temp = get_rejection_threshold(epochs)  # removed detrend=10 to ensure no antialiasing happens                                     
    # Drop bad epochs based on peak-to-peak magnitude
    epochs_temp = deepcopy(epochs)  # this is temporary to find bad channels
    epochs_temp.drop_bad(reject=reject_temp)

    # Check if a few channles have most of bad epochs and mark them as bad
    # instead of dropping epochs 
    fig_bads_temp = epochs_temp.plot_drop_log()  # rejected epochs
    
    return fig_bads_temp

def cleaning_epochs(stim, epochs):
    if stim:
        deriv_fname = op.join(deriv_folder, bids_path.basename 
                               + '_' + stim_suffix + '_' + deriv_suffix + extension)  
    else:
        deriv_fname = op.join(deriv_folder, bids_path.basename 
                               + '_' + no_stim_suffix + '_' + deriv_suffix + extension)  
    # Might need to reconsider this section. might not even be needed
    bad_channels = input('Are there any bad channels after rejecting bad epochs? name of channel, e.g. FT10:  (return if none). n.b. Make sure to remove from both stim and no-stim')
    # Mark bad channels before ICA
    if len(bad_channels) > 0:
        original_bads = deepcopy(epochs.info["bads"])
        print(f'These are the original bads: {original_bads}')
        bad_chs = [bad_channels] #["FT10"]  
        print(f'{len(bad_chs)}')
        epochs.copy().pick(bad_chs).compute_psd(fmin=0.1, fmax=100).plot()  # double check bad channels
        if len(bad_chs) == 1:
            print('one bad channel removing')
            epochs.info["bads"].append(bad_chs[0])  # add a single channel
        elif len(bad_chs) > 1:
            print(f'{len(bad_chs)} bad channels removing')
            epochs.info["bads"].extend(bad_chs)  # add a list of channels - should there be more than one channel to drop

    reject = get_rejection_threshold(epochs)  # reject without bad channels                               
    print(f"\n\n Numer of epochs BEFORE rejection: {len(epochs.events)} \n\n")
    epochs.drop_bad(reject=reject)
    print(f"\n\n Numer of epochs AFTER rejection: {len(epochs.events)} \n\n")
    fig_bads = epochs.plot_drop_log()  # rejected epochs
    fig_psd = epochs.copy().compute_psd(fmin=0.1,fmax=100,method='welch',n_fft=int(2*epochs.info['sfreq'])).plot()
    # Save the epoched data
    epochs.save(deriv_fname, overwrite=True)

    return fig_bads, fig_psd, epochs

# BIDS settings: fill these out 
subject = '107'
session = '01'
task = 'SpAtt'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
input_suffix = 'ica'
stim_suffix = 'stim'
no_stim_suffix = 'no-stim'
deriv_suffix = 'epo'
extension = '.fif'

runs = ['01']
stim_segments_ls = [False, True]

pilot = False  # is it pilot data or real data?
test_plot = False
platform = 'mac'  # are you using 'bluebear' or 'mac'?

if platform == 'bluebear':
    rds_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    camcan_dir = '/rds/projects/q/quinna-camcan/dataman/data_information'
elif platform == 'mac':
    rds_dir = '/Volumes/jenseno-avtemporal-attention'
    camcan_dir = '/Volumes/quinna-camcan/dataman/data_information'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
if pilot:
    bids_root = op.join(project_root, 'data', 'pilot-BIDS')
else:
    bids_root = op.join(project_root, 'data', 'BIDS')

# Epoch stim segments and add to report
report_root = op.join(project_root, 'derivatives/reports')  
report_folder = op.join(report_root , 'sub-' + subject)

report_fname = op.join(report_folder, 
                    f'sub-{subject}_preproc.hdf5')    # it is in .hdf5 for later adding images
html_report_fname = op.join(report_folder, f'sub-{subject}_preproc.html')

report = mne.open_report(report_fname)

for stim in stim_segments_ls:
    for run in runs:
        bids_path = BIDSPath(subject=subject, session=session,
                     task=task, run=run, root=bids_root, 
                     datatype ='eeg', suffix=eeg_suffix)
        deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results

        epochs, events, events_id = segment_epoching(stim)
        fig_bads_temp = finding_bad_channel(epochs)
        fig_bads, fig_psd, epochs = cleaning_epochs(stim, epochs)
        report.add_figure(fig=fig_bads, title=f'stim: {stim}, dropped epochs',
                    caption=f'stim: {stim}: epochs dropped- no bad channels', 
                    tags=('epo'),
                    section='stim'
                    )
        report.add_figure(fig=fig_psd, title=f'stim: {stim}, psd after dropped',
                    caption=f'stim: {stim}, psd with bad epochs dropped and no bad channels', 
                    tags=('epo'),
                    section='stim'
                    ) 

report.save(report_fname, overwrite=True, open_browser=True)
report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks


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

    # Plots the average of one epoch type - pick best sensors for report
    epochs['cue_onset_left'].average().copy().filter(1,60).plot()
    epochs['cue_onset_right'].average().copy().filter(1,60).plot()





