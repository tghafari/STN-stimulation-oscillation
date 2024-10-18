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
    1) which epochs to keep?

Issues: 
    
Questions:

"""

import os.path as op
import os
import numpy as np

import mne
from mne.preprocessing import ICA
from copy import deepcopy
from mne_bids import BIDSPath
from autoreject import get_rejection_threshold


# BIDS settings: fill these out 
subject = '108'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
input_suffix = 'ica'
deriv_suffix = 'epo'
extension = '.fif'

pilot = False  # is it pilot data or real data?
find_bad_chns = True  # do you want to check if there are a few channels that have most of bad epochs and mark them as bad?
summary_rprt = True # do you want to add evokeds figures to the summary report?
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
    bids_root = op.join(project_root, 'data', 'pilot-BIDS')
else:
    bids_root = op.join(project_root, 'data', 'BIDS')

# Specify specific file names
bids_path = BIDSPath(subject=subject, session=session,
                     task=task, run=run, root=bids_root, 
                     datatype ='eeg', suffix=eeg_suffix)
deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results
input_fname = op.join(deriv_folder, bids_path.basename + '_' + input_suffix + extension)
deriv_fname = op.join(deriv_folder, bids_path.basename + '_' + deriv_suffix + extension)  # prone to change if annotation worked for eeg brainvision

# read annotated data
raw_ica = mne.io.read_raw_fif(input_fname, verbose=True, preload=True)
raw_ica.filter(l_freq=0.1, h_freq=100),  # get rid of stim frequency before epoching:
                                         # unfiltered epochs had >%61 bad epochs (sub108)
print(f'double checking bad channels: {raw_ica.info['bads']}')

events, events_id = mne.events_from_annotations(raw_ica, event_id='auto')
events_id_to_consider = {'cue_onset_left': 4, 'cue_onset_right': 5}

# Reject threshold from mne-python website
"""https://mne.tools/stable/auto_tutorials/intro/10_overview.html#sphx-glr-auto-tutorials-intro-10-overview-py"""
reject_mne_python = dict(mag=4000e-15,  # 4000 fT
                    grad=4000e-13,  # 4000 fT/cm
                    eeg=150e-6,  # 150 µV
                    eog=250e-6,  # 250 µV
                    )  
reject_FLUX = dict(#grad=5000e-13,  # unit: T / m (gradiometers)
              #mag=4e-12,  # unit: T (magnetometers)
              eeg=40e-6,  # unit: V (EEG channels)
              #eog=250e-6  # unit: V (EOG channels)
                )
# Neither FLUX nor mne-python thresholds work as they drop all epochs!

# Make epochs (1.7 seconds on cue onset)
epochs = mne.Epochs(raw_ica,
                    events, 
                    events_id,  # select events_picks and events_picks_id                   
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

# Validate stimulation sequence - which segment has a peak at 130Hz in psd? - only works for all event epochs
new_seg = epochs["new_stim_segment"]
for seg in range(len(new_seg)):
    epochs[seg].compute_psd(tmin=0).plot()  # remove leakage from previous segment

if find_bad_chns:  
    reject_temp = get_rejection_threshold(epochs)  # removed detrend=10 to ensure no antialiasing happens                                     

    # Drop bad epochs based on peak-to-peak magnitude
    epochs_temp = deepcopy(epochs)  # this is temporary to find bad channels
    print(f"\n\n Numer of epochs BEFORE rejection: {len(epochs.events)} \n\n")
    epochs_temp.drop_bad(reject=reject_temp)
    print(f"\n\n Numer of epochs AFTER rejection: {len(epochs.events)} \n\n")

    # Check if a few channles have most of bad epochs and mark them as bad
    # instead of dropping epochs 
    fig_bads_temp = epochs_temp.plot_drop_log()  # rejected epochs

    bad_channels = True  # are there any bad channels after rejecting bad epochs?
    # Mark bad channels before ICA
    if bad_channels:
        original_bads = deepcopy(epochs.info["bads"])
        bad_chs = ["FT10"]  # write the name of the bad channels here
        epochs.copy().pick(bad_chs).compute_psd(fmin=0.1, fmax=100).plot()  # double check bad channels
        if len(bad_chs) == 1:
            print('one bad channel removing')
            epochs.info["bads"].append(bad_chs[0])  # add a single channel
        else:
            print(f'{len(bad_chs)} bad channels removing')
            epochs.info["bads"].extend(bad_chs)  # add a list of channels - should there be more than one channel to drop

reject = get_rejection_threshold(epochs)  # reject without bad channels                               
print(f"\n\n Numer of epochs BEFORE rejection: {len(epochs.events)} \n\n")
epochs.drop_bad(reject=reject)
print(f"\n\n Numer of epochs AFTER rejection: {len(epochs.events)} \n\n")
fig_bads = epochs.plot_drop_log()  # rejected epochs
fig_psd = epochs.compute_psd(fmin=0.1,fmax=100,method='welch',n_fft=int(2*epochs.info['sfreq'])).plot()
# Save the epoched data
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

    # Plots the average of one epoch type - pick best sensors for report
    epochs['cue_onset_left'].average().copy().filter(1,60).plot()
    epochs['cue_onset_right'].average().copy().filter(1,60).plot()

# for checking which method to drop bad channels
epochs_cue_before = op.join(project_root,'derivatives/figures/only-cue-epochs-reject-before-badchannel.png')
epochs_cue_after = op.join(project_root, 'derivatives/figures/only-cue-epochs-reject-after-badchannel.png')
epochs_all_before = op.join(project_root, 'derivatives/figures/all-epochs-reject-before-badchannel.png')

if summary_rprt:

    report_root = op.join(project_root, 'derivatives/reports')  
    if not op.exists(op.join(report_root , 'sub-' + subject)):
        os.makedirs(op.join(report_root , 'sub-' + subject))
    report_folder = op.join(report_root , 'sub-' + subject)

    report_fname = op.join(report_folder, 
                        f'sub-{subject}_preproc_1.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub-{subject}_preproc_1.html')

    report = mne.open_report(report_fname)

    report.add_figure(fig=fig_bads, title='dropped epochs',
                    caption='epochs dropped and channel-one bad channel (FT10) added after plotting bad epochs', 
                    tags=('epo'),
                    section='epocheds'
                    )
    report.add_figure(fig=fig_psd, title='psd after dropped',
                    caption='psd with bad channels marked after dropping bad epochs', 
                    tags=('epo'),
                    section='epocheds'
                    )
    # report.add_image(epochs_cue_before,
    #                 title='epoched on cues',
    #                 caption='epoched on cues, threshold for reject calculated before dropping bad epoch channel',
    #                 tags=('epo'))
    # report.add_image(epochs_cue_after,
    #             title='epoched on cues',
    #             caption='epoched on cues, threshold for reject calculated after dropping bad epoch channel',
    #             tags=('epo'))
    # report.add_image(epochs_all_before,
    #             title='epoched on all events',
    #             caption='epoched on alle events, threshold for reject calculated before dropping bad epoch channel',
    #             tags=('epo'))
 
    report.save(report_fname, overwrite=True, open_browser=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
