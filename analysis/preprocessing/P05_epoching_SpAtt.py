# -*- coding: utf-8 -*-
"""
===============================================
08. Epoching raw data based on conditions

This code will epoch continuous EEG based
on conditions that are annotated in the
data and generates an HTML report about epochs.

IMO, cleanest way is to epoch based on all events,
calculate reject threshold, find channel with most 
bad epochs, remove channel, calculate threshold 
again, apply on epochs. To ensure only clean
epochs are selected, last step do a visual 
inspection and annotation.


written by Tara Ghafari
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

    print(f'double checking bad channels: {segmented_ica.info["bads"]}')

    events, events_id = mne.events_from_annotations(segmented_ica, event_id='auto')

    # Make epochs (-0.5 t0 1.5 seconds on cue onset or all events)
    epochs = mne.Epochs(segmented_ica,
                        events, 
                        events_id,  # events_id for all events, events_id_to_consider for only cue onsets                  
                        tmin=-0.5, 
                        tmax=1.5,
                        baseline=None, # apply baseline in erp
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

def cleaning_and_saving_epochs(stim, epochs):
    if stim:
        deriv_fname = op.join(deriv_folder, bids_path.basename 
                               + '_' + stim_suffix + '_' + deriv_suffix + extension)  
    else:
        deriv_fname = op.join(deriv_folder, bids_path.basename 
                               + '_' + no_stim_suffix + '_' + deriv_suffix + extension)  
    reject = get_rejection_threshold(epochs)  # reject without bad channels                               
    print(f"\n\n Numer of epochs BEFORE rejection: {len(epochs.events)} \n\n")
    epochs.drop_bad(reject=reject)
    print(f"\n\n Numer of epochs AFTER rejection: {len(epochs.events)} \n\n")
    fig_bads = epochs.plot_drop_log()  # rejected epochs
    fig_psd = epochs.copy().compute_psd(fmin=0.1,fmax=100,method='welch',n_fft=int(2*epochs.info['sfreq'])).plot()
    # Save the epoched data
    epochs.save(deriv_fname, overwrite=True)

    return fig_bads, fig_psd

# BIDS settings: fill these out 
subject = '110'
session = '01'
task = 'SpAtt'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
input_suffix = 'ica'
stim_suffix = 'stim'
no_stim_suffix = 'no-stim'
extension = '.fif'

runs = ['01']
stim_segments_ls = [False, True]
epoching_types = {
    'cue': ['cue_onset_right', 'cue_onset_left'],
    'stim': ['stim_onset']
}

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
bids_root = op.join(project_root, 'data', 'BIDS')
# for bear outage
bids_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/BIDS'

# Epoch stim segments and add to report
report_root = op.join(project_root, 'derivatives/reports')  
# for bear outage
report_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/reports'

report_folder = op.join(report_root , 'sub-' + subject)

report_fname = op.join(report_folder, 
                    f'sub-{subject}_091224.hdf5')    # it is in .hdf5 for later adding images
html_report_fname = op.join(report_folder, f'sub-{subject}_091224.html')

report = mne.open_report(report_fname)

for stim in stim_segments_ls:
    print(f'stimulation: {stim}')
    for run in runs:
        bids_path = BIDSPath(subject=subject, session=session,
                    task=task, run=run, root=bids_root, 
                    datatype ='eeg', suffix=eeg_suffix)
        deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results

        epochs, events, events_id = segment_epoching(stim)

        for epoching, event_keys in epoching_types.items():
            deriv_suffix = f"epo-{epoching}"
            epochs_of_interest = epochs[event_keys]
            print(f'Working on {epochs_of_interest} stim = {stim}')

            manual_annotation = input('Do you want to visually inspect the clean epochs? y/n')
            if manual_annotation == 'y':
                epochs_of_interest.plot()
                input("Press return when you're done annotating bad segments ...")

                rejected_epochs = input("Copy & paste the rejected epochs:")
                                              
                # manual_rejection_html = (f"<p>These epochs were rejected manually:</p> 
                # <ol>
                # <li> {rejected_epochs} </li>
                # </ol>

                # [0, 6, 7, 10, 11, 21, 28, 31, 40, 64, 75, 78, 80, 105, 109, 120, 146]\
                #     The following epochs were marked as bad and are dropped:\
                #     [4, 42, 50, 77, 86, 157, 205, 225, 293, 449, 516, 537, 557, 734, 760, 837, 1022]\
                #     Channels marked as bad:\
                #     ['TP9', 'TP10', 'Fp1', 'FCz', 'AF4', 'Pz', 'F6', 'FT7']</li>\
                # </ol>")  # cue, stim off
                # Dropped 25 epochs: 0, 5, 10, 20, 24, 28, 31, 36, 43, 57, 58, 62, 63, 67, 69, 70, 72, 85, 89, 94, 102, 108, 114, 119, 132
                # The following epochs were marked as bad and are dropped:
                # [7, 44, 88, 174, 200, 226, 246, 295, 339, 444, 451, 485, 492, 517, 531, 538, 559, 660, 689, 721, 793, 839, 879, 919, 1023]
                # Channels marked as bad:
                # ['TP9', 'TP10', 'Fp1', 'FCz', 'AF4', 'Pz', 'F6', 'FT7']  # stim, stim off

                # Dropped 24 epochs: 17, 29, 33, 40, 47, 57, 66, 68, 73, 80, 88, 91, 92, 93, 99, 112, 120, 125, 126, 134, 136, 140, 149, 157
                # The following epochs were marked as bad and are dropped:
                # [110, 179, 203, 260, 300, 365, 418, 430, 461, 525, 574, 596, 603, 611, 646, 726, 787, 819, 825, 875, 889, 914, 968, 1015]
                # Channels marked as bad:
                # ['TP9', 'TP10', 'Fp1', 'FCz', 'AF4', 'Pz', 'F6', 'FT7']  # cue- stim on

                # Dropped 21 epochs: 4, 13, 19, 30, 32, 34, 36, 48, 63, 65, 72, 79, 83, 89, 100, 108, 113, 117, 122, 123, 134
                # The following epochs were marked as bad and are dropped:
                # [30, 86, 128, 206, 220, 232, 261, 347, 449, 462, 526, 576, 605, 648, 728, 789, 820, 852, 884, 891, 969]
                # Channels marked as bad:
                # ['TP9', 'TP10', 'Fp1', 'FCz', 'AF4', 'Pz', 'F6', 'FT7']  # stim- stim on

            if epoching == 'cue':  # only cue has two (right and left) epochs
                mne.epochs.equalize_epoch_counts([epochs_of_interest['cue_onset_right'], epochs_of_interest['cue_onset_left']])

            fig_bads_temp = finding_bad_channel(epochs_of_interest)
            fig_bads, fig_psd = cleaning_and_saving_epochs(stim, epochs_of_interest)

            report.add_figure(fig=fig_bads, title=f'stim: {stim}, dropped epochs',
                        caption=f'stim: {stim}: bad epochs and channels dropped-{epoching}=0', 
                        tags=('epo'),
                        section='stim'
                        )
            report.add_html(title=f"stim: {stim}, epoching on {epoching}- manually rejected epochs", 
                            html=manual_rejection_html,
                            tags=('epo'),
                            section='stim')

            report.add_figure(fig=fig_psd, title=f'stim: {stim}, psd after dropped',
                        caption=f'stim: {stim}, psd with bad epochs and channels dropped-{epoching}=0', 
                        tags=('epo'),
                        section='stim'
                        ) 
        
        del epochs  # refresh variables
        
report.save(report_fname, overwrite=True, open_browser=True)
report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks


