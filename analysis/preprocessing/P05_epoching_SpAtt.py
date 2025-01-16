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

    original_bads = deepcopy(epochs.info["bads"])

    print(f'these are original bads: {original_bads}')
    user_list = input('Any other bad channels from autorejec bad drop plot? name of channel, e.g. FT10 T9 (separate by space) or return.')
    bad_channels = user_list.split()

    if len(bad_channels) > 0:
        epochs.copy().pick(bad_channels).compute_psd().plot()  # double check bad channels
        if len(bad_channels) == 1:
            print('one bad channel removing')
            epochs.info["bads"].append(bad_channels[0])  # add a single channel
        else:
            print(f'{len(bad_channels)} bad channels removing')
            epochs.info["bads"].extend(bad_channels)  # add a list of channels - should there be more than one channel to drop
    
    return fig_bads_temp

def cleaning_and_saving_epochs(stim, epochs, autoreject=False):
    """saves manually or automatically cleaned epochs."""
    if stim:
        deriv_fname = op.join(deriv_folder, bids_path.basename 
                               + '_' + stim_suffix + '_' + deriv_suffix + extension)  
    else:
        deriv_fname = op.join(deriv_folder, bids_path.basename 
                               + '_' + no_stim_suffix + '_' + deriv_suffix + extension)  
    if autoreject:
        reject = get_rejection_threshold(epochs)  # reject without bad channels                               
        print(f"\n\n Numer of epochs BEFORE rejection: {len(epochs.events)} \n\n")
        epochs.drop_bad(reject=reject)
        print(f"\n\n Numer of epochs AFTER rejection: {len(epochs.events)} \n\n")
        fig_bads = epochs.plot_drop_log()  # rejected epochs
    else:
        fig_bads = None

    fig_psd = epochs.copy().compute_psd(fmin=0.1,fmax=100,method='welch',n_fft=int(2*epochs.info['sfreq'])).plot()
    
    # Save the epoched data
    epochs.save(deriv_fname, overwrite=True)

    return fig_psd, fig_bads

# BIDS settings: fill these out 
subject = '101'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
input_suffix = 'ica'
stim_suffix = 'stim'
no_stim_suffix = 'no-stim'
extension = '.fif'

stim_segments_ls = [True]#[False, True]
epoching_types = {
    'cue': ['cue_onset_right', 'cue_onset_left'],
    # 'stim': ['stim_onset']
}

pilot = False  # is it pilot data or real data?
test_plot = False
platform = 'mac'  # are you using 'bluebear' or 'mac'?
autoreject = False  # do you manually remove the artifacts or use autoreject? (can also be both)

if platform == 'bluebear':
    rds_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    camcan_dir = '/rds/projects/q/quinna-camcan/dataman/data_information'
elif platform == 'mac':
    rds_dir = '/Volumes/jenseno-avtemporal-attention'
    camcan_dir = '/Volumes/quinna-camcan/dataman/data_information'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
bids_root = op.join(project_root, 'data', 'BIDS')
# for bear outage
bids_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/data/BIDS'

# Epoch stim segments and add to report
report_root = op.join(project_root, 'derivatives/reports')  
# for bear outage
report_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/derivatives/reports' # only for bear outage time

report_folder = op.join(report_root , 'sub-' + subject)

report_fname = op.join(report_folder, 
                    f'sub-{subject}_150125.hdf5')    # it is in .hdf5 for later adding images
html_report_fname = op.join(report_folder, f'sub-{subject}_150125.html')

report = mne.open_report(report_fname)

for stim in stim_segments_ls:
    print(f'stimulation: {stim}')
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

        if epoching == 'cue':  # only cue has two (right and left) epochs
            mne.epochs.equalize_epoch_counts([epochs_of_interest['cue_onset_right'], epochs_of_interest['cue_onset_left']])

        if autoreject:
            fig_bads_temp = finding_bad_channel(epochs_of_interest)
            
        fig_psd, fig_bads = cleaning_and_saving_epochs(stim, epochs_of_interest)

        if autoreject:
            report.add_figure(fig=fig_bads, title=f'stim: {stim}, dropped epochs',
                        caption=f'stim: {stim}: bad epochs and channels dropped-{epoching}=0', 
                        tags=('epo'),
                        section='stim'
                        )

        report.add_figure(fig=fig_psd, title=f'stim: {stim}, psd after dropped',
                    caption=f'stim: {stim}, psd with bad epochs and channels dropped-{epoching}=0', 
                    tags=('epo'),
                    section='stim'
                    ) 
        
    del epochs  # refresh variables
        
report.save(report_fname, overwrite=True, open_browser=True)
report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks


# Copy & paste the rejected epochs:

####################################### sub-110 #################################################
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

####################################### sub-102 #################################################
#cue, stim off
# Dropped 21 epochs: 5, 6, 9, 25, 42, 55, 57, 58, 63, 64, 80, 91, 93, 100, 104, 115, 116, 120, 144, 145, 146
# The following epochs were marked as bad and are dropped:
# [27, 32, 45, 120, 209, 272, 281, 286, 308, 312, 583, 639, 649, 686, 707, 760, 764, 799, 916, 921, 926]
# Channels marked as bad:
# ['TP9', 'TP10', 'F7', 'TP7', 'PO7', 'F6', 'FT8', 'Fp1', 'F8', 'FT7', 'FC6', 'F5', 'Fp2', 'C5']
# stim, stim off
# Dropped 31 epochs: 0, 7, 21, 22, 30, 32, 35, 50, 54, 71, 73, 74, 75, 76, 85, 87, 89, 94, 96, 100, 108, 117, 124, 125, 129, 130, 131, 132, 134, 135, 143
# The following epochs were marked as bad and are dropped:
# [7, 46, 115, 121, 162, 176, 190, 268, 292, 388, 590, 599, 604, 609, 659, 669, 681, 708, 718, 741, 801, 849, 884, 889, 912, 917, 922, 927, 937, 942, 989]
# Channels marked as bad:
# ['TP9', 'TP10', 'F7', 'TP7', 'PO7', 'F6', 'FT8', 'Fp1', 'F8', 'FT7', 'FC6', 'F5', 'Fp2', 'C5']
# cue, stim on
# Dropped 32 epochs: 0, 9, 15, 27, 29, 32, 33, 34, 35, 37, 39, 40, 41, 47, 48, 49, 50, 63, 64, 66, 78, 80, 81, 85, 86, 120, 121, 123, 124, 126, 145, 148
# The following epochs were marked as bad and are dropped:
# [2, 46, 75, 136, 146, 161, 166, 171, 176, 185, 193, 213, 219, 249, 253, 258, 265, 329, 333, 343, 406, 438, 445, 464, 470, 651, 655, 665, 670, 682, 772, 787]
# Channels marked as bad:
# ['TP9', 'TP10', 'F7', 'TP7', 'PO7', 'F6', 'FT8', 'Fp1', 'F8', 'FT7', 'FC6', 'F5', 'Fp2', 'C5']
# stim, stim on
# Dropped 26 epochs: 0, 12, 25, 26, 28, 33, 36, 37, 38, 41, 42, 43, 45, 50, 51, 72, 80, 83, 85, 91, 100, 108, 112, 118, 124, 129
# The following epochs were marked as bad and are dropped:
# [5, 68, 137, 142, 152, 177, 215, 224, 230, 245, 250, 254, 267, 296, 301, 441, 487, 502, 513, 543, 593, 656, 677, 708, 736, 768]
# Channels marked as bad:
# ['TP9', 'TP10', 'F7', 'TP7', 'PO7', 'F6', 'FT8', 'Fp1', 'F8', 'FT7', 'FC6', 'F5', 'Fp2', 'C5']

####################################### sub-101 #################################################
#cue, stim off
# Dropped 21 epochs: 5, 6, 9, 25, 42, 55, 57, 58, 63, 64, 80, 91, 93, 100, 104, 115, 116, 120, 144, 145, 146
# The following epochs were marked as bad and are dropped:
# [27, 32, 45, 120, 209, 272, 281, 286, 308, 312, 583, 639, 649, 686, 707, 760, 764, 799, 916, 921, 926]
# Channels marked as bad:
# ['TP9', 'TP10', 'F7', 'TP7', 'PO7', 'F6', 'FT8', 'Fp1', 'F8', 'FT7', 'FC6', 'F5', 'Fp2', 'C5']
# stim, stim off
# Dropped 31 epochs: 0, 7, 21, 22, 30, 32, 35, 50, 54, 71, 73, 74, 75, 76, 85, 87, 89, 94, 96, 100, 108, 117, 124, 125, 129, 130, 131, 132, 134, 135, 143
# The following epochs were marked as bad and are dropped:
# [7, 46, 115, 121, 162, 176, 190, 268, 292, 388, 590, 599, 604, 609, 659, 669, 681, 708, 718, 741, 801, 849, 884, 889, 912, 917, 922, 927, 937, 942, 989]
# Channels marked as bad:
# ['TP9', 'TP10', 'F7', 'TP7', 'PO7', 'F6', 'FT8', 'Fp1', 'F8', 'FT7', 'FC6', 'F5', 'Fp2', 'C5']
# cue, stim on
# Dropped 32 epochs: 0, 9, 15, 27, 29, 32, 33, 34, 35, 37, 39, 40, 41, 47, 48, 49, 50, 63, 64, 66, 78, 80, 81, 85, 86, 120, 121, 123, 124, 126, 145, 148
# The following epochs were marked as bad and are dropped:
# [2, 46, 75, 136, 146, 161, 166, 171, 176, 185, 193, 213, 219, 249, 253, 258, 265, 329, 333, 343, 406, 438, 445, 464, 470, 651, 655, 665, 670, 682, 772, 787]
# Channels marked as bad:
# ['TP9', 'TP10', 'F7', 'TP7', 'PO7', 'F6', 'FT8', 'Fp1', 'F8', 'FT7', 'FC6', 'F5', 'Fp2', 'C5']
# stim, stim on
# Dropped 26 epochs: 0, 12, 25, 26, 28, 33, 36, 37, 38, 41, 42, 43, 45, 50, 51, 72, 80, 83, 85, 91, 100, 108, 112, 118, 124, 129
# The following epochs were marked as bad and are dropped:
# [5, 68, 137, 142, 152, 177, 215, 224, 230, 245, 250, 254, 267, 296, 301, 441, 487, 502, 513, 543, 593, 656, 677, 708, 736, 768]
# Channels marked as bad:
# ['TP9', 'TP10', 'F7', 'TP7', 'PO7', 'F6', 'FT8', 'Fp1', 'F8', 'FT7', 'FC6', 'F5', 'Fp2', 'C5']
