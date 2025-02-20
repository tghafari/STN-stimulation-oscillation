# -*- coding: utf-8 -*-
"""
===============================================
08. Epoching raw data based on conditions

This code will epoch continuous EEG based
on conditions that are annotated in the
data and generates an HTML report about epochs.

best cleaning outcome: do not autoreject, 
remove bad channels and clean only by manual 
inspection. remove any epoch with slightest 
uncleannes.


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
    # Have to explicitly assign values to events otherwise it assigns arbitrary event_ids
    event_dict = {'cue_onset_right':1,
            'cue_onset_left':2,
            'trial_onset':3,
            'stim_onset':4,
            'catch_onset':5,
            'dot_onset_right':6,
            'dot_onset_left':7,
            'response_press_onset':8,
            'block_onset':20,
            'block_end':21,
            'experiment_end':30, 
            'new_stim_segment':99999, 
            }
    events, events_id = mne.events_from_annotations(segmented_ica, event_id=event_dict)

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
subject = '102'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
input_suffix = 'ica'
stim_suffix = 'stim'
no_stim_suffix = 'no-stim'
extension = '.fif'

stim_segments_ls = [False, True]
epoching_types = {
    'cue': ['cue_onset_right', 'cue_onset_left'],
    'stim': ['stim_onset']
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

# project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
# bids_root = op.join(project_root, 'data', 'BIDS')

# for bear outage
project_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD'  # only for bear outage time
bids_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/data/BIDS'

# Epoch stim segments and add to report
report_root = op.join(project_root, 'derivatives/reports')  
report_folder = op.join(report_root , 'sub-' + subject)
report_fname = op.join(report_folder, 
                    f'sub-{subject}_200225.hdf5')    # it is in .hdf5 for later adding images
html_report_fname = op.join(report_folder, f'sub-{subject}_200225.html')

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

####################################### sub-107 #################################################
# cue onset, stim off
# Dropped 73 epochs: 1, 4, 6, 7, 8, 9, 10, 13, 14, 16, 17, 22, 24, 25, 26, 29, 32, 34, 36, 37, 38, 40, 42, 47, 48, 50, 51, 52, 55, 57, 58, 59, 63, 67, 69, 70, 71, 73, 74, 76, 77, 79, 80, 82, 83, 89, 90, 94, 99, 101, 103, 104, 105, 108, 110, 111, 112, 114, 116, 118, 120, 125, 127, 130, 131, 132, 136, 140, 145, 148, 150, 152, 157
# The following epochs were marked as bad and are dropped:
# [7, 18, 28, 33, 38, 43, 48, 63, 68, 78, 83, 108, 118, 123, 128, 143, 157, 167, 177, 180, 185, 197, 206, 231, 236, 246, 251, 256, 271, 279, 284, 289, 307, 327, 337, 342, 345, 353, 358, 368, 373, 383, 390, 398, 403, 433, 438, 456, 480, 490, 500, 503, 508, 523, 533, 538, 542, 552, 562, 572, 584, 607, 617, 630, 635, 640, 658, 678, 701, 716, 726, 736, 761]
# Channels marked as bad:
# ['TP9', 'TP10', 'TP7', 'FC3', 'C5', 'AF8', 'P7', 'FC5', 'C2', 'FC1', 'F3', 'AF7', 'F1', 'AFz', 'C6', 'AF4', 'P3', 'Pz']
# stim onset, stim off
# .Dropped 61 epochs: 1, 2, 3, 6, 7, 8, 11, 14, 15, 18, 20, 23, 24, 26, 29, 31, 32, 33, 34, 37, 38, 42, 43, 45, 46, 47, 48, 49, 50, 53, 58, 61, 63, 65, 66, 69, 71, 74, 82, 83, 86, 87, 90, 91, 97, 99, 103, 104, 105, 108, 114, 121, 125, 127, 129, 131, 132, 134, 135, 140, 143
# The following epochs were marked as bad and are dropped:
# [8, 19, 24, 39, 44, 49, 64, 79, 84, 99, 109, 124, 129, 139, 158, 168, 173, 181, 186, 202, 207, 227, 232, 242, 247, 252, 257, 262, 267, 285, 313, 328, 338, 354, 359, 374, 384, 404, 447, 452, 467, 471, 486, 491, 524, 534, 558, 563, 568, 585, 618, 659, 679, 689, 702, 712, 717, 727, 732, 757, 772]
# Channels marked as bad:
# ['TP9', 'TP10', 'TP7', 'FC3', 'C5', 'AF8', 'P7', 'FC5', 'C2', 'FC1', 'F3', 'AF7', 'F1', 'AFz', 'C6', 'AF4', 'P3', 'Pz']
# cue onset, stim on:
# Dropped 70 epochs: 0, 3, 7, 10, 14, 17, 19, 21, 24, 25, 28, 29, 31, 32, 33, 38, 39, 41, 44, 48, 49, 51, 52, 54, 56, 60, 61, 65, 66, 67, 68, 70, 72, 73, 76, 77, 80, 82, 83, 87, 88, 90, 91, 92, 94, 98, 99, 104, 105, 106, 108, 110, 112, 114, 118, 120, 125, 130, 132, 137, 139, 142, 146, 147, 151, 152, 154, 155, 158, 159
# The following epochs were marked as bad and are dropped:
# [2, 17, 37, 52, 72, 87, 97, 107, 118, 121, 136, 139, 149, 154, 159, 183, 188, 200, 215, 235, 240, 250, 255, 265, 275, 293, 298, 316, 321, 324, 329, 337, 347, 352, 367, 372, 390, 400, 403, 419, 424, 434, 439, 444, 454, 474, 479, 504, 509, 514, 522, 532, 542, 552, 572, 584, 607, 630, 640, 665, 675, 690, 710, 715, 733, 738, 748, 753, 768, 771]
# Channels marked as bad:
# ['TP9', 'TP10', 'TP7', 'FC3', 'C5', 'AF8', 'P7', 'FC5', 'C2', 'FC1', 'F3', 'AF7', 'F1', 'AFz', 'C6', 'AF4', 'P3', 'Pz']
# .Dropped 57 epochs: 0, 2, 4, 7, 8, 9, 12, 15, 16, 17, 19, 21, 25, 26, 29, 30, 33, 34, 35, 36, 39, 43, 44, 45, 47, 48, 57, 59, 62, 64, 68, 69, 70, 73, 75, 78, 80, 81, 84, 90, 93, 96, 98, 101, 103, 106, 108, 116, 119, 121, 123, 124, 125, 130, 134, 135, 141
# The following epochs were marked as bad and are dropped:
# [3, 13, 23, 38, 43, 48, 63, 78, 83, 88, 98, 108, 140, 145, 160, 165, 180, 184, 189, 196, 211, 231, 236, 241, 251, 256, 307, 317, 338, 348, 368, 373, 378, 396, 412, 430, 440, 445, 460, 490, 505, 523, 533, 548, 558, 573, 585, 631, 646, 656, 666, 671, 676, 701, 724, 729, 759]
# Channels marked as bad:
# ['TP9', 'TP10', 'TP7', 'FC3', 'C5', 'AF8', 'P7', 'FC5', 'C2', 'FC1', 'F3', 'AF7', 'F1', 'AFz', 'C6', 'AF4', 'P3', 'Pz']

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
# Dropped 47 epochs: 1, 6, 9, 11, 14, 19, 23, 25, 27, 28, 42, 55, 57, 58, 61, 62, 64, 74, 75, 77, 79, 80, 81, 83, 84, 85, 88, 91, 93, 96, 99, 100, 103, 104, 110, 111, 115, 120, 129, 130, 137, 139, 143, 145, 148, 152, 153
# The following epochs were marked as bad and are dropped:
# [6, 26, 39, 49, 62, 84, 102, 110, 119, 123, 188, 250, 259, 263, 275, 280, 289, 337, 342, 352, 360, 367, 372, 381, 386, 391, 406, 421, 431, 445, 460, 465, 480, 485, 513, 518, 538, 564, 606, 611, 646, 656, 674, 684, 699, 719, 722]
# Channels marked as bad:
# ['TP9', 'TP10', 'Fp1', 'F7', 'Fp2', 'F8', 'TP7', 'PO7', 'F6', 'F5', 'FC6', 'AF7', 'AF3', 'FT7', 'F3', 'C5']
# stim, stim off
# Dropped 45 epochs: 0, 7, 8, 11, 16, 21, 22, 24, 32, 35, 44, 50, 51, 52, 54, 55, 70, 71, 73, 74, 76, 89, 90, 93, 94, 99, 101, 107, 108, 116, 119, 121, 122, 124, 125, 129, 130, 131, 133, 134, 135, 136, 140, 141, 142
# The following epochs were marked as bad and are dropped:
# [3, 40, 45, 63, 85, 107, 111, 120, 160, 173, 218, 246, 255, 260, 269, 276, 353, 361, 373, 382, 392, 461, 466, 481, 486, 514, 524, 558, 565, 607, 622, 632, 637, 647, 652, 675, 680, 685, 695, 700, 705, 710, 736, 741, 746]
# Channels marked as bad:
# ['TP9', 'TP10', 'Fp1', 'F7', 'Fp2', 'F8', 'TP7', 'PO7', 'F6', 'F5', 'FC6', 'AF7', 'AF3', 'FT7', 'F3', 'C5']
# cue, stim on
# Dropped 36 epochs: 0, 9, 13, 19, 24, 25, 27, 28, 29, 35, 36, 37, 40, 41, 47, 48, 50, 54, 56, 58, 63, 64, 69, 78, 80, 85, 86, 98, 121, 123, 126, 131, 137, 143, 145, 148
# The following epochs were marked as bad and are dropped:
# [2, 43, 63, 91, 116, 121, 131, 136, 141, 171, 176, 180, 195, 199, 227, 231, 239, 257, 267, 277, 301, 305, 330, 375, 387, 409, 414, 474, 583, 591, 606, 631, 659, 687, 695, 710]
# Channels marked as bad:
# ['TP9', 'TP10', 'Fp1', 'F7', 'Fp2', 'F8', 'TP7', 'PO7', 'F6', 'F5', 'FC6', 'AF7', 'AF3', 'FT7', 'F3', 'C5']
# stim, stim on
# Dropped 27 epochs: 0, 3, 12, 20, 25, 26, 28, 33, 35, 36, 41, 42, 43, 44, 45, 51, 58, 61, 70, 71, 72, 80, 91, 108, 111, 129, 132
# The following epochs were marked as bad and are dropped:
# [3, 19, 64, 107, 132, 137, 147, 172, 189, 196, 224, 228, 232, 236, 240, 273, 316, 331, 376, 381, 388, 430, 485, 584, 597, 691, 706]
# Channels marked as bad:
# ['TP9', 'TP10', 'Fp1', 'F7', 'Fp2', 'F8', 'TP7', 'PO7', 'F6', 'F5', 'FC6', 'AF7', 'AF3', 'FT7', 'F3', 'C5']


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

####################################### sub-112 #################################################
# #cue, stim off
# Press return when you're done annotating bad segments ...Dropped 44 epochs: 2, 5, 20, 25, 26, 29, 33, 36, 40, 41, 42, 43, 44, 45, 46, 47, 48, 52, 57, 60, 62, 64, 68, 71, 74, 77, 79, 80, 86, 93, 95, 97, 108, 112, 116, 122, 124, 126, 128, 140, 142, 147, 154, 156
# The following epochs were marked as bad and are dropped:
# [22, 47, 178, 225, 236, 273, 315, 344, 412, 423, 430, 439, 452, 463, 470, 481, 494, 540, 595, 624, 646, 666, 706, 741, 772, 805, 825, 836, 888, 941, 953, 967, 1042, 1074, 1106, 1166, 1184, 1200, 1218, 1314, 1333, 1378, 1437, 1449]
# Channels marked as bad:
# ['TP9', 'TP10', 'Fp1', 'CP6', 'FC5', 'F7', 'FC6', 'AF7', 'AF3', 'Fp2', 'F8', 'F4', 'F5', 'FT7', 'C5', 'TP7', 'AF8', 'FT8', 'C6', 'AF4', 'F6', 'Cz', 'P1', 'CPz']

# stim, stim off
# Press return when you're done annotating bad segments ...Dropped 37 epochs: 2, 17, 22, 26, 27, 28, 31, 33, 37, 38, 39, 40, 41, 42, 43, 46, 49, 56, 59, 62, 63, 65, 71, 82, 88, 90, 93, 94, 100, 107, 108, 109, 115, 119, 123, 139, 140
# The following epochs were marked as bad and are dropped:
# [23, 179, 228, 276, 287, 300, 327, 347, 415, 424, 431, 442, 455, 484, 497, 543, 574, 680, 709, 742, 753, 777, 839, 933, 985, 1001, 1034, 1045, 1091, 1162, 1167, 1176, 1244, 1278, 1315, 1450, 1455]
# Channels marked as bad:
# ['TP9', 'TP10', 'Fp1', 'CP6', 'FC5', 'F7', 'FC6', 'AF7', 'AF3', 'Fp2', 'F8', 'F4', 'F5', 'FT7', 'C5', 'TP7', 'AF8', 'FT8', 'C6', 'AF4', 'F6', 'Cz', 'P1', 'CPz']

# cue, stim on
# Press return when you're done annotating bad segments ...Dropped 36 epochs: 11, 26, 34, 38, 43, 45, 47, 49, 50, 52, 54, 58, 60, 63, 65, 66, 67, 68, 69, 70, 71, 73, 80, 83, 87, 88, 89, 92, 94, 100, 132, 137, 143, 144, 145, 149
# The following epochs were marked as bad and are dropped:
# [63, 158, 213, 244, 283, 297, 312, 328, 334, 347, 359, 385, 397, 419, 434, 440, 448, 452, 459, 466, 472, 488, 539, 564, 594, 597, 604, 627, 641, 689, 943, 984, 1038, 1049, 1060, 1096]
# Channels marked as bad:
# ['TP9', 'TP10', 'Fp1', 'CP6', 'FC5', 'F7', 'FC6', 'AF7', 'AF3', 'Fp2', 'F8', 'F4', 'F5', 'FT7', 'C5', 'TP7', 'AF8', 'FT8', 'C6', 'AF4', 'F6', 'Cz', 'P1', 'CPz']

# stim, stim on
# Press return when you're done annotating bad segments ...Dropped 35 epochs: 9, 12, 14, 20, 24, 26, 28, 29, 30, 37, 43, 45, 47, 48, 53, 54, 56, 59, 62, 65, 66, 70, 75, 79, 86, 89, 93, 96, 120, 121, 122, 128, 129, 130, 131
# The following epochs were marked as bad and are dropped:
# [52, 71, 84, 121, 149, 166, 180, 188, 195, 254, 299, 314, 329, 336, 380, 387, 398, 421, 441, 474, 482, 515, 565, 598, 651, 674, 715, 748, 962, 971, 987, 1039, 1052, 1063, 1072]
# Channels marked as bad:
# ['TP9', 'TP10', 'Fp1', 'CP6', 'FC5', 'F7', 'FC6', 'AF7', 'AF3', 'Fp2', 'F8', 'F4', 'F5', 'FT7', 'C5', 'TP7', 'AF8', 'FT8', 'C6', 'AF4', 'F6', 'Cz', 'P1', 'CPz']

####################################### sub-103 #################################################
# #cue, stim off
# Dropped 48 epochs: 0, 22, 23, 24, 32, 33, 38, 41, 43, 44, 49, 53, 77, 78, 79, 80, 81, 84, 87, 88, 89, 91, 92, 93, 94, 102, 103, 104, 105, 106, 107, 108, 109, 110, 115, 120, 122, 126, 127, 135, 136, 137, 138, 141, 154, 155, 156, 157
# The following epochs were marked as bad and are dropped:
# [3, 140, 146, 152, 199, 205, 235, 264, 276, 281, 310, 334, 479, 485, 492, 502, 509, 527, 545, 551, 557, 569, 576, 583, 587, 637, 643, 649, 655, 661, 667, 672, 679, 684, 718, 761, 777, 800, 807, 858, 864, 871, 877, 895, 977, 984, 990, 998]
# Channels marked as bad:
# ['TP9', 'TP10', 'P5', 'PO7', 'AF4', 'TP8', 'FC1', 'AFz', 'F6', 'P1', 'P6', 'F4', 'AF3']

# stim, stim off
# Press return when you're done annotating bad segments ...Dropped 32 epochs: 0, 15, 21, 34, 42, 59, 69, 70, 71, 72, 74, 76, 79, 81, 84, 86, 87, 91, 92, 93, 96, 102, 107, 108, 112, 113, 121, 122, 123, 126, 133, 134
# The following epochs were marked as bad and are dropped:
# [6, 112, 147, 230, 300, 420, 480, 487, 493, 503, 516, 528, 546, 558, 577, 595, 601, 626, 632, 638, 656, 710, 744, 762, 795, 802, 859, 865, 872, 896, 941, 947]
# Channels marked as bad:
# ['TP9', 'TP10', 'P5', 'PO7', 'AF4', 'TP8', 'FC1', 'AFz', 'F6', 'P1', 'P6', 'F4', 'AF3']

# cue, stim on
# Press return when you're done annotating bad segments ...Dropped 30 epochs: 24, 51, 54, 58, 63, 75, 80, 81, 87, 89, 101, 103, 106, 111, 119, 120, 125, 126, 127, 128, 129, 132, 134, 137, 138, 139, 140, 147, 148, 158
# The following epochs were marked as bad and are dropped:
# [139, 316, 336, 358, 388, 458, 492, 498, 537, 551, 628, 641, 662, 694, 746, 761, 791, 797, 805, 810, 817, 834, 847, 868, 874, 881, 888, 933, 940, 1004]
# Channels marked as bad:
# ['TP9', 'TP10', 'P5', 'PO7', 'AF4', 'TP8', 'FC1', 'AFz', 'F6', 'P1', 'P6', 'F4', 'AF3']

# stim, stim on
# Press return when you're done annotating bad segments ...Dropped 28 epochs: 6, 32, 46, 48, 49, 72, 73, 89, 90, 91, 92, 93, 94, 95, 96, 97, 104, 108, 113, 114, 115, 116, 117, 123, 124, 125, 133, 136
# The following epochs were marked as bad and are dropped:
# [37, 199, 318, 331, 337, 493, 499, 616, 623, 630, 636, 642, 648, 655, 664, 670, 727, 763, 792, 798, 806, 812, 818, 862, 869, 876, 935, 966]
# Channels marked as bad:
# ['TP9', 'TP10', 'P5', 'PO7', 'AF4', 'TP8', 'FC1', 'AFz', 'F6', 'P1', 'P6', 'F4', 'AF3']