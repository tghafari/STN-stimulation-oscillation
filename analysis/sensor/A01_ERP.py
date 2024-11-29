# -*- coding: utf-8 -*-
"""
===============================================
A01. Event related fields

This code will generate ERFs in response to the
visual input.

written by Tara Ghafari
adapted from flux pipeline
==============================================
ToDos:    
Questions:
    1) which conditions to equalize?

"""

import os.path as op
import os
import numpy as np

import mne
from mne_bids import BIDSPath


def reading_epochs_evoking(stim):
    if stim:
        input_fname = op.join(deriv_folder, bids_path.basename
                               + '_' + stim_suffix + '_' + input_suffix + extension)
        deriv_fname = op.join(deriv_folder, bids_path.basename 
                               + '_' + stim_suffix + '_' + deriv_suffix + extension)  
    else:
        input_fname = op.join(deriv_folder, bids_path.basename
                               + '_' + no_stim_suffix + '_' + input_suffix + extension)
        deriv_fname = op.join(deriv_folder, bids_path.basename 
                               + '_' + no_stim_suffix + '_' + deriv_suffix + extension)  

    # Read epoched data and equalize right and left
    epochs = mne.read_epochs(input_fname, verbose=True, preload=True)  # -.7 to 1.7sec

    # Make evoked data for conditions of interest and save
    evoked = epochs['cue_onset_right','cue_onset_left'].copy().average(method='mean').filter(0.0,30).crop(-.1,1)
    evoked = evoked.apply_baseline(-.1,0) 
    mne.write_evokeds(deriv_fname, evoked, verbose=True, overwrite=True)

    return epochs, evoked

# BIDS settings: fill these out 
subject = '107'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
stim_suffix = 'stim'
no_stim_suffix = 'no-stim'
input_suffix = 'epo'
deriv_suffix = 'ave'
extension = '.fif'

runs = ['01']
stim_segments_ls = [False, True]

pilot = False  # is it pilot data or real data?
platform = 'bluebear'  # are you using 'bluebear', 'mac', or 'windows'?
test_plot = False

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
                    f'sub-{subject}_preproc_ica.hdf5')    # it is in .hdf5 for later adding images
html_report_fname = op.join(report_folder, f'sub-{subject}_preproc_ica.html')

report = mne.open_report(report_fname)

evoked_list = []
for stim in stim_segments_ls:
    print(f'Reading stim:{stim}')
    for run in runs:
        print(f'Reading run:{run}')
        bids_path = BIDSPath(subject=subject, session=session,
                     task=task, run=run, root=bids_root, 
                     datatype ='eeg', suffix=eeg_suffix)
        deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results

        epochs, evoked = reading_epochs_evoking(stim)
        evoked.comment = f'stim:{stim_segments_ls[stim]}'
        evoked_list.append(evoked)  # append evokeds for later comparison

        # # Plot ERF for summary report
        # topos_times = np.arange(50,450,30)*0.001
        # fig_evo = evoked.copy().plot_joint(times=topos_times)
       
        # report.add_figure(fig=fig_evo, title=f'stim:{stim}, evoked response',
        #                     caption=f'evoked response for cue = 0-200ms\
        #                         and stim = 1200ms', 
        #                     tags=('evo'),
        #                     section='stim'
        #                     )

        if test_plot:
            # ==================================== RIGHT LEFT SEPARATELY ==============================================
            evoked_right = epochs['cue_onset_right'].copy().average(method='mean').filter(0.0,60).crop(-.7,1.7) 
            evoked_left = epochs['cue_onset_left'].copy().average(method='mean').filter(0.0,60).crop(-.7,1.7)
            evokeds = [evoked_right, evoked_left]

            # Plot evoked_right data
            epochs['cue_onset_right'].copy().filter(0.0,30).crop(-.2,1.7).plot_image()
            evoked_right.copy().apply_baseline(baseline=(-.5,-.2))
            evoked_right.copy().plot_topo(title='cue onset right')
            evoked_right.copy().plot_topomap(.1, time_unit='s')

            # Plot magnetometers for summary report
            fig_right = evoked_right.copy().plot_joint(times=[0.150,0.270,0.410])
            fig_left = evoked_left.copy().plot_joint(times=[0.150,0.255,0.395])

        # ==================================== RIGHT LEFT TOGETHER ==============================================
            # Plot evoked data
            evoked.copy().apply_baseline(baseline=(-.5,-.2))
            evoked.copy().plot_topo(title='Evoked response')
            evoked.copy().plot_topomap(.2, time_unit='s')

            # Explore the epoched dataset
            resampled_epochs = epochs.copy().resample(200)  
            resampled_epochs.compute_psd(fmin=1.0, fmax=60.0).plot(spatial_colors=True)  # explore the frequency content of the epochs
            resampled_epochs.compute_psd().plot_topomap(normalize=False)  # spatial distribution of the PSD

# Select ROI sensors
occipital_channels = ['O2', 'Oz', 'O1']
# , 'PO8', 'PO4', 'POz', 'PO3', 'PO7', 'P8', 'P6', 'P4', 'P2',
#                     'Pz', 'P1', 'P3', 'P5', 'P7', 'TP10', 'TP8', 'CP6', 'CP4', 'CP2', 'CPz',
#                     'CP1', 'CP3', 'CP5', 'TP7', 'TP9']

# Plot both stim and no stim evoked in one plot
fig_comp = mne.viz.plot_compare_evokeds(evoked_list, 
                                        picks=occipital_channels,
                                        colors=['blue','orange'], 
                                        combine="mean", 
                                        ci=0.9,
                                        show_sensors=True)

report.add_figure(fig=fig_comp, title=f'compare evoked responses',
                            caption=f'evoked response for cue = 0-200ms\
                                and stim = 1200ms for both stim and no stimulation conditions', 
                            tags=('evo'),
                            section='stim'
                            )

report.save(report_fname, overwrite=True)
report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
