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

"""

import os.path as op
import os
import numpy as np

import matplotlib.pyplot as plt
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
    epochs = mne.read_epochs(input_fname, verbose=True, preload=True)  # -.5 to 1.5sec
    # Make evoked data for conditions of interest and save
    evoked = epochs.copy().average(method='mean').filter(0.0,30).crop(-.1,1)
    evoked = evoked.apply_baseline(baseline=(-.1,0), verbose=True) 
    mne.write_evokeds(deriv_fname, evoked, verbose=True, overwrite=True)

    return epochs, evoked


# BIDS settings: fill these out 
subject = '110'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
stim_suffix = 'stim'
no_stim_suffix = 'no-stim'
extension = '.fif'

runs = ['01']
stim_segments_ls = [False, True]
epoching_list = ['cue', 'stim']  # epoching on cue onset or stimulus onset

platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
test_plot = False

# Select ROI sensors for erp
occipital_channels = ['O1', 'O2', 'Oz', 'PO4', 'POz', 'PO3']

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

for epoching in epoching_list:
    print(f'Working on {epoching}')
    input_suffix = 'epo-' + epoching
    deriv_suffix = 'evo-' + epoching
    evoked_list = []

    for stim in stim_segments_ls:
        print(f'Working on stim = {stim}')

        for run in runs:
            print(f'Reading run:{run}')

            bids_path = BIDSPath(subject=subject, session=session,
                        task=task, run=run, root=bids_root, 
                        datatype ='eeg', suffix=eeg_suffix)
            deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results

            epochs, evoked = reading_epochs_evoking(stim)
            evoked.comment = f'stim:{stim_segments_ls[stim]}, {epoching} onset'
            evoked_list.append(evoked)  # append evokeds for later comparison

            # Plot ERPs for summary report
            topos_times = np.arange(50, 450, 30)*0.001
            fig_evo = evoked.copy().plot_joint(times=topos_times)
        
            report.add_figure(fig=fig_evo, title=f'stim:{stim}, evoked response',
                                caption=f'evoked response for {epoching}- baseline=(-100,0), filter=(0,30) \
                                        cue=200ms, ISI=1000, stim=1000-2000ms', 
                                tags=('evo'),
                                section='stim'
                                )
            # Plot epochs separately 
            fig_epochs, axis = plt.subplots(6, 2, figsize=(24, 6))
            for ax, ch in enumerate(occipital_channels):
                epochs.plot_image(picks=ch,
                                 axes=axis[ax,:],
                                 colorbar=False,
                                 show=False)
                axis[ax][0].title.set_text(f'{ch}')

            fig_epochs.suptitle(f"stim={stim}- epochs for {epoching}")
            fig_epochs.set_tight_layout(True)
            plt.show()
            
            report.add_figure(fig=fig_epochs, title=f'stim:{stim}, epochs separately',
                                caption=f'epochs for {epoching}- baseline=(-100,0), filter=(0,30) \
                                        cue=200ms, ISI=1000, stim=1000-2000ms', 
                                tags=('epo'),
                                section='stim'
                                )

            del epochs, evoked

    # Plot both stim and no stim evoked in one plot
    fig_comp_chs, axis = plt.subplots(6, 1, figsize=(24, 6))
    for ax, ch in enumerate(occipital_channels):
         mne.viz.plot_compare_evokeds(evoked_list, 
                                    picks=ch,
                                    colors=['blue','orange'], 
                                    combine="mean",
                                    axes=axis[ax], 
                                    show_sensors=True,
                                    invert_y=False,
                                    truncate_xaxis=False,
                                    truncate_yaxis=False)
         axis[ax].title.set_text(f'{ch}')

        
    fig_comp_plot_topo = mne.viz.plot_evoked_topo(evoked_list,
                                          color=['blue','orange'], 
                                          vline=(0.0))

    report.add_figure(fig=fig_comp_chs, title=f'compare evoked responses',
                                caption=f'evoked response for {epoching} \
                                    cue=200ms, ISI=1000, stim=1000-2000ms', 
                                tags=('evo'),
                                section='stim'
                                )
    report.add_figure(fig=fig_comp_plot_topo, title=f'compare evoked responses',
                                caption=f'evoked response for {epoching} \
                                    cue=200ms, ISI=1000, stim=1000-2000ms', 
                                tags=('evo'),
                                section='stim'
                                )

report.save(report_fname, overwrite=True)
report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
