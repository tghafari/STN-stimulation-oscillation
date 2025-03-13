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
subject = '104'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
stim_suffix = 'stim'
no_stim_suffix = 'no-stim'
extension = '.fif'

stim_segments_ls = [False, True]
epoching_list = ['cue', 'stim']  # epoching on cue onset or stimulus onset

platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
test_plot = False

# Select ROI sensors for erp
occipital_channels = ['O1', 'PO3', 'O2', 'PO4', 'Oz', 'POz']

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
                    f'sub-{subject}_130325.hdf5')    # it is in .hdf5 for later adding images
html_report_fname = op.join(report_folder, f'sub-{subject}_130325.html')

report = mne.open_report(report_fname)

for epoching in epoching_list:
    print(f'Working on {epoching}')
    input_suffix = 'epo-' + epoching
    deriv_suffix = 'evo-' + epoching
    evoked_list_cropped = []  #  -0.1 to 0.5
    evoked_list = []  # -0.1 to 1

    for stim in stim_segments_ls:
        print(f'Working on stim = {stim}')

        bids_path = BIDSPath(subject=subject, session=session,
                    task=task, run=run, root=bids_root, 
                    datatype ='eeg', suffix=eeg_suffix)
        deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results

        epochs, evoked = reading_epochs_evoking(stim)
        evoked.comment = f'stim:{stim_segments_ls[stim]}, {epoching} onset'
        evoked_list_cropped.append(evoked.copy().crop(-0.1,0.5))
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
    fig_comp_chs, axis = plt.subplots(3, 2, figsize=(24, 12)) 
    axis = axis.flatten()  # flatten the axes for easier iteration (3x2 grid)

    # Plot for each occipital channel
    for ax_idx, ch in enumerate(occipital_channels):
        mne.viz.plot_compare_evokeds(
            evoked_list_cropped,
            picks=ch,
            colors=['blue', 'orange'],  # Specify colors for comparison
            combine="mean",
            axes=axis[ax_idx],  # Use correct axis
            show_sensors=True,  
            invert_y=False,
            truncate_xaxis=False,
            truncate_yaxis=False,
            show=False
        )
        axis[ax_idx].set_title(f'{ch}')
        axis[ax_idx].set_xlim(-0.1, 0.5)
    plt.show()
        
    fig_comp_plot_topo = mne.viz.plot_evoked_topo(evoked_list,
                                          color=['blue','orange'], 
                                          vline=(0.0))

    report.add_figure(fig=fig_comp_chs, title=f'compare evoked responses',
                                caption=f'evoked response for {epoching} \
                                    cue=200ms, ISI=1000, stim=1000-2000ms', 
                                tags=('evo'),
                                section='evoked'
                                )
    report.add_figure(fig=fig_comp_plot_topo, title=f'compare evoked responses',
                                caption=f'evoked response for {epoching} \
                                    cue=200ms, ISI=1000, stim=1000-2000ms', 
                                tags=('evo'),
                                section='evoked'
                                )

report.save(report_fname, overwrite=True)
report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
