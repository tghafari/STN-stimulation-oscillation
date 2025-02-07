# -*- coding: utf-8 -*-
"""
===============================================
A03. Event related fields

This code will 
    1. navigate to each subjects clean epochs
    2. separately concatenates cue-stim on, 
    cue-stim off, stim-stim on, stim-stim off
    for all subjects.
    3. plots stim on vs stim off evoked responses
    for group level.


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


def reading_epochs_evoking(stim, deriv_folder, basename, save=False):
    if stim:
        input_fname = op.join(deriv_folder, basename
                               + '_' + stim_suffix + '_' + input_suffix + extension)
        deriv_fname = op.join(deriv_folder, basename 
                               + '_' + stim_suffix + '_' + deriv_suffix + extension)  
    else:
        input_fname = op.join(deriv_folder, basename
                               + '_' + no_stim_suffix + '_' + input_suffix + extension)
        deriv_fname = op.join(deriv_folder, basename 
                               + '_' + no_stim_suffix + '_' + deriv_suffix + extension)  

    # Read epoched data and equalize right and left
    epoch = mne.read_epochs(input_fname, verbose=True, preload=True)  # -.5 to 1.5sec
    # Make evoked data for conditions of interest and save
    evoked = epoch.copy().average(method='mean').filter(0.0,30).crop(-.1,1)
    evoked = evoked.apply_baseline(baseline=(-.1,0), verbose=True) 
    if save:
        mne.write_evokeds(deriv_fname, evoked, verbose=True, overwrite=True)

    return epoch, evoked

def fig_compare_chs_plot_topos(occipital_channels, evoked_list_chs, evoked_list_topo, epoching):

    # Plot both stim and no stim evoked in one plot
    fig_comp_chs, axis = plt.subplots(3, 2, figsize=(24, 12)) 
    axis = axis.flatten()  # flatten the axes for easier iteration (3x2 grid)

    # Plot for each occipital channel
    for ax_idx, ch in enumerate(occipital_channels):
        mne.viz.plot_compare_evokeds(
            evoked_list_chs,
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
        
    fig_comp_plot_topo = mne.viz.plot_evoked_topo(evoked_list_topo,
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
            

# BIDS settings: fill these out 
subject_list = ['101', '102', '107', '108', '110', '112', '103', 'concat']
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

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
bids_root = op.join(project_root, 'data', 'BIDS')
# for bear outage
# bids_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/data/BIDS'
deriv_folder_group = op.join(bids_root, 'derivatives', 'group') 
deriv_group_basename = 'sub-concat_ses-01_task-SpAtt_run-01_eeg'

# Epoch stim segments and add to report
report_root = op.join(project_root, 'derivatives/reports')  
# for bear outage
# report_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/derivatives/reports' # only for bear outage time

report_folder = op.join(report_root , 'group')
report_fname = op.join(report_folder, 'subs_101-102-107-108-110-112-103_070225.hdf5')
html_report_fname = op.join(report_folder, 'subs_101-102-107-108-110-112-103_070225.html')
report = mne.Report(title='subs_101-102-107-108-110-112-103')

# Concatenate subjects together based on conditions
for epoching in epoching_list:
    input_suffix = 'epo-' + epoching
    deriv_suffix = 'evo-' + epoching
    print(f'Working on {epoching}')

    for stim in stim_segments_ls:
        print(f'Working on {epoching} stim = {stim}')
        deriv_epoching_stim_fname = op.join(deriv_folder_group, deriv_group_basename 
                                            + '_' + stim_suffix + '_' + input_suffix + extension)
        epochs_all_subs_ls = []
        for subject in subject_list[:-1]:  
            bids_path = BIDSPath(subject=subject, session=session,
            task=task, run=run, root=bids_root, 
            datatype ='eeg', suffix=eeg_suffix)
            deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results

            epoch, _ = reading_epochs_evoking(stim, deriv_folder, bids_path.basename)
            if subject in ['107', '108']:
                epoch.event_id.update({'cue_onset_left':5,'cue_onset_right':6})

            epochs_all_subs_ls.append(epoch.pick(occipital_channels))
            del epoch

        epochs_concat_all_subs = mne.concatenate_epochs(epochs_all_subs_ls)
        epochs_concat_all_subs.save(deriv_epoching_stim_fname)

for subject in subject_list:  
    for epoching in epoching_list:
        print(f'Working on {epoching}')
        input_suffix = 'epo-' + epoching
        deriv_suffix = 'evo-' + epoching
        evoked_list_cropped = []  #  -0.1 to 0.5
        evoked_list = []  # -0.1 to 1

        for stim in stim_segments_ls:
            print(f'Working on stim = {stim}')            
            if subject == subject_list[-1]:
                epochs, evoked = reading_epochs_evoking(stim, deriv_folder_group, 
                                                        deriv_group_basename, save=True)
            else:
                bids_path = BIDSPath(subject=subject, session=session,
                                    task=task, run=run, root=bids_root, 
                                    datatype ='eeg', suffix=eeg_suffix)
                deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results
                epochs, evoked = reading_epochs_evoking(stim, deriv_folder, bids_path.basename)

            evoked.comment = f'stim:{stim_segments_ls[stim]}, {epoching} onset'
            evoked_list_cropped.append(evoked.copy().crop(-0.1,0.5))
            evoked_list.append(evoked)  # append evokeds for later comparison
            del epochs, evoked

            fig_compare_chs_plot_topos(occipital_channels, evoked_list_cropped, evoked_list, epoching)

report.save(report_fname, overwrite=True)
report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks
