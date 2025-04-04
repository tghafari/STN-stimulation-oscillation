"""
===============================================
A04. Grand average TFRs

This code will 
    1. navigate to each subjects clean epochs
    2. appends each subject's tfr to a list.
    3. calculates a grand average of al subjects'
    tfrs.
    then:
    A. Peak Alpha Frequency
    1. calculate TFR for cue right and left
    2. plot TFR plot_topo and TFR on two
    sensors
    3. crop the tfr into time point and frequency (4-14Hz)
      of interest and pick occipital and parietal sensors
    4. find peak alpha frequency range
    5. plot range of peak alpha frequency and topography
    of cue left and cue right on PAF

    B. MI
    6. calculate MI = (attend right - attend left) \
    / (attend right + attend left) 
    7. plot MI over time

    note that plot_topomap is irrelavant as in the
    group average we only keep the occipital 
    channels


written by Tara Ghafari
==============================================
ToDos:    
Questions:

"""

import os.path as op
import os
import numpy as np
import pandas as pd

import mne
from mne_bids import BIDSPath
import matplotlib.pyplot as plt

def tfr_calculation_first_plot(stim, report):
    if stim:
        input_fname = op.join(deriv_folder, bids_path.basename
                               + '_' + stim_suffix + '_' + input_suffix + extension)
        deriv_fname_both = op.join(deriv_folder, bids_path.basename 
                               + '_both_' + stim_suffix + '_' + deriv_suffix + extension) 
        deriv_fname_right = op.join(deriv_folder, bids_path.basename 
                               + '_right_' + stim_suffix + '_' + deriv_suffix + extension) 
        deriv_fname_left = op.join(deriv_folder, bids_path.basename 
                               + '_left_' + stim_suffix + '_' + deriv_suffix + extension)  
    else:
        input_fname = op.join(deriv_folder, bids_path.basename
                               + '_' + no_stim_suffix + '_' + input_suffix + extension)
        deriv_fname_both = op.join(deriv_folder, bids_path.basename 
                               + '_both_' + stim_suffix + '_' + deriv_suffix + extension) 
        deriv_fname_right = op.join(deriv_folder, bids_path.basename 
                               + '_right_' + no_stim_suffix + '_' + deriv_suffix + extension)  
        deriv_fname_left = op.join(deriv_folder, bids_path.basename 
                               + '_left_' + no_stim_suffix + '_' + deriv_suffix + extension)  

    # Read epoched data
    epochs = mne.read_epochs(input_fname, verbose=True, preload=True)  # epochs are from -.5 to 1.5sec

    # ========================================= TFR CALCULATIONS AND FIRST PLOT (PLOT_TOPO) ====================================
    # Calculate tfr for post cue alpha
    tfr_params = dict(use_fft=True, return_itc=False, average=True, decim=2, n_jobs=4, verbose=True)

    freqs = np.arange(2,31,1)  # the frequency range over which we perform the analysis
    n_cycles = freqs / 2  # the length of sliding window in cycle units. 
    time_bandwidth = 2.0  # '(2deltaTdeltaF) number of DPSS tapers to be used + 1.'
                        # 'it relates to the temporal (deltaT) and spectral (deltaF)' 
                        # 'smoothing'
                        # 'the more tapers, the more smooth'->useful for high freq data
    baseline = [-0.3, -0.1]  # baseline for TFRs are longer than for ERPs
    
    tfr_slow_cue_both = epochs['cue_onset_right','cue_onset_left'].compute_tfr(
                                                    method='multitaper',
                                                    freqs=freqs, 
                                                    n_cycles=n_cycles,
                                                    time_bandwidth=time_bandwidth, 
                                                    **tfr_params                                                  
                                                    )
    mne.time_frequency.write_tfrs(deriv_fname_both, tfr_slow_cue_both, overwrite=True) 

    tfr_slow_cue_right = epochs['cue_onset_right'].compute_tfr(
                                                    method='multitaper',
                                                    freqs=freqs, 
                                                    n_cycles=n_cycles,
                                                    time_bandwidth=time_bandwidth, 
                                                    **tfr_params                                                  
                                                    )
    mne.time_frequency.write_tfrs(deriv_fname_right, tfr_slow_cue_right, overwrite=True)                                                

    tfr_slow_cue_left = mne.time_frequency.tfr_multitaper(epochs['cue_onset_left'],  
                                                    freqs=freqs, 
                                                    n_cycles=n_cycles,
                                                    time_bandwidth=time_bandwidth, 
                                                    **tfr_params                                                  
                                                    )
    mne.time_frequency.write_tfrs(deriv_fname_left, tfr_slow_cue_right, overwrite=True)                                                

    # Plot TFR on all sensors and check
    fig_plot_topo_both = tfr_slow_cue_both.plot_topo(tmin=-.5, 
                                                    tmax=1.5, 
                                                    baseline=baseline, 
                                                    mode='percent',
                                                    fig_facecolor='w', 
                                                    font_color='k',
                                                    vmin=-1, 
                                                    vmax=1, 
                                                    title=f'stim={stim}-TFR of power < 30Hz - cue both')
     
    fig_plot_topo_right = tfr_slow_cue_right.plot_topo(tmin=-.5, 
                                                    tmax=1.5, 
                                                    baseline=baseline, 
                                                    mode='percent',
                                                    fig_facecolor='w', 
                                                    font_color='k',
                                                    vmin=-1, 
                                                    vmax=1, 
                                                    title=f'stim={stim}-TFR of power < 30Hz - cue right')
    fig_plot_topo_left = tfr_slow_cue_left.plot_topo(tmin=-.5, 
                                                    tmax=1.5,
                                                    baseline=baseline, 
                                                    mode='percent',
                                                    fig_facecolor='w', 
                                                    font_color='k',
                                                    vmin=-1, 
                                                    vmax=1, 
                                                    title=f'stim={stim}-TFR of power < 30Hz - cue left')
    
    report.add_figure(fig=fig_plot_topo_both, title=f'stim:{stim}, TFR of power < 30Hz - cue both',
                        caption=f'Time Frequency Representation for \
                        cue both- -0.5 to 1.5- baseline corrected {baseline}', 
                        tags=('tfr'),
                        section='TFR'  
                        )

    report.add_figure(fig=fig_plot_topo_right, title=f'stim:{stim}, TFR of power < 30Hz - cue right',
                        caption=f'Time Frequency Representation for \
                        cue right- -0.5 to 1.5- baseline corrected {baseline}', 
                        tags=('tfr'),
                        section='TFR'  
                        )
    
    report.add_figure(fig=fig_plot_topo_left, title=f'stim:{stim}, TFR of power < 30Hz - cue left',
                        caption=f'Time Frequency Representation for \
                            cue left- -0.5 to 1.5- baseline corrected {baseline}', 
                        tags=('tfr'),
                        section='TFR'  
                        )

    return epochs, tfr_slow_cue_both, tfr_slow_cue_right, tfr_slow_cue_left, report

def calculate_grand_average(stim, all_subs_tfr_slow_cue_both_ls, all_subs_tfr_slow_cue_right_ls, all_subs_tfr_slow_cue_left_ls):
    """this function calculates grand average of all tfrs, generates proper names and saves them"""

    if stim:
        deriv_fname_group_both = op.join(deriv_folder_group, deriv_group_basename
                               + '_both_' + stim_suffix + '_' + deriv_suffix + extension) 
        deriv_fname_group_right = op.join(deriv_folder_group, deriv_group_basename
                               + '_right_' + stim_suffix + '_' + deriv_suffix + extension) 
        deriv_fname_group_left = op.join(deriv_folder_group, deriv_group_basename
                               + '_left_' + stim_suffix + '_' + deriv_suffix + extension)  
    else:
        deriv_fname_group_both = op.join(deriv_folder_group, deriv_group_basename 
                               + '_both_' + stim_suffix + '_' + deriv_suffix + extension) 
        deriv_fname_group_right = op.join(deriv_folder_group, deriv_group_basename 
                               + '_right_' + no_stim_suffix + '_' + deriv_suffix + extension)  
        deriv_fname_group_left = op.join(deriv_folder_group, deriv_group_basename 
                               + '_left_' + no_stim_suffix + '_' + deriv_suffix + extension)  
        
    grand_avg_cue_both = mne.grand_average(all_subs_tfr_slow_cue_both_ls)    
    grand_avg_cue_both.write_tfr(deriv_fname_group_both)

    grand_avg_cue_right = mne.grand_average(all_subs_tfr_slow_cue_right_ls) 
    grand_avg_cue_both.write_tfr(deriv_fname_group_right)

    grand_avg_cue_left = mne.grand_average(all_subs_tfr_slow_cue_left_ls)      
    grand_avg_cue_both.write_tfr(deriv_fname_group_left)

    return grand_avg_cue_both, grand_avg_cue_right, grand_avg_cue_left                               


def representative_sensors_second_plot(grand_avg_cue_right, grand_avg_cue_left, report):
    # ========================================= SECOND PLOT (REPRESENTATIVE SENSROS) ====================================

    # Plot TFR for representative sensors - same in all participants
    fig_tfr, axis = plt.subplots(6, 2, figsize = (30, 10))
    occipital_channels = ['O1', 'PO3', 'O2', 'PO4', 'Oz', 'POz']
    baseline = [-.3, -.1] # [-0.5, -0.2]

    for idx, ch in enumerate(occipital_channels):
        grand_avg_cue_left.plot(picks=ch, 
                                baseline=baseline,
                                mode='percent', 
                                tmin=-.5, 
                                tmax=1.5,
                                vmin=-.75, 
                                vmax=.75,
                                axes=axis[idx,0], 
                                show=False)
        axis[idx, 0].set_title(f'stim={stim}-cue left-{ch}') 
        # axis[idx, 1].set_xlabel('')        
        grand_avg_cue_right.plot(picks=ch,
                                baseline=baseline,
                                mode='percent', 
                                tmin=-.5, 
                                tmax=1.5,
                                vmin=-.75, 
                                vmax=.75, 
                                axes=axis[idx,1],
                                show=False)
        axis[idx, 1].set_title(f'stim={stim}-cue right-{ch}') 
        # axis[idx, 1].set_ylabel('') 
        # axis[idx, 1].set_xlabel('') 
            
    # axis[0, 1].set_xlabel('Time (s)')  # Remove x-axis label for top plots
    # axis[1, 1].set_xlabel('Time (s)')

    fig_tfr.set_tight_layout(True)
    plt.show()      

    report.add_figure(fig=fig_tfr, title=f'stim:{stim}, TFR on two sensors',
                        caption='Group Time Frequency Representation on \
                        right and left sensors', 
                        tags=('tfr'),
                        section='TFR'
                        )

    return report

def peak_alpha_calculation_third_plot(occipital_channels, grand_avg_cue_right, grand_avg_cue_left, epochs, report):
    # ========================================= PEAK ALPHA FREQUENCY (PAF) AND THIRD PLOT ====================================

    # Crop post stim alpha
    tfr_slow_cue_right_post_stim = grand_avg_cue_right.copy().crop(tmin=.3,tmax=.8,fmin=8, fmax=14).pick(occipital_channels)
    tfr_slow_cue_left_post_stim = grand_avg_cue_left.copy().crop(tmin=.3,tmax=.8,fmin=8, fmax=14).pick(occipital_channels)

    # Find the frequency with the highest power by averaging over sensors and time points (data)
    freq_idx_right = np.argmax(np.mean(np.abs(tfr_slow_cue_right_post_stim.data), axis=(0,2)))
    freq_idx_left = np.argmax(np.mean(np.abs(tfr_slow_cue_left_post_stim.data), axis=(0,2)))

    # Get the corresponding frequencies
    peak_freq_cue_right = tfr_slow_cue_right_post_stim.freqs[freq_idx_right]
    peak_freq_cue_left = tfr_slow_cue_left_post_stim.freqs[freq_idx_left]

    peak_alpha_freq = np.average([peak_freq_cue_right, peak_freq_cue_left])
    peak_alpha_freq_range = np.arange(peak_alpha_freq-2, peak_alpha_freq+3)  # for MI calculations
    # np.savez(peak_alpha_fname, **{'peak_alpha_freq':peak_alpha_freq, 'peak_alpha_freq_range':peak_alpha_freq_range})

    # Plot psd and show the peak alpha frequency for this participant
    n_fft = int((epochs.tmax - epochs.tmin)*1000)
    psd_params = dict(picks=occipital_channels, method="welch", fmin=1, fmax=60, n_jobs=4, verbose=True, n_fft=n_fft, n_overlap=int(n_fft/2))
    psd_slow_right_post_stim = epochs['cue_onset_right','cue_onset_left'].copy().compute_psd(**psd_params)

    # Average across epochs and get data
    psd_slow_right_post_stim_avg = psd_slow_right_post_stim.average()
    psds, freqs = psd_slow_right_post_stim_avg.get_data(return_freqs=True)
    psds_mean = psds.mean(axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs[0:int(len(freqs)/2)], psds_mean[0:int(len(freqs)/2)], color='black')  # remove frequencies higher than 30Hz for plotting
    ymin, ymax = ax.get_ylim()
    # Indicate peak_alpha_freq_range with a gray shadow
    ax.axvline(x=peak_alpha_freq_range[0], 
                color='gray', 
                linestyle='--', 
                linewidth=2)
    ax.axvline(x=peak_alpha_freq_range[-1], 
                color='gray', 
                linestyle='--', 
                linewidth=2)
    ax.fill_betweenx([ymin, ymax],
                    peak_alpha_freq_range[0], 
                    peak_alpha_freq_range[-1], 
                    color='lightgray', 
                    alpha=0.5)
    ax.text(np.max(freqs)-5, 
            np.min(psds_mean)*3, 
            f'PAF = {peak_alpha_freq} Hz', 
            color='black', 
            ha='right', 
            va='bottom')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (T/m)^2/Hz')
    plt.title(f'Group {epoching} onset- PAF = {peak_alpha_freq} Hz- stim={stim}')

    plt.grid(True)
    fig_peak_alpha = plt.gcf()
    plt.show()

    report.add_figure(fig=fig_peak_alpha, title=f'stim:{stim}, PSD and PAF',
                        caption='range of peak alpha frequency on \
                        occipital channels', 
                        tags=('tfr'),
                        section='TFR'  
                        )

    return peak_alpha_freq_range, report

def MI_calculation_overtime_plot(tfr_params, peak_alpha_freq_range, grand_avg_cue_right, grand_avg_cue_left, occipital_channels, report):
    # ================================== B. MI over time - poststim alpha topomap ==================================

    freqs = peak_alpha_freq_range  # peak frequency range calculated earlier
    n_cycles = freqs / 2  # the length of sliding window in cycle units. 
    time_bandwidth = 2.0 
                        
                        # figure out how to only do alpha peak for grand averages.
    tfr_right_peak_alpha_all_chans = grand_avg_cue_right.filter(alpha_PAF_range, 
                                                    )                                                
    tfr_left_peak_alpha_all_chans = mne.time_frequency.tfr_multitaper(epochs['cue_onset_left'],  
                                                    freqs=freqs, 
                                                    n_cycles=n_cycles,
                                                    time_bandwidth=time_bandwidth, 
                                                    **tfr_params,
                                                    )   

    # Crop tfrs to post-stim alpha and right sensors
    tfr_right_post_stim_alpha_occ_chans = tfr_right_peak_alpha_all_chans.copy().pick(occipital_channels).crop(tmin=.3, tmax=.8)
    tfr_left_post_stim_alpha_occ_chans = tfr_left_peak_alpha_all_chans.copy().pick(occipital_channels).crop(tmin=.3, tmax=.8)

    # Calculate power modulation for attention right and left (always R - L)
    tfr_alpha_MI_occ_chans = tfr_right_post_stim_alpha_occ_chans.copy()
    tfr_alpha_MI_occ_chans.data = (tfr_right_post_stim_alpha_occ_chans.data - tfr_left_post_stim_alpha_occ_chans.data) \
                                / (tfr_right_post_stim_alpha_occ_chans.data + tfr_left_post_stim_alpha_occ_chans.data)  # shape: #channels, #freqs, #time points

    # Average across time points and alpha frequencies
    tfr_avg_alpha_MI_occ_chans_power = np.mean(tfr_alpha_MI_occ_chans.data, axis=(1,2))   # the order of channels is the same as right_sensors (I double checked)

    # Save to dataframe
    MI_occ_chans_df = pd.DataFrame({'MI': tfr_avg_alpha_MI_occ_chans_power,
                                    'ch_names': occipital_channels})  

    # Plot MI on topoplot with highlighted ROI sensors
    tfr_alpha_modulation_power = tfr_right_peak_alpha_all_chans.copy()
    tfr_alpha_modulation_power.data = (tfr_right_peak_alpha_all_chans.data - tfr_left_peak_alpha_all_chans.data) \
                                    / (tfr_right_peak_alpha_all_chans.data + tfr_left_peak_alpha_all_chans.data)

    # Plot MI avg across ROI over time
    fig, axs = plt.subplots(figsize=(12, 6))

    # Plot average power and std for tfr_alpha_MI_left_ROI
    axs.plot(tfr_alpha_MI_occ_chans.times, tfr_alpha_MI_occ_chans.data.mean(axis=(0, 1)), label='Average MI', color='red')
    axs.fill_between(tfr_alpha_MI_occ_chans.times,
                        tfr_alpha_MI_occ_chans.data.mean(axis=(0, 1)) - tfr_alpha_MI_occ_chans.data.std(axis=(0, 1)),
                        tfr_alpha_MI_occ_chans.data.mean(axis=(0, 1)) + tfr_alpha_MI_occ_chans.data.std(axis=(0, 1)),
                        color='red', alpha=0.3, label='Standard Deviation')
    axs.set_title(f'stim={stim}- MI on occipital and parietal channels')
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Average MI (PAF)')
    axs.set_ylim(min(tfr_alpha_MI_occ_chans.data.mean(axis=(0, 1))) - 0.3, 
                        max(tfr_alpha_MI_occ_chans.data.mean(axis=(0, 1))) + 0.3)
    axs.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure in a variable
    fig_mi_overtime = fig

    # Show plot (optional)
    plt.show()

    report.add_figure(fig=fig_mi_overtime, title=f'stim:{stim}, MI over time',
                caption='MI average on occipital channels \
                in PAF ', 
                tags=('mi'),
                section='MI'  
                )

    
    return tfr_alpha_MI_occ_chans, report


# =================================================================================================================
# BIDS settings: fill these out 
subject_list = ['101', '102', '107', '108', '110', '112', '103', 'group'] # all subjects
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
stim_suffix = 'stim'
no_stim_suffix = 'no-stim'
extension = '.fif'

stim_segments_ls = [False, True]
epoching_list = ['cue']#, 'stim']  # epoching on cue onset or stimulus onset

pilot = False  # is it pilot data or real data?
summary_rprt = True  # do you want to add evokeds figures to the summary report?
platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
test_plot = False

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
report_fname = op.join(report_folder, 'subs_101-102-107-108-110-112-103_170225.hdf5')
html_report_fname = op.join(report_folder, 'subs_101-102-107-108-110-112-103_170225.html')
report = mne.Report(title='subs_101-102-107-108-110-112-103')

# Specify specific file names
ROI_dir = op.join(project_root, 'derivatives/lateralisation-indices')
peak_alpha_fname = op.join(ROI_dir, 'group_peak_alpha.npz')  # 2 numpy arrays saved into an uncompressed file

# Select ROI sensors
occipital_channels = ['PO4', 'POz', 'PO3']

tfr_params = dict(use_fft=True, return_itc=False, average=True, decim=2, n_jobs=4, verbose=True)

all_subs_tfr_slow_cue_both_ls = []
all_subs_tfr_slow_cue_right_ls = []
all_subs_tfr_slow_cue_left_ls = []

for subject in subject_list: 
    for epoching in epoching_list:
        input_suffix = epoching + '-epo'
        deriv_suffix = epoching + '-tfr'
        evoked_list = []

        for stim in stim_segments_ls:
            print(f'Reading stim:{stim}')
            bids_path = BIDSPath(subject=subject, session=session,
                        task=task, run=run, root=bids_root, 
                        datatype ='eeg', suffix=eeg_suffix)
            deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results

            (epochs, tfr_slow_cue_both, tfr_slow_cue_right, 
                tfr_slow_cue_left, report) = tfr_calculation_first_plot(stim, report)
            
            all_subs_tfr_slow_cue_both_ls.append(tfr_slow_cue_both)
            all_subs_tfr_slow_cue_right_ls.append(tfr_slow_cue_right) 
            all_subs_tfr_slow_cue_left_ls.append(tfr_slow_cue_left)  

            # grand average all tfrs


            report = representative_sensors_second_plot(tfr_slow_cue_right, 
                                                        tfr_slow_cue_left, 
                                                        report)
            peak_alpha_freq_range, report = peak_alpha_calculation_third_plot(occipital_channels, 
                                                                            tfr_slow_cue_right, 
                                                                            tfr_slow_cue_left, 
                                                                            epochs,
                                                                            report)
            
            tfr_alpha_MI_occ_chans, report = MI_calculation_overtime_plot(tfr_params, 
                                                                    peak_alpha_freq_range, 
                                                                    epochs, 
                                                                    occipital_channels,
                                                                    report)

report.save(report_fname, overwrite=True)
report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks





