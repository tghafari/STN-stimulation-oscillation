# -*- coding: utf-8 -*-
"""
===============================================
A02. calculate MI and select ROI

This code will:

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
    7. plot MI topographically 


written by Tara Ghafari
==============================================
ToDos:

questions?

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
    
    tfr_slow_cue_both = mne.time_frequency.tfr_multitaper(epochs['cue_onset_right','cue_onset_left'],  
                                                    freqs=freqs, 
                                                    n_cycles=n_cycles,
                                                    time_bandwidth=time_bandwidth, 
                                                    **tfr_params                                                  
                                                    )
    mne.time_frequency.write_tfrs(deriv_fname_both, tfr_slow_cue_both, overwrite=True) 

    tfr_slow_cue_right = mne.time_frequency.tfr_multitaper(epochs['cue_onset_right'],  
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

def representative_sensors_second_plot(tfr_slow_cue_right, tfr_slow_cue_left, report):
    # ========================================= SECOND PLOT (REPRESENTATIVE SENSROS) ====================================

    # Plot TFR for representative sensors - same in all participants
    fig_tfr, axis = plt.subplots(6, 2, figsize = (30, 10))
    occipital_channels = ['O1', 'PO3', 'O2', 'PO4', 'Oz', 'POz']
    baseline = [-.3, -.1] # [-0.5, -0.2]

    for idx, ch in enumerate(occipital_channels):
        tfr_slow_cue_left.plot(picks=ch, 
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
        tfr_slow_cue_right.plot(picks=ch,
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
                        caption='Time Frequency Representation on \
                        right and left sensors', 
                        tags=('tfr'),
                        section='TFR'
                        )

    return report

def peak_alpha_calculation_third_plot(occipital_channels, tfr_slow_cue_right, tfr_slow_cue_left, epochs, report):
    # ========================================= PEAK ALPHA FREQUENCY (PAF) AND THIRD PLOT ====================================

    # Crop post stim alpha
    tfr_slow_cue_both_post_stim = tfr_slow_cue_both.copy().crop(tmin=.3,tmax=.8,fmin=6, fmax=14).pick(occipital_channels)
    tfr_slow_cue_right_post_stim = tfr_slow_cue_right.copy().crop(tmin=.3,tmax=.8,fmin=6, fmax=14).pick(occipital_channels)
    tfr_slow_cue_left_post_stim = tfr_slow_cue_left.copy().crop(tmin=.3,tmax=.8,fmin=8, fmax=14).pick(occipital_channels)

    # Find the frequency with the highest power by averaging over sensors and time points (data)
    freq_idx_both = np.argmax(np.mean(np.abs(tfr_slow_cue_both_post_stim.data), axis=(0,2)))
    freq_idx_right = np.argmax(np.mean(np.abs(tfr_slow_cue_right_post_stim.data), axis=(0,2)))
    freq_idx_left = np.argmax(np.mean(np.abs(tfr_slow_cue_left_post_stim.data), axis=(0,2)))

    # Get the corresponding frequencies
    peak_freq_cue_both = tfr_slow_cue_both_post_stim.freqs[freq_idx_both]
    peak_freq_cue_right = tfr_slow_cue_right_post_stim.freqs[freq_idx_right]
    peak_freq_cue_left = tfr_slow_cue_left_post_stim.freqs[freq_idx_left]
    print(peak_freq_cue_both)

    peak_alpha_freq_no_both = np.average([peak_freq_cue_right, peak_freq_cue_left])
    peak_alpha_freq_range_no_both = np.arange(peak_alpha_freq_no_both-2, peak_alpha_freq_no_both+3)  # for MI calculations
    peak_alpha_freq_range = np.arange(peak_freq_cue_both-2, peak_freq_cue_both+3)  # ±2 Hz around the peak
    # np.savez(peak_alpha_fname, **{'peak_alpha_freq':peak_alpha_freq, 'peak_alpha_freq_range':peak_alpha_freq_range})
    
    # Plot psd and show the peak alpha frequency for this participant
    peak_alph_both_occ = tfr_slow_cue_both.copy().pick(occipital_channels)
    avg_power = np.mean(peak_alph_both_occ.data, axis=(0, 2))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(peak_alph_both_occ.freqs, avg_power, color='black')
    ymin, ymax = ax.get_ylim()
    ax.axvline(x=peak_alpha_freq_range[0], color='gray', linestyle='--', linewidth=2)
    ax.axvline(x=peak_alpha_freq_range[-1], color='gray', linestyle='--', linewidth=2)
    ax.fill_betweenx([ymin, ymax], peak_alpha_freq_range[0], peak_alpha_freq_range[-1],
                     color='lightgray', alpha=0.5)
    # ax.set_xlabel('Frequency (Hz)')
    # ax.set_ylabel('Power')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (T/m)^2/Hz')
    plt.title(f'{epoching} onset- PAF = {peak_freq_cue_both} Hz- stim={stim}')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

    fig_peak_alpha = plt.gcf()
    plt.show()

    report.add_figure(fig=fig_peak_alpha, title=f'stim:{stim}, PSD and PAF',
                        caption='range of peak alpha frequency on \
                        occipital channels', 
                        tags=('tfr'),
                        section='TFR'  
                        )

    return peak_alpha_freq_range, report

def topographic_maps_fourth_plot(peak_alpha_freq_range, tfr_slow_cue_both, tfr_slow_cue_right, tfr_slow_cue_left, report):
    # ========================================= TOPOGRAPHIC MAPS AND FOURTH PLOT ============================================
    # Plot post cue peak alpha range topographically
    topomap_params = dict(fmin=peak_alpha_freq_range[0], 
                        fmax=peak_alpha_freq_range[-1],
                        tmin=.3,
                        tmax=.8,
                        vlim=(-.5,.5),
                        baseline=(-0.3, -0.1), # only baseline that's tuple (not list)
                        mode='percent')

    fig_topo, axis = plt.subplots(1, 3, figsize=(8, 4))
    tfr_slow_cue_both.plot_topomap(**topomap_params,
                            axes=axis[0],
                            show=False)    
    tfr_slow_cue_left.plot_topomap(**topomap_params,
                            axes=axis[1],
                            show=False)
    tfr_slow_cue_right.plot_topomap(**topomap_params,
                                axes=axis[2],
                                show=False)
    axis[0].title.set_text('cue both')
    axis[1].title.set_text('cue left')
    axis[2].title.set_text('cue right')
    fig_topo.suptitle(f"Post stim alpha (PAF)-stim={stim}")
    fig_topo.set_tight_layout(True)
    plt.show()

    report.add_figure(fig=fig_topo, title=f'stim:{stim}, post stim alpha',
                            caption='PAF range, 0.3-0.8sec, \
                            baseline corrected (-0.3, -0.1)', 
                            tags=('tfr'),
                            section='TFR'  # only in ver 1.1
                            )   
    return report

def MI_calculation_fifth_plot(tfr_params, peak_alpha_freq_range, epochs, occipital_channels, report):
    # ================================== B. MI on occipital (+ parietal) channels - poststim alpha topomap ==================================

    freqs = peak_alpha_freq_range  # peak frequency range calculated earlier
    n_cycles = freqs / 2  # the length of sliding window in cycle units. 
    time_bandwidth = 2.0 
                        
    tfr_right_peak_alpha_all_chans = mne.time_frequency.tfr_multitaper(epochs['cue_onset_right'],  
                                                    freqs=freqs, 
                                                    n_cycles=n_cycles,
                                                    time_bandwidth=time_bandwidth, 
                                                    **tfr_params,
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

    fig, ax = plt.subplots()
    fig_mi = tfr_alpha_modulation_power.plot_topomap(tmin=.3, 
                                                    tmax=.8, 
                                                    fmin=peak_alpha_freq_range[0],
                                                    fmax=peak_alpha_freq_range[-1],
                                                    vlim=(-.2,.2),
                                                    show=False, axes=ax)

    # Plot markers for the sensors in occipital (+ parietal) channels
    for chan in occipital_channels:
        ch_idx = tfr_alpha_modulation_power.info['ch_names'].index(chan)
        x, y = tfr_alpha_modulation_power.info['chs'][ch_idx]['loc'][:2]
        ax.plot(x, y, 'ko', markerfacecolor='none', markersize=10)
                                    
    fig_mi.suptitle(f'stim:{stim}- attention right - attention left (PAF range on occipital channels)')
    plt.show()  

    report.add_figure(fig=fig_mi, title=f'stim:{stim}, MI and ROI',
                            caption='MI on PAF range and \
                            occipital channels (0.3 to 0.8 sec)', 
                            tags=('mi'),
                            section='MI'  
                            )  
    
    return tfr_alpha_MI_occ_chans, report

def MI_overtime_sixth_plot(tfr_alpha_MI_occ_chans, report):
    # ========================================= MI OVER TIME AND SIXTH PLOT =======================================
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

    return report

# =================================================================================================================
# BIDS settings: fill these out 
subject = '105'
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

# project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
# bids_root = op.join(project_root, 'data', 'BIDS')

# for bear outage
project_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD'  # only for bear outage time
bids_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/data/BIDS'

# Specify specific file names
ROI_dir = op.join(project_root, 'derivatives/lateralisation-indices')
peak_alpha_fname = op.join(ROI_dir, f'sub-{subject}_peak_alpha.npz')  # 2 numpy arrays saved into an uncompressed file

# Select ROI sensors
occipital_channels = ['PO4', 'POz', 'PO3']

tfr_params = dict(use_fft=True, return_itc=False, average=True, decim=2, n_jobs=4, verbose=True)

# Epoch stim segments and add to report
report_root = op.join(project_root, 'derivatives/reports')  
report_folder = op.join(report_root , 'sub-' + subject)
report_fname = op.join(report_folder, 
                    f'sub-{subject}_130325.hdf5')    # it is in .hdf5 for later adding images
html_report_fname = op.join(report_folder, f'sub-{subject}_130325.html')

report = mne.open_report(report_fname)

for epoching in epoching_list:
    input_suffix = 'epo-' + epoching
    deriv_suffix = 'tfr-' + epoching
    evoked_list = []

    for stim in stim_segments_ls:
        print(f'Reading stim:{stim}')
        bids_path = BIDSPath(subject=subject, session=session,
                    task=task, run=run, root=bids_root, 
                    datatype ='eeg', suffix=eeg_suffix)
        deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results

        (epochs, tfr_slow_cue_both, tfr_slow_cue_right, 
            tfr_slow_cue_left, report) = tfr_calculation_first_plot(stim, report)
        
        report = representative_sensors_second_plot(tfr_slow_cue_right, 
                                                    tfr_slow_cue_left, 
                                                    report)
        peak_alpha_freq_range, report = peak_alpha_calculation_third_plot(occipital_channels, 
                                                                        tfr_slow_cue_right, 
                                                                        tfr_slow_cue_left, 
                                                                        epochs,
                                                                        report)
        report = topographic_maps_fourth_plot(peak_alpha_freq_range, 
                                                tfr_slow_cue_both, 
                                                tfr_slow_cue_right, 
                                                tfr_slow_cue_left, 
                                                report
                                                )
        
        tfr_alpha_MI_occ_chans, report = MI_calculation_fifth_plot(tfr_params, 
                                                                peak_alpha_freq_range, 
                                                                epochs, 
                                                                occipital_channels,
                                                                report)
        report = MI_overtime_sixth_plot(tfr_alpha_MI_occ_chans, report)  # Don't plot now, it looks terrible 

report.save(report_fname, overwrite=True)
report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks





