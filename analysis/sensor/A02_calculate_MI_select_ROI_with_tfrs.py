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
    can we actually rely on posterior sensors and alpha lateralisation and alpha peak? given the smearing of signal in eeg?

"""

import os.path as op
import os
import numpy as np
import pandas as pd

import mne
from mne_bids import BIDSPath
import matplotlib.pyplot as plt

# BIDS settings: fill these out 
subject = '01'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
eeg_extension = '.vhdr'
input_suffix = 'epo'
deriv_suffix = 'tfr'
extension = '.fif'

pilot = False  # is it pilot data or real data?
summary_rprt = True  # do you want to add evokeds figures to the summary report?
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
ROI_dir = op.join(project_root, 'derivatives/lateralisation-indices')
bids_path = BIDSPath(subject=subject, session=session,
                     task=task, run=run, root=bids_root, 
                     datatype ='eeg', suffix=eeg_suffix)
deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)  # RDS folder for results
input_fname = op.join(deriv_folder, bids_path.basename + '_' + input_suffix + extension)
deriv_fname = op.join(deriv_folder, bids_path.basename + '_' + deriv_suffix + extension)  # prone to change if annotation worked for eeg brainvision

peak_alpha_fname = op.join(ROI_dir, f'sub-{subject}_peak_alpha.npz')  # 2 numpy arrays saved into an uncompressed file

# Read epoched data
epochs = mne.read_epochs(input_fname, verbose=True, preload=True)  # epochs are from -.7 to 1.7sec

# ========================================= TFR CALCULATIONS AND FIRST PLOT (PLOT_TOPO) ====================================
# Calculate tfr for post cue alpha
tfr_params = dict(use_fft=True, return_itc=False, average=True, decim=2, n_jobs=4, verbose=True)

freqs = np.arange(2,31,1)  # the frequency range over which we perform the analysis
n_cycles = freqs / 2  # the length of sliding window in cycle units. 
time_bandwidth = 2.0  # '(2deltaTdeltaF) number of DPSS tapers to be used + 1.'
                      # 'it relates to the temporal (deltaT) and spectral (deltaF)' 
                      # 'smoothing'
                      # 'the more tapers, the more smooth'->useful for high freq data
                      
tfr_slow_cue_right = mne.time_frequency.tfr_multitaper(epochs['cue_onset_right'],  
                                                  freqs=freqs, 
                                                  n_cycles=n_cycles,
                                                  time_bandwidth=time_bandwidth, 
                                                  **tfr_params                                                  
                                                  )
                                                
tfr_slow_cue_left = mne.time_frequency.tfr_multitaper(epochs['cue_onset_left'],  
                                                  freqs=freqs, 
                                                  n_cycles=n_cycles,
                                                  time_bandwidth=time_bandwidth, 
                                                  **tfr_params                                                  
                                                  )

# Plot TFR on all sensors and check
fig_plot_topo_right = tfr_slow_cue_right.plot_topo(tmin=-.5, 
                                                   tmax=1.5, 
                                                   baseline=[-.5,-.2], 
                                                   mode='percent',
                                                   fig_facecolor='w', 
                                                   font_color='k',
                                                   vmin=-1, 
                                                   vmax=1, 
                                                   title='TFR of power < 30Hz - cue right')
fig_plot_topo_left = tfr_slow_cue_left.plot_topo(tmin=-.5, 
                                                 tmax=1.5,
                                                 baseline=[-.5,-.2], 
                                                 mode='percent',
                                                 fig_facecolor='w', 
                                                 font_color='k',
                                                 vmin=-1, 
                                                 vmax=1, 
                                                 title='TFR of power < 30Hz - cue left')

# ========================================= SECOND PLOT (REPRESENTATIVE SENSROS) ====================================
# Plot TFR for representative sensors - same in all participants
fig_tfr, axis = plt.subplots(2, 2, figsize = (7, 7))
sensors = ['C5','O2']

for idx, sensor in enumerate(sensors):
    tfr_slow_cue_left.plot(picks=sensor, 
                            baseline=[-.5,-.2],
                            mode='percent', 
                            tmin=-.5, 
                            tmax=1.5,
                            vmin=-.75, 
                            vmax=.75,
                            axes=axis[idx,0], 
                            show=False)
    axis[idx, 0].set_title(f'cue left-{sensor}')        
    tfr_slow_cue_right.plot(picks=sensor,
                            baseline=[-.5,-.2],
                            mode='percent', 
                            tmin=-.5, 
                            tmax=1.5,
                            vmin=-.75, 
                            vmax=.75, 
                            axes=axis[idx,1], 
                            show=False)
    axis[idx, 1].set_title(f'cue right-{sensor}') 
        
axis[0, 0].set_ylabel('left sensors')  
axis[1, 0].set_ylabel('right sensors')  
axis[0, 1].set_ylabel('left sensors')  
axis[1, 1].set_ylabel('right sensors')
axis[0, 0].set_xlabel('')  # Remove x-axis label for top plots
axis[0, 1].set_xlabel('')

fig_tfr.set_tight_layout(True)
plt.show()      

# ========================================= PEAK ALPHA FREQUENCY (PAF) AND THIRD PLOT ====================================
# Select occipital sensors
occipital_channels = ['O2', 'Oz', 'O1', 'PO8', 'PO4', 'POz', 'PO3', 'PO7', 'P8', 'P6', 'P4', 'P2',
                       'Pz', 'P1', 'P3', 'P5', 'P7', 'TP10', 'TP8', 'CP6', 'CP4', 'CP2', 'CPz',
                       'CP1', 'CP3', 'CP5', 'TP7', 'TP9']

# Crop post stim alpha
tfr_slow_cue_right_post_stim = tfr_slow_cue_right.copy().crop(tmin=.3,tmax=.8,fmin=4, fmax=14).pick(occipital_channels)
tfr_slow_cue_left_post_stim = tfr_slow_cue_left.copy().crop(tmin=.3,tmax=.8,fmin=4, fmax=14).pick(occipital_channels)

# Find the frequency with the highest power by averaging over sensors and time points (data)
freq_idx_right = np.argmax(np.mean(np.abs(tfr_slow_cue_right_post_stim.data), axis=(0,2)))
freq_idx_left = np.argmax(np.mean(np.abs(tfr_slow_cue_left_post_stim.data), axis=(0,2)))

# Get the corresponding frequencies
peak_freq_cue_right = tfr_slow_cue_right_post_stim.freqs[freq_idx_right]
peak_freq_cue_left = tfr_slow_cue_left_post_stim.freqs[freq_idx_left]

peak_alpha_freq = np.average([peak_freq_cue_right, peak_freq_cue_left])
peak_alpha_freq_range = np.arange(peak_alpha_freq-2, peak_alpha_freq+3)  # for MI calculations
np.savez(peak_alpha_fname, **{'peak_alpha_freq':peak_alpha_freq, 'peak_alpha_freq_range':peak_alpha_freq_range})

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
plt.title(f'PSDs- PAF = {peak_alpha_freq} Hz')

plt.grid(True)
fig_peak_alpha = plt.gcf()
plt.show()

# ========================================= TOPOGRAPHIC MAPS AND FOURTH PLOT ============================================
# Plot post cue peak alpha range topographically
topomap_params = dict(fmin=peak_alpha_freq_range[0], 
                      fmax=peak_alpha_freq_range[-1],
                      tmin=.3,
                      tmax=.8,
                      vlim=(-.5,.5),
                      baseline=(-.5, -.2), 
                      mode='percent')

fig_topo, axis = plt.subplots(1, 2, figsize=(7, 4))
tfr_slow_cue_left.plot_topomap(**topomap_params,
                           axes=axis[0],
                           show=False)
tfr_slow_cue_right.plot_topomap(**topomap_params,
                            axes=axis[1],
                            show=False)
axis[0].title.set_text('cue left')
axis[1].title.set_text('cue right')
fig_topo.suptitle("Post stim alpha (PAF)")
fig_topo.set_tight_layout(True)
plt.show()

# ================================== B. MI on occipital (+ parietal) channels - poststim alpha topomap ==================================
tfr_alpha_params = dict(use_fft=True, return_itc=False, average=True, decim=2, n_jobs=4, verbose=True)
tfr_params = dict(use_fft=True, return_itc=False, average=True, decim=2, n_jobs=4, verbose=True)

freqs = peak_alpha_freq_range  # peak frequency range calculated earlier
n_cycles = freqs / 2  # the length of sliding window in cycle units. 
time_bandwidth = 2.0 
                      
tfr_right_alpha_all_chans = mne.time_frequency.tfr_multitaper(epochs['cue_onset_right'],  
                                                  freqs=freqs, 
                                                  n_cycles=n_cycles,
                                                  time_bandwidth=time_bandwidth, 
                                                  **tfr_params,
                                                  )                                                
tfr_left_alpha_all_chans = mne.time_frequency.tfr_multitaper(epochs['cue_onset_left'],  
                                                  freqs=freqs, 
                                                  n_cycles=n_cycles,
                                                  time_bandwidth=time_bandwidth, 
                                                  **tfr_params,
                                                  )   

# Crop tfrs to post-stim alpha and right sensors
tfr_right_post_stim_alpha_occ_chans = tfr_right_alpha_all_chans.copy().pick(occipital_channels).crop(tmin=.3, tmax=.8)
tfr_left_post_stim_alpha_occ_chans = tfr_left_alpha_all_chans.copy().pick(occipital_channels).crop(tmin=.3, tmax=.8)

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
tfr_alpha_modulation_power = tfr_right_alpha_all_chans.copy()
tfr_alpha_modulation_power.data = (tfr_right_alpha_all_chans.data - tfr_left_alpha_all_chans.data) \
                                / (tfr_right_alpha_all_chans.data + tfr_left_alpha_all_chans.data)

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
                                 
fig_mi.suptitle('attention right - attention left (PAF range on occipital channels)')
plt.show()  

# ========================================= MI OVER TIME AND SIXTH PLOT =======================================
# Don't plot now, it looks terrible 
# Plot MI avg across ROI over time
fig, axs = plt.subplots(figsize=(12, 6))

# Plot average power and std for tfr_alpha_MI_left_ROI
axs.plot(tfr_alpha_MI_occ_chans.times, tfr_alpha_MI_occ_chans.data.mean(axis=(0, 1)), label='Average MI', color='red')
axs.fill_between(tfr_alpha_MI_occ_chans.times,
                    tfr_alpha_MI_occ_chans.data.mean(axis=(0, 1)) - tfr_alpha_MI_occ_chans.data.std(axis=(0, 1)),
                    tfr_alpha_MI_occ_chans.data.mean(axis=(0, 1)) + tfr_alpha_MI_occ_chans.data.std(axis=(0, 1)),
                    color='red', alpha=0.3, label='Standard Deviation')
axs.set_title('MI on occipital and parietal channels')
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

# =================================================================================================================

if summary_rprt:

    report_root = op.join(project_root, 'derivatives/reports')  
    if not op.exists(op.join(report_root , 'sub-' + subject)):
        os.makedirs(op.join(report_root , 'sub-' + subject))
    report_folder = op.join(report_root , 'sub-' + subject)

    report_fname = op.join(report_folder, 
                        f'sub-{subject}_preproc_1.hdf5')    # it is in .hdf5 for later adding images
    html_report_fname = op.join(report_folder, f'sub-{subject}_preproc_1.html')

    report = mne.open_report(report_fname)

    report.add_figure(fig=fig_plot_topo_right, title='TFR of power < 30Hz - cue right',
                    caption='Time Frequency Representation for \
                    cue right- -0.5 to 1.5- baseline corrected', 
                    tags=('tfr'),
                    section='TFR'  # only in ver 1.1
                    )
    report.add_figure(fig=fig_plot_topo_left, title='TFR of power < 30Hz - cue left',
                    caption='Time Frequency Representation for \
                        cue left- -0.5 to 1.5- baseline corrected', 
                    tags=('tfr'),
                    section='TFR'  # only in ver 1.1
                    )
    report.add_figure(fig=fig_tfr, title='TFR on two sensors',
                    caption='Time Frequency Representation on \
                    right and left sensors', 
                    tags=('tfr'),
                    section='TFR'  # only in ver 1.1
                    )
    report.add_figure(fig=fig_peak_alpha, title='PSD and PAF',
                     caption='range of peak alpha frequency on \
                        occipital gradiometers', 
                     tags=('tfr'),
                     section='TFR'  # only in ver 1.1
                     )
    report.add_figure(fig=fig_topo, title='post stim alpha',
                     caption='PAF range, 0.3-0.8sec, \
                        baseline corrected', 
                     tags=('tfr'),
                     section='TFR'  # only in ver 1.1
                     )   
    report.add_figure(fig=fig_mi, title='MI and ROI',
                     caption='MI on PAF range and \
                        occipital channels (0.3 to 0.8 sec)', 
                     tags=('mi'),
                     section='MI'  
                     )  
#    report.add_figure(fig=fig_mi_overtime, title='MI over time',
#                     caption='MI average on occipital channels \
#                     in PAF ', 
#                     tags=('mi'),
#                     section='MI'  
#                     )

    report.save(report_fname, overwrite=True)
    report.save(html_report_fname, overwrite=True, open_browser=True)  # to check how the report looks





