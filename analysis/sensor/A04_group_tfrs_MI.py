"""
==============================================================
Group Analysis of TFRs, Peak Alpha Frequency, and MI over Time
==============================================================

This script:
  1. Reads each subject’s epoched data (for a given epoching type and stimulation condition).
  2. Computes time–frequency representations (TFRs) with multitaper (2–31 Hz).
  3. Saves the subject TFRs and adds sensor topography plots to an MNE Report.
  4. Aggregates subject TFRs by condition and computes group (grand) averages,
     which are then saved and plotted.
  5. Computes the peak alpha frequency (PAF) from the group grand averages and
     plots the power spectrum with the PAF range highlighted.
  6. Crops the group TFRs to the PAF range, computes a modulation index (MI) over time,
     and plots the MI time series.
  7. Saves all figures in a single MNE Report.

Author: tara.ghafari@psych.ox.ac.uk
"""

import os
import os.path as op
import numpy as np
import mne
from mne_bids import BIDSPath
import matplotlib.pyplot as plt

# ---------------------------
# Global Parameters
# ---------------------------
# Frequency analysis parameters
FREQS = np.arange(2, 32, 0.5)  # 2 to 31 Hz inclusive
N_CYCLES = FREQS / 2.0       # adaptive cycles
TIME_BANDWIDTH = 2.0
BASELINE = [-0.3, -0.1]

TFR_PARAMS = dict(
    method='multitaper',
    freqs=FREQS,
    return_itc=False,
    average=True,
    decim=2,
    n_jobs=4,
    verbose=True,
    n_cycles=N_CYCLES,
    time_bandwidth=TIME_BANDWIDTH,
    use_fft=True,
    zero_mean=True,
)

# ---------------------------
# Subject-level processing functions
# ---------------------------
def process_subject_tfr(stim_flag, deriv_folder, bids_path, input_suffix, deriv_suffix, extension, subj, report):
    """
    Load subject epochs and compute TFRs for three event types: cue both, cue right, and cue left.
    Saves TFR files and adds topography figures to the MNE Report.
    
    Parameters
    ----------
    stim_flag : bool
        True for stimulation condition; False for no_stim.
    deriv_folder : str
        Subject-specific derivative folder.
    bids_path : BIDSPath
        BIDSPath object for the subject.
    input_suffix : str
        Suffix for epoch file.
    deriv_suffix : str
        Suffix for TFR file.
    extension : str
        File extension.
    report : mne.Report
        MNE Report to append figures.
    
    Returns
    -------
    epochs : mne.Epochs
        Loaded epochs.
    tfr_dict : dict
        Dictionary with keys 'both', 'right', and 'left' mapping to the computed TFRs.
    report : mne.Report
        Updated report.
    """
    # Set file naming based on stim_flag
    if stim_flag:
        cond = 'stim'
        suffix = 'stim'
    else:
        cond = 'no_stim'
        suffix = 'no-stim'
    
    base = bids_path.basename
    input_fname = op.join(deriv_folder, f"{base}_{suffix}_{input_suffix}{extension}")
    fname_both = op.join(deriv_folder, f"{base}_both_{suffix}_{deriv_suffix}{extension}")
    fname_right = op.join(deriv_folder, f"{base}_right_{suffix}_{deriv_suffix}{extension}")
    fname_left = op.join(deriv_folder, f"{base}_left_{suffix}_{deriv_suffix}{extension}")

    # Load epochs
    epochs = mne.read_epochs(input_fname, preload=True, verbose=True)
    
    # Compute TFRs using compute_tfr (parameters defined in TFR_PARAMS)
    tfr_both = epochs['cue_onset_right', 'cue_onset_left'].compute_tfr(**TFR_PARAMS)
    tfr_right = epochs['cue_onset_right'].compute_tfr(**TFR_PARAMS)
    tfr_left = epochs['cue_onset_left'].compute_tfr(**TFR_PARAMS)
    
    # Save TFR files
    tfr_both.save(fname_both, overwrite=True)
    tfr_right.save(fname_right, overwrite=True)
    tfr_left.save(fname_left, overwrite=True)
    
    # Plot topographies with white background and add figures to report
    figs = {
        'cue both': tfr_both.plot_topo(tmin=-0.5, tmax=1.5, baseline=BASELINE, mode='percent',
                                       title=f'{cond}: TFR (cue both)', show=False,
                                       vmin=-0.75, vmax=0.75,
                                       fig_facecolor='w', font_color='k'),
        'cue right': tfr_right.plot_topo(tmin=-0.5, tmax=1.5, baseline=BASELINE, mode='percent',
                                         title=f'{cond}: TFR (cue right)', show=False,
                                         vmin=-0.75, vmax=0.75,
                                         fig_facecolor='w', font_color='k'),
        'cue left': tfr_left.plot_topo(tmin=-0.5, tmax=1.5, baseline=BASELINE, mode='percent',
                                        title=f'{cond}: TFR (cue left)', show=False,
                                        vmin=-0.75, vmax=0.75,
                                        fig_facecolor='w', font_color='k')
    }

    for key, fig in figs.items():
        report.add_figure(fig=fig, title=f'{cond} TFR {key}- {subj}',
                          caption=f'TFR (2-31 Hz) for {key} (baseline: {BASELINE})',
                          tags=('tfr',), section='TFR')
    
    tfr_dict = {'both': tfr_both, 'right': tfr_right, 'left': tfr_left}
    return epochs, tfr_dict, report

# ---------------------------
# Group-level processing functions
# ---------------------------
def compute_group_grand_avg(tfr_collection, deriv_folder_group, base_name, suffix, deriv_suffix, extension, report):
    """
    Compute and save the group grand average TFR for each cue type ('both', 'right', 'left'),
    and plot the topographies with added figures in the MNE Report.
    
    Parameters
    ----------
    tfr_collection : dict
        Dictionary with keys 'both', 'right', and 'left' mapping to lists of subject TFRs.
    deriv_folder_group : str
        Folder to save group-level derivative files.
    base_name : str
        Base filename for the group average.
    suffix : str
        Condition suffix ('stim' or 'no_stim').
    deriv_suffix : str
        Suffix for TFR derivative.
    extension : str
        File extension (e.g., '.fif').
    report : mne.Report
        MNE Report object to which the figures will be added.
        
    Returns
    -------
    group_avg : dict
        Dictionary with grand averages for each cue type.
    report : mne.Report
        Updated report with the topography figures added.
    """
    # Ensure BASELINE is defined (global constant)
    group_avg = {}
    for key in ['both', 'right', 'left']:
        # Compute the grand average for the current cue type
        avg = mne.grand_average(tfr_collection[key])
        # Build filename and save the TFR
        fname = op.join(deriv_folder_group, f"{base_name}_{key}_{suffix}_{deriv_suffix}{extension}")
        avg.save(fname, overwrite=True)
        group_avg[key] = avg
        
        # Generate topography plot for the grand average
        fig = avg.plot_topo(tmin=-0.5, tmax=1.5, baseline=BASELINE, mode='percent',
                            title=f'{suffix}: Grand Average TFR ({key})', show=False,
                            vmin=-0.75, vmax=0.75,
                            fig_facecolor='w', font_color='k')
        plt.show()

        report.add_figure(fig=fig,
                          title=f'{suffix} Grand Average TFR ({key})',
                          caption=f'Grand Average TFR (2-31 Hz) for {key} (baseline: {BASELINE})',
                          tags=('tfr', 'group'), section='TFR')
    return group_avg, report

def plot_group_representative_channel(grand_avg_right, grand_avg_left, cond_label, report):
    """
    Plot grand average TFR topographies on representative occipital sensors.
    
    Parameters
    ----------
    grand_avg_right : mne.time_frequency.AverageTFR
        Grand average TFR for cue right.
    grand_avg_left : mne.time_frequency.AverageTFR
        Grand average TFR for cue left.
    cond_label : str
        Condition label.
    report : mne.Report
        MNE Report to update.
        
    Returns
    -------
    report : mne.Report
    """
    occipital = ['PO3', 'PO4', 'POz']
    fig, axes = plt.subplots(len(occipital), 2, figsize=(50, 20))
    for i, ch in enumerate(occipital):
        grand_avg_left.plot(picks=ch, baseline=BASELINE, mode='percent',
                            tmin=-0.5, tmax=1.5, vmin=-0.75, vmax=0.75,
                            axes=axes[i, 0], show=False)
        axes[i, 0].set_title(f'{cond_label} cue left - {ch}')
        grand_avg_right.plot(picks=ch, baseline=BASELINE, mode='percent',
                             tmin=-0.5, tmax=1.5, vmin=-0.75, vmax=0.75,
                             axes=axes[i, 1], show=False)
        axes[i, 1].set_title(f'{cond_label} cue right - {ch}')    
    fig.tight_layout()
    plt.show()

    report.add_figure(fig=fig, title=f'{cond_label} Group TFR (Occipital Sensors)',
                      caption='Group TFR plots on selected occipital channels',
                      tags=('tfr', 'group'), section='TFR')
    return report

def plot_group_sensor_stim_nostim(group_avg, report):
    """Plot grand average TFR on representative occipital sensors."""

    occipital = ['PO3', 'PO4', 'POz']
    fig, axes = plt.subplots(len(occipital), 2, figsize=(50, 20))
    for i, ch in enumerate(occipital):
        group_avg['stim']['both'].plot(picks=ch, baseline=BASELINE, mode='percent',
                            tmin=-0.5, tmax=1.5, vmin=-0.75, vmax=0.75,
                            axes=axes[i, 0], show=False)
        axes[i, 0].set_title(f'stim cue both - {ch}')
        group_avg['no_stim']['both'].plot(picks=ch, baseline=BASELINE, mode='percent',
                             tmin=-0.5, tmax=1.5, vmin=-0.75, vmax=0.75,
                             axes=axes[i, 1], show=False)
        axes[i, 1].set_title(f'no stim- cue both - {ch}')    
    fig.tight_layout()
    plt.show()

    report.add_figure(fig=fig, title=f'Group TFR (Occipital Sensors)-stim vs no stim',
                      caption='Group TFR plots on selected occipital channels',
                      tags=('tfr', 'group'), section='TFR')
    return report

def plot_group_peak_alpha(grand_avg_both, grand_avg_right, grand_avg_left, occipital, cond_label, report):
    """
    Compute and plot the peak alpha frequency (PAF) from grand average TFRs.
    I decided to use the grand average to calculate PAF instead of the average of
    right and left attention.
    
    Parameters
    ----------
    grand_avg_right, grand_avg_left : mne.time_frequency.AverageTFR
        Grand averages for cue right and left.
    occipital : list
        Channels to include.
    cond_label : str
        Condition label.
    report : mne.Report
        MNE Report to update.
        
    Returns
    -------
    paf_range : ndarray
        Array of frequencies representing the PAF range.
    report : mne.Report
    """
    # Crop to alpha band (8-14 Hz) in post-stimulus window (0.3 to 0.8 s)
    """right and left aren't being used now"""
    tfr_right_crop = grand_avg_right.copy().crop(tmin=0.3, tmax=0.8, fmin=8, fmax=14).pick(occipital)
    tfr_left_crop = grand_avg_left.copy().crop(tmin=0.3, tmax=0.8, fmin=8, fmax=14).pick(occipital)
    tfr_both_crop = grand_avg_both.copy().crop(tmin=0.3, tmax=0.8, fmin=8, fmax=14).pick(occipital)

    # Find peak frequency index by averaging power over channels and time
    peak_idx_right = np.argmax(np.mean(np.abs(tfr_right_crop.data), axis=(0, 2)))
    peak_idx_left  = np.argmax(np.mean(np.abs(tfr_left_crop.data), axis=(0, 2)))
    peak_idx_both  = np.argmax(np.mean(np.abs(tfr_both_crop.data), axis=(0, 2)))

    # Get the corresponding frequencies
    peak_freq_not_both = np.mean([tfr_right_crop.freqs[peak_idx_right], tfr_left_crop.freqs[peak_idx_left]])
    peak_freq = tfr_both_crop.freqs[peak_idx_both]
    paf_range = np.arange(peak_freq - 2, peak_freq + 3)  # ±2 Hz around the peak
    
    # Plot group PSD (using both grand average on occipital channels)
    grand_avg_both_occ = grand_avg_both.copy().pick(occipital)
    avg_power = np.mean(grand_avg_both_occ.data, axis=(0, 2))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(grand_avg_both_occ.freqs, avg_power, color='black')
    ymin, ymax = ax.get_ylim()
    ax.axvline(x=paf_range[0], color='gray', linestyle='--', linewidth=2)
    ax.axvline(x=paf_range[-1], color='gray', linestyle='--', linewidth=2)
    ax.fill_betweenx([ymin, ymax], paf_range[0], paf_range[-1],
                     color='lightgray', alpha=0.5)

    plt.title(f'{cond_label} Group Peak Alpha Frequency ({peak_freq:.2f} Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (T/m)^2/Hz')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

    fig.tight_layout()
    plt.show()

    report.add_figure(fig=fig, title=f'{cond_label} Group Peak Alpha Frequency',
                      caption='PSD with peak alpha frequency range highlighted',
                      tags=('PAF', 'group'), section='PAF')
    
    return paf_range, report

def compute_and_plot_stim_effects_all(grand_avg_stim_dict, grand_avg_no_stim_dict, report):
    """
    Compute and plot stimulation effects for each cue type ('both', 'right', 'left').

    For each cue type:
      1. Baseline-corrected difference:
           (grand_avg_stim - grand_avg_no_stim)
         Both TFRs are first baseline corrected using BASELINE and mode 'percent'.

      2. Ratio effect (uncorrected):
           (grand_avg_no_stim - grand_avg_stim) / (grand_avg_no_stim + grand_avg_stim)
         This calculation is done on the uncorrected TFR data.

    Both results are plotted using plot_topo and added to the MNE Report.

    Parameters
    ----------
    grand_avg_stim_dict : dict
         Dictionary with keys 'both', 'right', and 'left' for the stimulation condition.
    grand_avg_no_stim_dict : dict
         Dictionary with keys 'both', 'right', and 'left' for the no stimulation condition.
    report : mne.Report
         MNE Report to which the figures will be added.

    Returns
    -------
    report : mne.Report
         Updated MNE Report with the stimulation effect figures added.
    """
    cues = ['both', 'right', 'left']
    eps = 0  # To avoid division by zero
    for cue in cues:
        # 1. Baseline-corrected difference:
        stim_bc = grand_avg_stim_dict[cue].copy()
        stim_bc.apply_baseline(baseline=BASELINE, mode='percent')
        no_stim_bc = grand_avg_no_stim_dict[cue].copy()
        no_stim_bc.apply_baseline(baseline=BASELINE, mode='percent')
        
        diff_bc = grand_avg_stim_dict[cue].copy()
        diff_bc.data = no_stim_bc.data - stim_bc.data
        
        fig_diff = diff_bc.plot_topo(tmin=-0.5, tmax=1.5,
                                      title=f'Baseline Corrected Difference ({cue}): no_stim - stim',
                                      show=False,
                                      fig_facecolor='w', font_color='k')
        report.add_figure(fig=fig_diff,
                          title=f'Baseline Corrected Stim Effect ({cue})',
                          caption=f'Baseline corrected difference for cue {cue}: (no_stim - stim)',
                          tags=('stim_effect', 'group'),
                          section='TFR')
        
        # 2. Ratio effect (uncorrected):
        ratio_effect = grand_avg_no_stim_dict[cue].copy()
        numerator = grand_avg_no_stim_dict[cue].data - grand_avg_stim_dict[cue].data
        denominator = grand_avg_no_stim_dict[cue].data + grand_avg_stim_dict[cue].data
        ratio = numerator / (denominator + eps)
        ratio_effect.data = ratio
        
        fig_ratio = ratio_effect.plot_topo(tmin=-0.5, tmax=1.5,
                                            title=f'Stim Effect Ratio ({cue}): (no_stim - stim)/(no_stim + stim)',
                                            show=False,
                                            fig_facecolor='w', font_color='k')
        report.add_figure(fig=fig_ratio,
                          title=f'Stim Effect Ratio ({cue})',
                          caption=f'Ratio effect for cue {cue} computed as (no_stim - stim)/(no_stim + stim) using uncorrected data',
                          tags=('stim_effect', 'group'),
                          section='TFR')
    return report

def compute_and_plot_stim_effects_occipital(grand_avg_stim_dict, grand_avg_no_stim_dict, report):
    """
    Compute and plot stimulation effects for each cue type ('both', 'right', 'left') 
    on occipital sensors (PO3, PO4, POz).

    For each cue type:
      1. Baseline-corrected difference:
           (grand_avg_no_stim - grand_avg_stim)
         Both TFRs are first baseline corrected using BASELINE and mode 'percent'.

      2. Ratio effect (uncorrected):
           (grand_avg_no_stim - grand_avg_stim) / (grand_avg_no_stim + grand_avg_stim)

    Plots are generated for selected occipital sensors and added to the MNE Report.

    Parameters
    ----------
    grand_avg_stim_dict : dict
         Dictionary with keys 'both', 'right', and 'left' for the stimulation condition.
    grand_avg_no_stim_dict : dict
         Dictionary with keys 'both', 'right', and 'left' for the no stimulation condition.
    report : mne.Report
         MNE Report to which the figures will be added.

    Returns
    -------
    report : mne.Report
         Updated MNE Report with the stimulation effect figures added.
    """
    occipital = ['PO3', 'PO4', 'POz']
    cues = ['both', 'right', 'left']
    eps = 1e-10  # Small constant to avoid division by zero

    for cue in cues:
        fig_diff, axes_diff = plt.subplots(len(occipital), 1, figsize=(10, 15))
        fig_ratio, axes_ratio = plt.subplots(len(occipital), 1, figsize=(10, 15))
        
        # 1. Baseline-corrected difference:
        stim_bc = grand_avg_stim_dict[cue].copy()
        stim_bc.apply_baseline(baseline=BASELINE, mode='percent')
        no_stim_bc = grand_avg_no_stim_dict[cue].copy()
        no_stim_bc.apply_baseline(baseline=BASELINE, mode='percent')

        diff_bc = grand_avg_stim_dict[cue].copy()
        diff_bc.data = no_stim_bc.data - stim_bc.data

        # 2. Ratio effect (uncorrected):
        ratio_effect = grand_avg_no_stim_dict[cue].copy()
        numerator = grand_avg_no_stim_dict[cue].data - grand_avg_stim_dict[cue].data
        denominator = grand_avg_no_stim_dict[cue].data + grand_avg_stim_dict[cue].data
        ratio = numerator / (denominator + eps)
        ratio_effect.data = ratio

        # Plot for each occipital channel
        for i, ch in enumerate(occipital):
            diff_bc.plot(picks=ch, baseline=None, tmin=-0.5, tmax=1.5, vmin=-0.75, vmax=0.75,
                         axes=axes_diff[i], show=False)
            axes_diff[i].set_title(f'{cue} - Baseline Corrected Diff (no_stim - stim) - {ch}')

            ratio_effect.plot(picks=ch, baseline=None, tmin=-0.5, tmax=1.5, vmin=-0.75, vmax=0.75,
                              axes=axes_ratio[i], show=False)
            axes_ratio[i].set_title(f'{cue} - Ratio Effect (no_stim - stim)/(no_stim + stim) - {ch}')
        
        fig_diff.tight_layout()
        plt.show(fig_diff)
        fig_ratio.tight_layout()
        plt.show(fig_ratio)

        report.add_figure(fig=fig_diff,
                          title=f'Baseline Corrected Stim Effect ({cue}) - Occipital',
                          caption=f'Baseline corrected difference for cue {cue} on occipital sensors',
                          tags=('stim_effect', 'group'),
                          section='TFR')

        report.add_figure(fig=fig_ratio,
                          title=f'Stim Effect Ratio ({cue}) - Occipital',
                          caption=f'Ratio effect for cue {cue} computed as (no_stim - stim)/(no_stim + stim) on occipital sensors',
                          tags=('stim_effect', 'group'),
                          section='TFR')

    return report

def plot_group_MI(paf_range, grand_avg_right, grand_avg_left, occipital, cond_label, report):
    """
    Crop the group TFRs to the PAF range and compute the modulation index (MI) over time.
    MI is defined as (right - left) / (right + left), averaged over the specified occipital channels.
    The function plots both the average MI time series and a shaded area representing ±1 standard
    deviation across channels and frequencies. The plot is then added to the MNE Report.
    
    Parameters
    ----------
    paf_range : ndarray
        Frequency range corresponding to the peak alpha region (e.g. np.arange(paf-2, paf+3)).
    grand_avg_right : mne.time_frequency.AverageTFR
        Group grand average TFR for cue right.
    grand_avg_left : mne.time_frequency.AverageTFR
        Group grand average TFR for cue left.
    occipital : list of str
        List of channel names to include.
    cond_label : str
        Condition label (e.g., 'stim' or 'no_stim').
    report : mne.Report
        MNE Report object to which the figure is added.
        
    Returns
    -------
    mi_ts : ndarray
        The average MI time series.
    report : mne.Report
        The updated MNE Report with the MI plot added.
    """
    # Crop the TFRs to the peak alpha frequency range and to the post-stimulus time window (0.3-0.8 s)
    right_crop = grand_avg_right.copy().crop(fmin=paf_range[0], fmax=paf_range[-1], tmin=0.3, tmax=0.8).pick(occipital)
    left_crop  = grand_avg_left.copy().crop(fmin=paf_range[0], fmax=paf_range[-1], tmin=0.3, tmax=0.8).pick(occipital)
    
    # Compute the MI for each time point:
    # MI = (right - left) / (right + left)
    MI = (right_crop.data - left_crop.data) / (right_crop.data + left_crop.data)
    
    # Compute average MI and standard deviation across channels and frequencies (axis=(0, 1))
    mi_ts = np.mean(MI, axis=(0, 1))
    mi_std = np.std(MI, axis=(0, 1))
    
    # Create the MI plot with fill_between for ± STD
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(right_crop.times, mi_ts, color='red', label='Average MI')
    # ax.fill_between(right_crop.times,
    #                 mi_ts - mi_std,
    #                 mi_ts + mi_std,
    #                 color='red', alpha=0.3, label='Standard Deviation')  # std is too high and makes the plot look weird
    ax.set_title(f'{cond_label} Group MI over Time (PAF)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MI')

    # Optionally, adjust y-limits based on data range:
    # y_min = mi_ts.min() - 0.1 * mi_ts.min()
    # y_max = mi_ts.max() + 0.1 * mi_ts.max()
    # ax.set_ylim(y_min, y_max)
    ax.legend()
    plt.tight_layout()
    
    # Add the figure to the MNE Report
    report.add_figure(fig=fig, title=f'{cond_label} Group MI over Time',
                      caption='Modulation Index (MI) over time computed from group TFRs in the peak alpha range, with shaded STD.',
                      tags=('MI', 'group'), section='MI')
    plt.show()
    return mi_ts, report

# ---------------------------
# Main Script
# ---------------------------

# Settings (customize these paths and lists for your study)
subject_list = ['102', '107', '110', '112', '103', '104', '105']  
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
extension = '.fif'
stim_suffix = 'stim'
no_stim_suffix = 'no-stim'
input_suffix = 'epo'
deriv_suffix = 'tfr'

platform = 'mac'  # 'bluebear', 'mac', or 'windows'
rds_dir = '/Volumes/jenseno-avtemporal-attention' if platform == 'mac' else '/rds/projects/j/jenseno-avtemporal-attention'

bids_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD', 'data', 'BIDS')
report_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD', 'derivatives', 'reports')

# # for bear outage:
# bids_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/data/BIDS'
# report_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/derivatives/reports' # only for bear outage time

deriv_folder_group = op.join(bids_root, 'derivatives', 'group')
group_base = 'sub-group_ses-01_task-SpAtt_run-01_eeg'

report_folder = op.join(report_root, 'group')
report_fname = op.join(report_folder, 'group_report-130325.hdf5')
html_report_fname = op.join(report_folder, 'group_report-130325.html')

# Create a report
report = mne.Report(title='Group TFR and PAF Report')

# Initialize dictionary for subject TFRs (by condition and cue type)
group_tfrs = {'stim': {'both': [], 'right': [], 'left': []},
              'no_stim': {'both': [], 'right': [], 'left': []}}

# Process each subject and each stimulation condition
for subj in subject_list:
    # Define subject derivative folder and BIDSPath
    subj_folder = op.join(bids_root, 'derivatives', f"sub-{subj}")
    bids_path = BIDSPath(subject=subj, session=session, task=task, run=run,
                         root=bids_root, datatype='eeg', suffix=eeg_suffix)
    for epoch_type in ['cue']:  # Extend list if needed
        subj_input_suffix = f"epo-{epoch_type}"
        for stim in [True, False]:
            print(f"Processing subject {subj}, epoch: {epoch_type}, stim: {stim}")
            _, tfrs, report = process_subject_tfr(stim, subj_folder, bids_path,
                                                  subj_input_suffix, deriv_suffix, extension, subj, report)
            condition = 'stim' if stim else 'no_stim'
            for key in ['both', 'right', 'left']:
                group_tfrs[condition][key].append(tfrs[key])

# Compute group grand averages and save
group_avg = {}
for condition in ['stim', 'no_stim']:
    group_avg[condition], report = compute_group_grand_avg(group_tfrs[condition],
                                                   deriv_folder_group, group_base,
                                                   condition, deriv_suffix, extension, report)

# For each condition, generate group-level plots (sensor topography, PAF, and MI)
for condition in ['stim', 'no_stim']:
    label = condition
    avg_both = group_avg[condition]['both']
    avg_right = group_avg[condition]['right']
    avg_left  = group_avg[condition]['left']
    
    report = plot_group_representative_channel(avg_right, avg_left, label, report)
    if condition == 'no_stim':
        report= plot_group_sensor_stim_nostim(group_avg, report)
    occipital_ch = ['PO4', 'POz', 'PO3']
    paf_range, report = plot_group_peak_alpha(avg_both, avg_right, avg_left, occipital_ch, label, report)
    mi_ts, report = plot_group_MI(paf_range, avg_right, avg_left, occipital_ch, label, report)

# Compute stimulation effects for all cues using the group grand averages:
report = compute_and_plot_stim_effects_all(group_avg['stim'], group_avg['no_stim'], report)
report = compute_and_plot_stim_effects_occipital(group_avg['stim'], group_avg['no_stim'], report)

# Save the final report in both HDF5 and HTML formats
report.save(report_fname, overwrite=True)
report.save(html_report_fname, overwrite=True, open_browser=True)
