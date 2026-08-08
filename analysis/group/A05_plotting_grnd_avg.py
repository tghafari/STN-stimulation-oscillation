#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grand Average Analysis Script for STN Stimulation Oscillation Study
====================================================================

This script performs the following analyses:
    1. Reads the grand average evoked responses for stimulation ('stim')
       and no stimulation ('no_stim') conditions.
    2. Plots the evoked responses for both conditions on occipital sensors
       (PO3, PO4, POz).
    3. Reads the grand average time–frequency representations (TFRs)
       for the 'both' cue condition for both stimulation conditions.
    4. Computes and plots:
         a. The baseline-corrected difference (no_stim - stim)
         b. The ratio effect computed as (no_stim - stim) / (no_stim + stim)
       on occipital channels (PO3, PO4, POz).
    5. Reads and plots the grand average modulation index (MI) for both conditions
       on the same channels.
    6. Displays each plot so you can review them before further processing.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Global Parameters and Directories
# ---------------------------

# Occipital channels to analyze
OCCIPITAL_CHANNELS = ['PO3', 'PO4', 'POz']

# Baseline period for baseline correction (adjust as needed)
BASELINE = (-0.5, 0)

# Time and frequency parameters for TFR plots
TMIN, TMAX = -0.5, 1.5      # Time window for plots (in seconds)
FMIN, FMAX = 1, 40          # Frequency range (in Hz)
EPS = 1e-10                 # Small constant to avoid division by zero

# Update these directory paths with your actual data locations
EVOKED_DIR = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/derivatives/group/evoked'
TFR_DIR = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/derivatives/group/tfr'
MI_DIR = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/derivatives/group/mi'

# ---------------------------
# Data Loading Helper Functions
# ---------------------------
def load_evoked_data(condition, stim_status):
    """
    Load the grand average evoked response from a FIF file.
    
    Parameters
    ----------
    condition : str
        Name of the condition (e.g., 'sub-group_ses-01_task-SpAtt_run-01_eeg').
    stim_status : str
        Stimulation status, either 'stim' or 'no_stim'.
    
    Returns
    -------
    evoked : instance of mne.Evoked
        The loaded evoked response.
    """
    filename = f'grand_avg_{condition}_{stim_status}-ave.fif'
    filepath = os.path.join(EVOKED_DIR, filename)
    evoked = mne.read_evokeds(filepath, condition=0)
    return evoked

def load_tfr_data(cue, stim_status):
    """
    Load the grand average time-frequency representation (TFR) from an HDF5 file.
    
    Parameters
    ----------
    cue : str
        Cue type (e.g., 'both').
    stim_status : str
        Stimulation status, either 'stim' or 'no_stim'.
    
    Returns
    -------
    tfr : instance of mne.time_frequency.AverageTFR
        The loaded TFR.
    """
    filename = f'grand_avg_tfr_{cue}_{stim_status}-tfr.h5'
    filepath = os.path.join(TFR_DIR, filename)
    tfr = mne.time_frequency.read_tfrs(filepath)[0]
    return tfr

def load_mi_data(stim_status):
    """
    Load the grand average modulation index (MI) TFR from an HDF5 file.
    
    Parameters
    ----------
    stim_status : str
        Stimulation status, either 'stim' or 'no_stim'.
    
    Returns
    -------
    mi : instance of mne.time_frequency.AverageTFR
        The loaded MI data.
    """
    filename = f'grand_avg_mi_{stim_status}-tfr.h5'
    filepath = os.path.join(MI_DIR, filename)
    mi = mne.time_frequency.read_tfrs(filepath)[0]
    return mi

# ---------------------------
# Plotting Functions
# ---------------------------
def plot_evoked_responses(evoked_stim, evoked_no_stim, channels):
    """
    Plot grand average evoked responses for stim and no_stim conditions on specified channels.
    
    Parameters
    ----------
    evoked_stim : instance of mne.Evoked
        Evoked data for the stimulation condition.
    evoked_no_stim : instance of mne.Evoked
        Evoked data for the no stimulation condition.
    channels : list of str
        List of channel names to plot.
    
    Displays
    --------
    A separate plot for each channel showing both stim and no_stim responses.
    """
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 15))
    for i, ch in enumerate(channels):
        # Plot the stim evoked response
        evoked_stim.plot(picks=ch, axes=axes[i], show=False, titles=f'Stim - {ch}', time_unit='s')
        # Overplot the no_stim response on the same axis
        evoked_no_stim.plot(picks=ch, axes=axes[i], show=False, titles=f'No Stim - {ch}', time_unit='s')
        axes[i].legend(['Stim', 'No Stim'])
    plt.tight_layout()
    plt.show()

def plot_tfr_differences_and_ratios(tfr_stim, tfr_no_stim, channels):
    """
    Compute and plot the TFR differences and ratio effects for stim vs. no_stim conditions.
    
    Both TFRs are first baseline corrected using the provided BASELINE in 'percent' mode.
    
    For each channel:
        - The baseline-corrected difference is computed as (no_stim - stim).
        - The ratio effect is computed as (no_stim - stim) / (no_stim + stim).
    
    Parameters
    ----------
    tfr_stim : instance of mne.time_frequency.AverageTFR
        TFR data for the stimulation condition.
    tfr_no_stim : instance of mne.time_frequency.AverageTFR
        TFR data for the no stimulation condition.
    channels : list of str
        List of channel names to plot.
    
    Displays
    --------
    Two plots per channel: one for the difference and one for the ratio effect.
    """
    # Apply baseline correction (in percent) to both TFRs
    tfr_stim.apply_baseline(baseline=BASELINE, mode='percent')
    tfr_no_stim.apply_baseline(baseline=BASELINE, mode='percent')

    # Compute difference and ratio for the TFR data
    diff_data = tfr_no_stim.data - tfr_stim.data
    ratio_data = (tfr_no_stim.data - tfr_stim.data) / (tfr_no_stim.data + tfr_stim.data + EPS)

    # Create new TFR objects for difference and ratio
    tfr_diff = tfr_stim.copy()
    tfr_diff.data = diff_data
    tfr_ratio = tfr_stim.copy()
    tfr_ratio.data = ratio_data

    # Plot the results for each channel
    for ch in channels:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        tfr_diff.plot(picks=ch, baseline=None, mode='mean', tmin=TMIN, tmax=TMAX,
                      fmin=FMIN, fmax=FMAX, axes=axes[0], show=False,
                      title=f'Baseline Corrected Difference ({ch})')
        tfr_ratio.plot(picks=ch, baseline=None, mode='mean', tmin=TMIN, tmax=TMAX,
                       fmin=FMIN, fmax=FMAX, axes=axes[1], show=False,
                       title=f'Ratio Effect ({ch})')
        plt.tight_layout()
        plt.show()

def plot_modulation_index(mi_stim, mi_no_stim, channels):
    """
    Plot the modulation index (MI) for stim and no_stim conditions on the same plot for each channel.
    
    Parameters
    ----------
    mi_stim : instance of mne.time_frequency.AverageTFR
        MI data for the stimulation condition.
    mi_no_stim : instance of mne.time_frequency.AverageTFR
        MI data for the no stimulation condition.
    channels : list of str
        List of channel names to plot.
    
    Displays
    --------
    A combined plot per channel showing MI for both stim and no_stim.
    """
    for ch in channels:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # Plot MI for stim condition
        mi_stim.plot(picks=ch, baseline=None, mode='mean', tmin=TMIN, tmax=TMAX,
                     fmin=FMIN, fmax=FMAX, axes=ax, show=False,
                     title=f'Modulation Index (MI) - {ch}')
        # Overplot MI for no_stim condition on the same axis
        mi_no_stim.plot(picks=ch, baseline=None, mode='mean', tmin=TMIN, tmax=TMAX,
                        fmin=FMIN, fmax=FMAX, axes=ax, show=False)
        ax.legend(['Stim', 'No Stim'])
        plt.tight_layout()
        plt.show()

# ---------------------------
# Main Script Execution
# ---------------------------
if __name__ == '__main__':
    # --- EVOKED RESPONSES ---
    # Update 'condition_name' as needed (e.g., group base name without suffix)
    condition_name = 'sub-group_ses-01_task-SpAtt_run-01_eeg'
    
    # Load grand average evoked responses for stim and no_stim
    evoked_stim = load_evoked_data(condition_name, 'stim')
    evoked_no_stim = load_evoked_data(condition_name, 'no_stim')
    
    # Plot evoked responses on occipital channels
    print("Displaying grand average evoked responses (stim vs. no_stim) on occipital channels...")
    plot_evoked_responses(evoked_stim, evoked_no_stim, OCCIPITAL_CHANNELS)
    
    # --- TFR ANALYSIS ---
    # Load grand average TFRs for the 'both' cue condition for stim and no_stim
    tfr_stim = load_tfr_data('both', 'stim')
    tfr_no_stim = load_tfr_data('both', 'no_stim')
    
    # Plot TFR baseline-corrected difference and ratio effect on occipital channels
    print("Displaying TFR differences and ratio effects (stim vs. no_stim) on occipital channels...")
    plot_tfr_differences_and_ratios(tfr_stim, tfr_no_stim, OCCIPITAL_CHANNELS)
    
    # --- MODULATION INDEX (MI) ANALYSIS ---
    # Load grand average MI for stim and no_stim
    mi_stim = load_mi_data('stim')
    mi_no_stim = load_mi_data('no_stim')
    
    # Plot MI for stim vs. no_stim on occipital channels in a single plot per channel
    print("Displaying modulation index (MI) for stim vs. no_stim on occipital channels...")
    plot_modulation_index(mi_stim, mi_no_stim, OCCIPITAL_CHANNELS)
