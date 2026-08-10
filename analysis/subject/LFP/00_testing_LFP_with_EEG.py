"""
========================================
00_teting_lfp_with_eeg

This code compares lfp timings with eeg and
attempts at adding the lfp channel to eeg.
(not yet sure if possible)

the main reason is sampling frequncy of lfp
doesn't seem to be correct.

For starter we are focusing on subject 102.

written by Tara Ghafari
tara.ghafari@gmail.com
=======================================
"""

# Import relevant Python modules
import os.path as op
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from copy import deepcopy

import mne
from mne_bids import BIDSPath, print_dir_tree, write_raw_bids, read_raw_bids
from mne_bids.stats import count_events


# Stimulation sequence
"""copy the stim sequence for each participant from here: 
https://github.com/tghafari/STN-stimulation-oscillation/wiki/Stimulation-table"""
stim_sequence = {'sub-102': ["no_stim-left rec", "no_stim-right rec", "Left stim- no rec", "Right stim- no rec"],
                 } 
# BIDS LFP settings
subject = '102'
session = '01'
task = 'SpAtt'
run = '01'

######################## LFP #########################
modality_lfp = 'lfp'
side = 'left'
events_suffix = 'events'  

platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
pilot = False  # is it pilot data or real data?
correct_sfreq = False  # do you want to correct sfreq?
sanity_test = False
eve_rprt = True
summary_rprt = True

if platform == 'bluebear':
    rds_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/jenseno-avtemporal-attention'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
# '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD'  # only for bear outage time
data_root = op.join(project_root, 'data/data-organised')
# '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/data/data-organised'  # only for bear outage time
bids_root = op.join(project_root, 'data', 'BIDS')
base_fpath = op.join(data_root, f'sub-{subject}', f'ses-{session}')

base_fname_lfp = f'sub-{subject}_ses-{session}_task-{task}_run-{run}_{modality_lfp}_{side}'
lfp_fname = op.join(base_fpath, f'{modality_lfp}', base_fname_lfp + '.edf') 
events_fname_lfp = op.join(base_fpath, f'{modality_lfp}', base_fname_lfp + '-eve.fif')
annotated_raw_fname_lfp = op.join(base_fpath, f'{modality_lfp}', base_fname_lfp + '_ann.fif')

# Read raw file in edf format
raw_lfp = mne.io.read_raw_edf(lfp_fname, preload=True)
# Crop lfp to remove out-of-task data
cropped_lfp = raw_lfp.copy().crop(tmin=280, tmax=673)

events_messed_up, _ = mne.events_from_annotations(cropped_lfp, event_id='auto')

mapping_102 = {1:'cue_onset_right',
            2:'cue_onset_left',
            3:'trial_onset',           
           4:'stim_onset',
           5:'catch_onset',
            6:'dot_onset_right',
           7:'dot_onset_left',
           8:'response_press_onset',
           14:'block_onset',  
           15:'block_end',
        } 

annotations_from_events_lfp = mne.annotations_from_events(events=events_messed_up,
                                                    event_desc=mapping_102,
                                                    sfreq=cropped_lfp.info["sfreq"],
                                                    orig_time=cropped_lfp.info["meas_date"],
                                                    )
cropped_lfp.set_annotations(annotations_from_events_lfp)

event_dict_lfp = {'cue_onset_right':1,
           'cue_onset_left':2,
           'trial_onset':3,
           'stim_onset':4,
           'catch_onset':5,
           'dot_onset_right':6,
           'dot_onset_left':7,
           'response_press_onset':8,
           'block_onset':20,
           'block_end':21,
            }
events_lfp, events_id_lfp = mne.events_from_annotations(cropped_lfp, event_id=event_dict_lfp)

######################## EEG #########################
modality_eeg = 'eeg'
extension = '.fif'
brainVision_basename = f'{subject[1:]}_ao'  # subject[-2:] might need modification per subject

base_fname_eeg = f'sub-{subject}_ses-{session}_task-{task}_run-{run}_{modality_eeg}'
eeg_fname = op.join(base_fpath, f'{modality_eeg}', brainVision_basename + '.eeg')  
vhdr_fname = op.join(base_fpath, f'{modality_eeg}', brainVision_basename + '.vhdr')
events_fname_eeg = op.join(base_fpath, f'{modality_eeg}', base_fname_eeg + '-eve.fif')
annotated_raw_fname_eeg = op.join(base_fpath, f'{modality_eeg}', base_fname_eeg + extension)
beh_fig_fname = op.join(project_root, 'derivatives/figures', f'sub-{subject}-beh-performance.png')  # where you save the matlab output of behavioural performance plots

# Read raw file in BrainVision (.vhdr, .vmrk, .eeg) format
raw_eeg = mne.io.read_raw_brainvision(vhdr_fname, eog=('HEOGL', 'HEOGR', 'VEOGb'), preload=True)
# Crop eeg to focus only on the blocks that correspond to lfp
cropped_eeg = raw_eeg.copy().crop(tmin=206, tmax=599)

cropped_eeg.rename_channels({'T8':'vEOG1', 
                     'FT10':'vEOG2',
                     'T7':'hEOG1',
                     'FT9':'hEOG2'}
                     )
# Set both vEOG and hEOG as EOG channels
cropped_eeg.set_channel_types({'vEOG1':'eog', 
                       'vEOG2':'eog',
                       'hEOG1':'eog', 
                       'hEOG2':'eog'}
                       )  
# Remove channels on mastoid
cropped_eeg.info["bads"].extend(['TP9','TP10'])  

# Read events from raw object
events_eeg, _ = mne.events_from_annotations(cropped_eeg, event_id='auto')
# Create Annotation object with correct labels
"""list of triggers https://github.com/tghafari/STN-stimulation-oscillation/blob/main/Instructions/triggers.md"""
mapping_eeg = {1:'cue_onset_right',
           2:'cue_onset_left',
           3:'trial_onset',
           4:'stim_onset',
           5:'catch_onset',
           6:'dot_onset_right',
           7:'dot_onset_left',
           8:'response_press_onset',
           20:'block_onset',
           21:'block_end',
           30:'experiment_end',  
           99999:'new_stim_segment',
        }
annotations_from_events_eeg = mne.annotations_from_events(events=events_eeg,
                                                    event_desc=mapping_eeg,
                                                    sfreq=cropped_eeg.info["sfreq"],
                                                    orig_time=cropped_eeg.info["meas_date"],
                                                    )
cropped_eeg.set_annotations(annotations_from_events_eeg)

# Have to explicitly assign values to events for brainvision data
event_dict_eeg = {'cue_onset_right':1,
           'cue_onset_left':2,
           'trial_onset':3,
           'stim_onset':4,
           'catch_onset':5,
           'dot_onset_right':6,
           'dot_onset_left':7,
           'response_press_onset':8,
           'block_onset':20,
           'block_end':21,
           'experiment_end':30,  #sub02 does not have this
           'abort':31,  # participant 04_wmf has abort
           'new_stim_segment_maybe':255,  # sub102 has an extra trigger
           'new_stim_segment':99999, 
        }
_, events_id_eeg = mne.events_from_annotations(cropped_eeg, event_id=event_dict_eeg)

######################################### Comparison ################################
"""Now the housekeeping is done for both lfp and eeg, let's
compare them."""

fig_lfp = mne.viz.plot_events(events_lfp, sfreq=cropped_lfp.info["sfreq"], 
                          first_samp=cropped_lfp.first_samp, 
                          event_id=events_id_lfp)
fig_eeg = mne.viz.plot_events(events_eeg, 
                          sfreq=cropped_eeg.info["sfreq"], 
                          first_samp=cropped_eeg.first_samp, 
                          event_id=events_id_eeg)

# Extract time (in seconds) from sample indices
time_lfp = (events_lfp[:, 0] - cropped_lfp.first_samp) / cropped_lfp.info["sfreq"]
time_eeg = (events_eeg[:, 0] - cropped_eeg.first_samp) / cropped_eeg.info["sfreq"]

# Extract event codes
event_codes_lfp = events_lfp[:, 2]
event_codes_eeg = events_eeg[:, 2]

fig, ax = plt.subplots(figsize=(12, 4))
# Plot LFP events (e.g., in red)
ax.plot(time_lfp, event_codes_lfp, 'o', label='LFP Events', color='red')

# Plot EEG events (e.g., in blue)
ax.plot(time_eeg, event_codes_eeg, 'x', label='EEG Events', color='blue')

# Labels and title
ax.set_xlabel('Time (s)')
ax.set_ylabel('Event ID')
ax.set_title('LFP and EEG Events Over Time')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.legend()

plt.tight_layout()
plt.show()


#============================== checking lfp and eeg without mne python ===================#
from scipy.signal import welch, spectrogram

fig, axes = plt.subplots(2, 3, figsize=(18, 8))  # 2 rows, 3 columns

# EEG Time Domain
axes[0, 0].plot(cropped_eeg.times[:2000], cropped_eeg.get_data()[47][:2000])
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_title("EEG - Time Domain")
axes[0, 0].grid(True, which='both')

# EEG Power Spectrum
f_eeg, Pxx_eeg = welch(cropped_eeg.get_data()[47], cropped_eeg.info["sfreq"], nperseg=2*int(cropped_eeg.info["sfreq"]))
axes[0, 1].semilogy(f_eeg, Pxx_eeg)
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].set_ylabel("Power Spectral Density")
axes[0, 1].set_title("EEG - Power Spectrum")
axes[0, 1].set_xlim(0, 200)

# EEG Spectrogram
f_eeg_s, t_eeg, Sxx_eeg = spectrogram(cropped_eeg.get_data()[47], cropped_eeg.info["sfreq"], nperseg=int(cropped_eeg.info["sfreq"]))
im0 = axes[0, 2].pcolormesh(t_eeg, f_eeg_s, np.log(Sxx_eeg), shading='gouraud')
axes[0, 2].set_xlabel("Time (s)")
axes[0, 2].set_ylabel("Frequency (Hz)")
axes[0, 2].set_title("EEG - Spectrogram")
axes[0, 2].set_ylim(0, 100)
fig.colorbar(im0, ax=axes[0, 2], label="Power (log)")

# LFP Time Domain
axes[1, 0].plot(cropped_lfp.times[:2000], cropped_lfp.get_data()[0][:2000])
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Amplitude")
axes[1, 0].set_title("LFP - Time Domain")
axes[1, 0].grid(True, which='both')

# LFP Power Spectrum
f_lfp, Pxx_lfp = welch(cropped_lfp.get_data()[0], cropped_lfp.info["sfreq"], nperseg=2*int(cropped_lfp.info["sfreq"]))
axes[1, 1].semilogy(f_lfp, Pxx_lfp)
axes[1, 1].set_xlabel("Frequency (Hz)")
axes[1, 1].set_ylabel("Power Spectral Density")
axes[1, 1].set_title("LFP - Power Spectrum")
axes[1, 1].set_xlim(0, 200)

# LFP Spectrogram
f_lfp_s, t_lfp, Sxx_lfp = spectrogram(cropped_lfp.get_data()[0], cropped_lfp.info["sfreq"], nperseg=int(cropped_lfp.info["sfreq"]))
im1 = axes[1, 2].pcolormesh(t_lfp, f_lfp_s, np.log(Sxx_lfp), shading='gouraud')
axes[1, 2].set_xlabel("Time (s)")
axes[1, 2].set_ylabel("Frequency (Hz)")
axes[1, 2].set_title("LFP - Spectrogram")
axes[1, 2].set_ylim(0, 100)
fig.colorbar(im1, ax=axes[1, 2], label="Power (log)")

plt.tight_layout()
plt.show()

#============================================================================#
