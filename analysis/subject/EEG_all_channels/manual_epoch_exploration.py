"""Manual work on segmented/epoched EEG -- deliberately simple copy/paste code.

This file does NOT use pipeline_config, helper functions, argparse, or the automated
pipeline. Change SUBJECT and REFERENCE below, then run individual sections in
IPython/VS Code. The examples use direct MNE commands so every analysis step is
visible. Nothing below overwrites the saved EEG files.
"""

# %% 1. Imports and explicit directory
import mne
import numpy as np
from pathlib import Path

SUBJECT = "103"
REFERENCE = "noref"          # use "noref" or "avgref"

# This is where participant derivative FIF files are stored on Tara's Mac.
data_dir = Path(
    "/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/"
    "STN-in-PD/data/BIDS/derivatives"
) / f"sub-{SUBJECT}"

print(data_dir)

# Optional: list the epoch files so you can see exactly what is available.
for file in sorted(data_dir.glob("*epo.fif")):
    print(file.name)


# %% 2. Open the segmented, cue-locked epochs made by P01
# These are BEFORE ICA and before manual bad-trial rejection.
# REFERENCE tells you explicitly whether P01 applied average rereferencing.
stim_file = data_dir / (
    f"sub-{SUBJECT}_ses-01_task-SpAtt_run-01_eeg.fif_"
    f"stim_desc-{REFERENCE}_epo.fif"
)

nostim_file = data_dir / (
    f"sub-{SUBJECT}_ses-01_task-SpAtt_run-01_eeg.fif_"
    f"no-stim_desc-{REFERENCE}_epo.fif"
)

stim = mne.read_epochs(stim_file, preload=True)
nostim = mne.read_epochs(nostim_file, preload=True)

print(stim)
print(nostim)
print("Stim bad channels:", stim.info["bads"])
print("No-stim bad channels:", nostim.info["bads"])


# %% 3. Look at the epoched EEG time series
# Scroll through trials and channels interactively.
stim.plot()

# Or inspect no-stimulation epochs.
nostim.plot()


# %% 4. Welch PSD -- stimulation epochs
# Same basic Welch settings used in the pipeline final epoch PSD.
n_fft = min(int(2 * stim.info["sfreq"]), len(stim.times))

psd_stim = stim.compute_psd(
    method="welch",
    fmin=0.1,
    fmax=100,
    n_fft=n_fft,
)

psd_stim.plot()


# %% 5. Welch PSD -- no-stimulation epochs
n_fft = min(int(2 * nostim.info["sfreq"]), len(nostim.times))

psd_nostim = nostim.compute_psd(
    method="welch",
    fmin=0.1,
    fmax=100,
    n_fft=n_fft,
)

psd_nostim.plot()


# %% 6. PSD of only the three posterior channels
posterior_stim = stim.copy().pick(["PO3", "POz", "PO4"])

n_fft = min(
    int(2 * posterior_stim.info["sfreq"]),
    len(posterior_stim.times),
)

psd_posterior = posterior_stim.compute_psd(
    method="welch",
    fmin=0.1,
    fmax=100,
    n_fft=n_fft,
)

psd_posterior.plot()


# %% 7. Multitaper TFR -- stimulation
# Same TFR parameters as the all-channel analysis.
freqs = np.arange(2.0, 31.0, 1.0)
n_cycles = freqs / 2.0

tfr_stim = stim.compute_tfr(
    method="multitaper",
    freqs=freqs,
    n_cycles=n_cycles,
    time_bandwidth=2.0,
    use_fft=True,
    return_itc=False,
    average=True,
    decim=2,
    n_jobs=4,
)


# %% 8. Percent baseline correction of stimulation TFR
# MNE mode="percent" gives fractional change from baseline:
#  0.20 = 20% increase; -0.20 = 20% decrease.
tfr_stim_percent = tfr_stim.copy()
tfr_stim_percent.apply_baseline(
    baseline=(-0.3, -0.1),
    mode="percent",
)

# Plot one channel.
tfr_stim_percent.plot(
    picks="POz",
    tmin=-0.3,
    tmax=1.4,
)


# %% 9. Multitaper TFR -- no stimulation
freqs = np.arange(2.0, 31.0, 1.0)
n_cycles = freqs / 2.0

tfr_nostim = nostim.compute_tfr(
    method="multitaper",
    freqs=freqs,
    n_cycles=n_cycles,
    time_bandwidth=2.0,
    use_fft=True,
    return_itc=False,
    average=True,
    decim=2,
    n_jobs=4,
)


# %% 10. Percent baseline correction of no-stimulation TFR
tfr_nostim_percent = tfr_nostim.copy()
tfr_nostim_percent.apply_baseline(
    baseline=(-0.3, -0.1),
    mode="percent",
)

tfr_nostim_percent.plot(
    picks="POz",
    tmin=-0.3,
    tmax=1.4,
)


# %% 11. Plot PO3, POz and PO4 separately
# You can copy any one of these lines independently.
tfr_stim_percent.plot(picks="PO3", tmin=-0.3, tmax=1.4)
tfr_stim_percent.plot(picks="POz", tmin=-0.3, tmax=1.4)
tfr_stim_percent.plot(picks="PO4", tmin=-0.3, tmax=1.4)


# %% 12. Important: unbaselined TFRs are still available
# Use tfr_stim and tfr_nostim (NOT the *_percent copies) if you want to
# experiment with stim - no-stim or (stim-no-stim)/(stim+no-stim), because
# those comparative analyses in the pipeline are intentionally unbaselined.
