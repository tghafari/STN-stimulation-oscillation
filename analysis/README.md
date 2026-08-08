# STN single-subject EEG analysis

This folder contains a single-subject runner based on the analysis workflow in the repository:

- BrainVision -> BIDS conversion
- stimulation on / off segmentation
- cue-locked epoching
- manual bad-epoch rejection from the time-series browser
- ERP analysis
- TFR analysis
- participant PDF reporting

## What to run

Use one participant at a time:

```bash
python run_single_subject.py --subject 115
```

You can also change the default paths if your data are stored somewhere else:

```bash
python run_single_subject.py \
  --subject 115 \
  --project-root "/path/to/STN-in-PD" \
  --github-root "/path/to/STN-stimulation-oscillation" \
  --data-root "/path/to/data/data-organised" \
  --bids-root "/path/to/data/BIDS"
```

## Manual rejection step

The script opens the epoch browser during preprocessing. That browser is the place where you inspect the time series and remove bad trials manually.

## Outputs

For each participant the script writes:

- BIDS EEG files
- segmented stim / no-stim FIF files
- cleaned epoch FIF files
- ERP FIF files
- TFR HDF5 files
- PNG figures
- participant PDF report

## Notes for new users

1. Install the dependencies listed in `requirements.txt`.
2. Keep the BrainVision raw data in the same structure used by the repository.
3. Update the crop table inside `run_single_subject.py` if you add new participants.
4. The script assumes the existing `analysis/utils/pdf_report.py` helper is available in the repository.

# Analysis folder guide

This analysis pipeline is organized into the same conceptual stages used in the repository:

## preprocessing

- **P01_first_look_BIDS_conversion**: reads BrainVision EEG, fixes triggers / annotations, writes BIDS, and stores QC information in the participant report.
- **P02_segmenting_stim**: crops the continuous recording into stimulation on and stimulation off segments, checks the PSD before filtering, then saves filtered segmented FIF files.
- **P03_epoching_SpAtt**: loads the segmented data, finds noisy channels, shows the epoch browser for manual trial rejection, and saves cleaned cue-locked epochs.

## sensor

- **A01_ERP**: computes cue-locked and grating-locked evoked responses for stimulation on and stimulation off, then plots the comparison at the posterior channels.
- **A02_three_channel_TFR_PDF**: computes multitaper TFRs on the posterior channels, then plots stimulation minus no stimulation and the ratio measure.

## Running order

1. `P01_first_look_BIDS_conversion`
2. `P02_segmenting_stim`
3. `P03_epoching_SpAtt`
4. `A01_ERP`
5. `A02_three_channel_TFR_PDF`

The single-subject runner in the repository root executes this order for one participant.
