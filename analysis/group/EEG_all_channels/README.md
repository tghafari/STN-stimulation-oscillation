# EEG all-channel group analysis

This folder contains the grand-average group analysis for the final cleaned all-channel EEG data.

## Analysis implemented

`G01_grand_average_report.py` uses the final `desc-clean_epo.fif` files produced by `analysis/subject/EEG_all_channels/preprocessing` and analyses **cue-onset epochs only**. Attention-left and attention-right cue trials are combined within stimulation condition.

There is deliberately **no concatenated-epochs group analysis**. Each subject is averaged first. For each EEG channel, only subjects for whom that channel is available and good in both stimulation and no-stimulation cleaned datasets contribute to that channel's grand average. Missing channels are not interpolated at group level.

The generated report places the subject/channel availability section **before** the manuscript-style analysis section.

### ERP output order

1. Stimulation versus no stimulation for all available EEG channels in scalp layout.
2. PO3, POz and PO4 shown separately in left-to-right anatomical order.

ERP parameters match the cue ERP approach used in the existing posterior-channel group analysis: trial mean, 30-Hz low-pass, baseline -0.1 to 0 s, cue onset at 0 s.

### TFR output order

For each of the following results, the report shows:

1. all available EEG channels in scalp layout;
2. PO3, POz and PO4 separately;
3. a posterior mean obtained by averaging the available PO3/POz/PO4 channels within each subject first, then grand-averaging subjects.

The TFR sections are ordered as:

1. no stimulation;
2. stimulation;
3. stimulation - no stimulation;
4. (stimulation - no stimulation) / (stimulation + no stimulation).

TFR parameters exactly follow `analysis/group/EEG_posterior_channels/G02_grand_average_report.py`: multitaper power, 2-31.5 Hz in 0.5-Hz steps, `n_cycles = frequency / 2`, time-bandwidth 2, decimation 2, FFT enabled, ITC disabled, and trial averaging enabled. Stimulation and no-stimulation displays use percent baseline correction from -0.3 to -0.1 s. The normalized ratio always uses original unbaselined power. The script asks at runtime whether the difference should follow the optional baseline-correction choice used by the posterior grand-average script.

For interpretability, every TFR result gets a robust symmetric scale based on the 98th percentile of the absolute displayed values. The exact same group TFR object and scale are reused for its scalp-layout and enlarged posterior views. Scalp-layout backgrounds are forced to white.

## Run

Mac example:

```bash
python analysis/group/EEG_all_channels/G01_grand_average_report.py \
  --subjects 115 116 118 119 \
  --platform mac
```

BlueBEAR example:

```bash
python analysis/group/EEG_all_channels/G01_grand_average_report.py \
  --subjects 115 116 118 119 \
  --platform bluebear \
  --n-jobs 8
```

You can override the project root with `--project-root`.

## Outputs

The PDF report and figures are written under:

`<PROJECT_ROOT>/derivatives/reports/group/EEG_all_channels_grand_average/`

Group FIF/HDF5 outputs, the per-channel subject availability CSV, and the analysis-parameter JSON are written under:

`<PROJECT_ROOT>/data/BIDS/derivatives/group/EEG_all_channels_grand_average/`
