# Posterior-channel group analyses

**First process every participant with the posterior-channel subject runner**, then run the all-channel subject pipeline for the same participants. These group scripts analyse posterior-channel results; they do not perform preprocessing.

From the repository root:

```bash
python analysis/group/EEG_posterior_channels/G01_concatenated_epochs_report.py --subjects 101 102 103
python analysis/group/EEG_posterior_channels/G02_grand_average_report.py --subjects 101 102 103
```

- **G01:** concatenate retained cue epochs before ERP/TFR computation; subjects with more epochs contribute more trials. Includes PO3/PO4/POz and posterior ROI comparisons, stimulation contrasts and PDF report.
- **G02:** calculate subject-level ERP/TFR first, then grand-average participants with equal subject weighting. Includes cue/grating ERP views, channel/ROI TFR and stimulation contrasts.

Stim/no-stim TFR displays use percent baseline (-0.3 to -0.1 s). Ratio is calculated from unbaselined power as `(stim - no-stim)/(stim + no-stim)`; difference baseline handling is selected interactively where supported. Inspect the figure captions and saved analysis notes for the exact choice.

All-channel group analysis: `analysis/group/EEG_all_channels/run_grand_average.py`.
