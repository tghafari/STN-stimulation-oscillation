# Posterior-channel analyses

## Supported scripts

| Script | Analysis | Prerequisite | Output |
|---|---|---|---|
| `../EEG_all_channels/run_grand_average.py` | Group ERP/TFR scalp and posterior ROI means | Final cleaned `stim`/`no-stim` cue epochs | `derivatives/reports/group/EEG_all_channels_complete_grand_average/` |
| `../../quality_controls/Q01_participant_eight_posterior_TFR_QC.py` | Individual-participant 3- and 8-sensor posterior TFRs, 10–12 Hz trial-level QC, and OFF-derived individual alpha-peak curves | Final cleaned cue epochs | `derivatives/reports/QC/posterior_TFR_all_subjects/` and peak CSV below |
| `stimulation_laterality/run_stim_laterality_tfr.py` | Laterality-specific STN ON/OFF and left/right comparisons | Cleaned epochs, verified block mapping CSV, individual peak CSV | `derivatives/reports/group/EEG_posterior_channels_stim_laterality/` |
| `group_utils.py` | Legacy derivative-loading and plotting helpers; not a runnable pipeline | Depends on caller | None directly |

The former `G02_individual_stimOn_stimOff_alpha_report.py` has been **merged into Q01**. Do not run the removed G02 script. Q01 uses its existing participant TFR computation to generate both standard QC and individual-alpha-peak figures, rather than independently recomputing the same TFRs.

## Commands

From the repository root, **after completing subject-level posterior and all-channel preprocessing**:

```bash
python analysis/quality_controls/Q01_participant_eight_posterior_TFR_QC.py --subjects 115 116 118 120 121 123 --platform mac
python analysis/group/EEG_all_channels/run_grand_average.py --subjects 115 116 118 120 121 123 --platform mac
```

These are independent analyses: Q01 does not grand-average subjects; G01 does. To run the laterality analysis, **first** run Q01 to create `<PROJECT_ROOT>/derivatives/reports/group/EEG_posterior_channels_alpha_changes_stimOn_stimOff/individual_alpha_peaks.csv`, and the subject block preparation `analysis/subject/EEG_posterior_channels/block_preparation/B03_prepare_block_alpha.py` to create verified `trial_alpha.csv`. Then run:

```bash
python analysis/group/EEG_posterior_channels/stimulation_laterality/run_stim_laterality_tfr.py --subjects 115 116 118 120 121 123 --platform mac
```

See [laterality README](stimulation_laterality/README.md) for the prerequisite mapping, options and three output PDFs; see [group README](../README.md) for block-level statistics and exclusions. Mac is the default platform; use `--project-root PATH` to specify another dataset root.

## Interpretations

Q01 descriptive ON/OFF TFRs use percent baseline (−0.3 to −0.1 s), while Q01 difference and normalized difference panels are unbaselined. G01's displayed ON−OFF contrast subtracts separately percent-corrected ON/OFF powers, and its ratio stays unbaselined. Laterality TFR contrasts use unbaselined absolute power. These are different outcome measures; check the figure titles and captions before comparison.
