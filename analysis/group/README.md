# Group EEG analysis

Group analyses use the final cleaned, cue-locked EEG epochs for stimulation ON (`stim`) and OFF (`no-stim`). Run commands from the repository root. The default platform is `mac`; use `--project-root PATH` to specify a different project directory.

## Scripts and order

| Step | Script | Purpose | Output (under `<PROJECT_ROOT>`) |
|---|---|---|---|
| 1 | [`EEG_posterior_channels/G02_peak_of_alpha_calculations_with_TFRs.py`](EEG_posterior_channels/README.md) | Individual posterior TFR QC and OFF-derived alpha peak per participant | `derivatives/reports/QC/posterior_TFR_all_subjects/`; `derivatives/reports/group/EEG_posterior_channels_alpha_changes_stimOn_stimOff/individual_alpha_peaks.csv` |
| 2 (independent) | [`EEG_all_channels/run_grand_average.py`](EEG_all_channels/README.md) | Group ERP/TFR maps, ROI summaries, and individual-alpha-peak-aligned group TFR | `derivatives/reports/group/EEG_all_channels_complete_grand_average/` |
| 3 | `analysis/subject/EEG_posterior_channels/block_preparation/B03_prepare_block_alpha.py` | Validates cue-trial/stimulation-period mapping and prepares absolute alpha-power trial table | `derivatives/reports/group/EEG_posterior_channels_block_mapping/` |
| 4 | [`EEG_posterior_channels/stimulation_laterality/run_stim_laterality_tfr.py`](EEG_posterior_channels/stimulation_laterality/README.md) | Laterality-specific posterior TFR and alpha curves | `derivatives/reports/group/EEG_posterior_channels_stim_laterality/` |
| 5 | `analysis/statistics/S02_absolute_alpha_block_permutation.py` | Whole-period permutation test using B03's trial table | `derivatives/reports/group/absolute_alpha_block_permutation/` |
| 6 (optional) | `analysis/statistics/S02_sensitivity_alpha.py` | Normalized and leave-one-out sensitivity analysis | Same statistics report folder |

**Prerequisites:** Steps 1 and 2 require completed subject-level posterior/all-channel cleaning. Step 3 also requires verified stimulation-period assignments; follow the block-preparation instructions before running it. Step 4 requires both the alpha-peak CSV from step 1 and the validated `trial_alpha.csv` from step 3. Step 5 requires `trial_alpha.csv`; step 6 follows step 5. Steps 1 and 2 can run independently after preprocessing.

For runnable commands and file descriptions, see the [posterior-channel README](EEG_posterior_channels/README.md), [all-channel README](EEG_all_channels/README.md), and [laterality README](EEG_posterior_channels/stimulation_laterality/README.md).
