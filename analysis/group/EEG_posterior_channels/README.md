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


# Individual posterior EEG QC

The supported script is `Q01_participant_eight_posterior_TFR_QC.py`. It replaces the previously separate posterior G02 individual-alpha-peak script. **It is descriptive; no group average or inferential statistics are calculated.**

## Prerequisite and command

Complete the participant-level posterior and all-channel cleaning pipelines so both final `stim` and `no-stim` cue epoch files exist (loaded using `pipeline_config.stage_path`). Then, from repository root:

```bash
python analysis/quality_controls/Q01_participant_eight_posterior_TFR_QC.py --subjects 115 116 118 120 121 123 --platform mac --n-jobs 4
```

Optional: `--project-root PATH`, `--session`, `--task`, `--run`, `--alpha-peak-half-width 1.0`. Use `--help` for details. Mac is the default platform.

## Output locations

- `<PROJECT_ROOT>/derivatives/reports/QC/posterior_TFR_all_subjects/`: combined participant QC PDF, individual-alpha-peak sections, and `figures/` PNGs.
- `<PROJECT_ROOT>/derivatives/reports/group/EEG_posterior_channels_alpha_changes_stimOn_stimOff/individual_alpha_peaks.csv`: OFF-derived individual alpha frequencies (`subject,peak_hz,channels,n_off,n_on,status`). **This CSV is a required input for the laterality runner.** Rerunning Q01 replaces this CSV with the subjects in that run, so run the complete intended cohort before laterality analysis.

## Methods and caveats

Multitaper TFR: 2–31.5 Hz in 0.5-Hz bins, cycles=f/2, time-bandwidth=2, decimation=2. Three-sensor ROI: PO3, POz, PO4; eight-sensor ROI adds O1, Oz, O2, PO7, PO8. Missing/bad channels are excluded; no interpolation is performed. Stim ON/OFF descriptive TFRs have condition-wise percent baseline (−0.3 to −0.1 s). Q01 raw difference and normalized ratio TFR maps are **not baseline-corrected**. Separate 10–12 Hz alpha curves apply percent baseline independently per trial then show participant mean ± SEM across trials. The **individual alpha peak** is chosen from unbaselined OFF power in 8–12 Hz, 0.3–0.8 s and is not an aperiodic-corrected spectral peak. ON/OFF peak±half-width curves are percent-baseline corrected after trial averaging. Participants are never averaged together.

G01's group ROI TFR uses different averaging and difference-baseline conventions; see [all-channel README](../group/EEG_all_channels/README.md). Laterality workflow: [instructions](../group/EEG_posterior_channels/stimulation_laterality/README.md).
