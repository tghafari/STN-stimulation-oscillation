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
