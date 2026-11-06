# Posterior-channel EEG analysis

| File | Purpose |
|---|---|
| `G02_peak_of_alpha_calculations_with_TFRs.py` | Individual-participant posterior TFR quality control, fixed 10–12 Hz alpha curves, and OFF-derived individual alpha peaks. No group averaging. |
| `group_utils.py` | Optional legacy helpers for loading derivatives and plotting; not a standalone script. |
| `stimulation_laterality/run_stim_laterality_tfr.py` | Compares left/right STN stimulation using verified stimulation periods and participant-specific alpha peaks. |

## Run G02

**Prerequisite:** final cleaned cue epochs for both `stim` and `no-stim`, produced by the subject EEG pipelines.

```bash
python analysis/group/EEG_posterior_channels/G02_peak_of_alpha_calculations_with_TFRs.py --subjects 115 116 118 120 121 123 --platform mac
```

G02 computes trial-averaged multitaper power for each participant. It identifies the individual alpha peak as the maximum **unbaselined OFF power** between 8 and 12 Hz, averaged over good posterior sensors and 0.3–0.8 s after cue onset. It then plots ON/OFF TFRs and alpha time courses around that peak (default ±1 Hz). The selected frequency is not an aperiodic-corrected oscillatory peak. Other plots show three-/eight-sensor ON/OFF TFRs, raw difference and normalized ratio, and single-trial 10–12 Hz alpha mean ± SEM.

**Outputs:**
- `<PROJECT_ROOT>/derivatives/reports/QC/posterior_TFR_all_subjects/`: PDF and PNG figures.
- `<PROJECT_ROOT>/derivatives/reports/group/EEG_posterior_channels_alpha_changes_stimOn_stimOff/individual_alpha_peaks.csv`: peak frequency and QC counts for each selected participant.

Use `--alpha-peak-half-width 1.0` to change the plotted peak-centred frequency band. The CSV is replaced each run, so include all intended participants before laterality analysis.

## Run stimulation laterality

First run G02 for the peak CSV and `analysis/subject/EEG_posterior_channels/block_preparation/B03_prepare_block_alpha.py` for verified `trial_alpha.csv`. Then:

```bash
python analysis/group/EEG_posterior_channels/stimulation_laterality/run_stim_laterality_tfr.py --subjects 115 116 118 120 121 123 --platform mac
```

Three PDFs are written to `<PROJECT_ROOT>/derivatives/reports/group/EEG_posterior_channels_stim_laterality/`. See [laterality README](stimulation_laterality/README.md) for the three comparisons and optional CSV paths. For group grand averages, see [all-channel README](../EEG_all_channels/README.md).
