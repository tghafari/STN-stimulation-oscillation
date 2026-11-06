# STN stimulation laterality: posterior EEG

`run_stim_laterality_tfr.py` creates descriptive, participant-level TFR and alpha-curve reports comparing stimulation side and posterior sensor hemisphere.

## Prerequisites

- Final cleaned cue epochs for `stim` and `no-stim`.
- Verified `trial_alpha.csv` from `analysis/subject/EEG_posterior_channels/block_preparation/B03_prepare_block_alpha.py`, mapping cleaned trials to stimulation periods.
- `individual_alpha_peaks.csv` from `analysis/group/EEG_posterior_channels/G02_peak_of_alpha_calculations_with_TFRs.py`. Peaks are estimated using unbaselined DBS OFF power, not stimulation contrasts.

If cleaned epochs change, regenerate the trial mapping and alpha-peak CSV before rerunning laterality.

## Run

From the repository root:

```bash
python analysis/group/EEG_posterior_channels/stimulation_laterality/run_stim_laterality_tfr.py --subjects 115 116 118 120 121 123 --platform mac
```

Optional: `--project-root PATH`, `--trial-csv PATH`, `--peaks-csv PATH`, `--output-dir PATH`, `--session`, `--task`, `--run`, and `--n-jobs`.

## Outputs

All three PDFs are saved in `<PROJECT_ROOT>/derivatives/reports/group/EEG_posterior_channels_stim_laterality/`:

| PDF | Comparison |
|---|---|
| `posterior_TFR_right_STN_ON.pdf` | Right STN ON versus pooled OFF, including left/right posterior sensor comparisons. |
| `posterior_TFR_left_STN_ON.pdf` | Left STN ON versus pooled OFF, including left/right posterior sensor comparisons. |
| `posterior_TFR_right_vs_left_STN_ON.pdf` | Right STN ON versus left STN ON, without OFF trials. |

Left posterior sensors: PO3, PO7, O1. Right: PO4, PO8, O2. Only sensors good in the required conditions are used. Multitaper TFR contrasts are **unbaselined absolute-power differences**. Alpha time courses use condition-wise percent baseline (−0.3 to −0.1 s), averaged around each participant's saved alpha peak ±1 Hz. Each report has a methods page and one page per eligible participant; missing or inconsistent trial mappings are reported and skipped. These plots do not perform inferential statistics.
