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
| `posterior_TFR_right_vs_left_STN_ON.pdf` | Normalised right-STN ON versus left-STN ON TFRs, separate alpha curves, and peak-aligned grand average. |

Left posterior sensors: PO3, PO7, O1. Right: PO4, PO8, O2. Only sensors good in the required conditions are used. Multitaper TFR contrasts are **unbaselined absolute-power differences**. Alpha time courses use condition-wise percent baseline (−0.3 to −0.1 s), averaged around each participant's saved alpha peak ±1 Hz. Each report has a methods page and one page per eligible participant; missing or inconsistent trial mappings are reported and skipped. These plots do not perform inferential statistics.

## Third report: symmetric normalisation and peak-aligned grand average

**Only the third PDF changes.** The right-STN ON and left-STN ON
versus OFF reports keep their previous five-panel layouts and
unbaselined ON-minus-OFF TFR differences.

For each participant and each sensor hemisphere, the two upper panels
of `posterior_TFR_right_vs_left_STN_ON.pdf` are:

```text
(Right-STN-ON power - Left-STN-ON power)
-------------------------------------------------
(Right-STN-ON power + Left-STN-ON power)
```

The numerator and denominator are computed at every frequency and
time point from **linear unbaselined multitaper power**. Each
condition's trials are averaged before this calculation, and the
posterior sensors within a hemisphere are averaged before the
normalisation. The result is dimensionless (not a percent or a
microvolt-squared difference), bounded between -1 and +1 when
the condition powers are nonnegative. A non-positive or nonfinite
denominator triggers an error rather than an arbitrary replacement.

The two lower panels of the third report continue to display the
**separate condition-wise percent-baseline-corrected alpha curves**
for right- and left-STN stimulation. They are not divided by
the sum of the two conditions: these plots show each condition
independently rather than an ON-minus-ON contrast.

The participant TFRs have **two black horizontal lines** at
their individual alpha peak **minus 1 Hz** and **plus 1 Hz**.
The individual peak is read from:

`<PROJECT_ROOT>/derivatives/reports/group/individual_alpha_peaks.csv`

An explicit `--peaks-csv PATH` can override this location.
The underlying alpha-peak CSV is derived from DBS OFF power.

### Final grand-average page (third PDF only)

The final page contains **two grand-average normalised TFRs**:
one for left posterior sensors and one for right posterior sensors.
It does **not** average power at the same absolute frequency across
participants, because individuals have different alpha peaks.

For every eligible participant:

1. Compute the normalised right-minus-left-STN TFR separately for
   left and right sensor groups.
2. Subtract the participant's alpha peak from the frequency axis.
   For example, 8 Hz in an 8-Hz-peak participant and 10 Hz in a
   10-Hz-peak participant are both plotted at **0 Hz relative to peak**.
3. Linearly interpolate each normalised TFR to a common relative
   frequency grid from **-5 to +5 Hz**, sampled every **0.5 Hz**.
4. Interpolate onto a common -0.5 to 1.5 s time grid in 10-ms steps.
   No frequency or time extrapolation is permitted.
5. Average the aligned TFRs across participants with **equal
   participant weights**. Left and right sensor groups are kept
   separate.

The y-axis is **frequency relative to the individual alpha peak
(Hz)**, not absolute Hz. A dotted line marks peak (0 Hz), and solid
black lines mark +/-1 Hz. The title reports the actual number of
participants included in the grand average. Participants lacking
sufficient frequency/time support are not included in this page and
are reported in the terminal; their individual pages remain.
No inferential statistics or baseline correction of TFRs are applied.
