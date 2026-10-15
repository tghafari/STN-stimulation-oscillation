# All-channel EEG group analysis

**Prerequisite:** for every included participant, run posterior-channel preprocessing **first**, then complete the all-channel subject pipeline including manual QC.

From the repository root:

```bash
python analysis/group/EEG_all_channels/run_grand_average.py --subjects 101 102 103 --platform mac
```

On BlueBEAR add `--platform bluebear`; `--n-jobs 8` controls parallel TFR work.

The runner calls `G01_complete_grand_average_report.py`. It reads final cleaned cue epochs, combines left/right attention, excludes channels marked bad in either condition, and calculates subject-level results before group averaging. **Each EEG channel uses its own eligible subject set** (no interpolation); the report records subject counts per channel.

ERP: trial mean, 30-Hz low-pass, baseline -0.1 to 0 s. TFR: multitaper 2–31.5 Hz (0.5-Hz steps), cycles = frequency/2, time-bandwidth=2, decim=2. Stim/no-stim descriptive power is percent-baselined (-0.3 to -0.1 s). Difference `stim - no-stim` and normalized ratio `(stim - no-stim)/(stim + no-stim)` are both calculated **from unbaselined participant power**, then averaged by channel. Scalp-layout and posterior ROI figures are saved in a PDF report.

Report: `<PROJECT_ROOT>/derivatives/reports/group/EEG_all_channels_complete_grand_average/`; analysis JSON: `<PROJECT_ROOT>/data/BIDS/derivatives/group/EEG_all_channels_complete_grand_average/`.

The `Q01`, `Q02` and `S01` scripts are additional QC/statistical analyses, not required to run the grand average.
