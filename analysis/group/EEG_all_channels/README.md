# All-channel EEG group analysis

Final cleaned stimulation ON/OFF cue epochs are required. The two Python files in this folder are the implementation and its plotting runner; use the runner for the complete output.

| File | Purpose |
|---|---|
| `G01_complete_grand_average_report.py` | Calculates participant-level ERP/TFR power, group sensor averages, three-/eight-channel posterior ROI TFRs, absolute alpha QC, and a second group TFR report aligned to each participant's OFF-derived alpha peak. |
| `run_grand_average.py` | Recommended entry point: calls G01 and adds detailed all-sensor and eight-posterior-sensor figures. |

## Run

From the repository root, after subject-level EEG cleaning:

```bash
python analysis/group/EEG_all_channels/run_grand_average.py --subjects 115 116 118 120 121 123 --platform mac
```

Options include `--project-root PATH`, `--session`, `--task`, `--run`, and `--n-jobs`. G01 can be run directly for debugging, but do not run both entry points to generate the same report.

## Outputs

Under `<PROJECT_ROOT>/derivatives/reports/group/EEG_all_channels_complete_grand_average/`:
- Standard group ERP/TFR PDF and `figures/` images.
- Individual-alpha-peak-aligned group TFR PDF (`*_peak_aligned_analysis_report.pdf`) and `figures/peak_aligned/` images.

Companion analysis JSON files are written under `<PROJECT_ROOT>/data/BIDS/derivatives/group/EEG_all_channels_complete_grand_average/`.

## Analysis notes

ERP baseline: −0.1 to 0 s. TFR ON and OFF: percent baseline −0.3 to −0.1 s for each participant/sensor. The plotted ON−OFF TFR subtracts the condition-wise percent changes (percentage points); the normalized ratio `(ON-OFF)/(ON+OFF)` uses unbaselined power. Sensor maps average participants per sensor. ROI TFRs average usable channels **within each participant first**, then participants equally. Peak-aligned plots estimate each participant's alpha peak from unbaselined OFF power at 8–12 Hz, 0.3–0.8 s, shift frequency to that peak, and interpolate onto a common relative-frequency axis. This is a maximum-power estimate, not an aperiodic-corrected peak.

See [group workflow](../README.md) for other analyses.
