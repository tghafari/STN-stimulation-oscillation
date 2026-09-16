# EEG channel-power ratio ROI analysis

This folder contains an **independent alternative group analysis**. It does not modify or replace the analyses in `EEG_all_channels`.

## Order of operations

For each participant and each available posterior channel:

1. calculate stimulation TFR power;
2. calculate no-stimulation TFR power;
3. calculate the channel-wise normalized ratio:

   `(stim power - no-stim power) / (stim power + no-stim power)`

Then, within that participant:

- ROI3 = mean of available channel-wise ratios for `PO3, POz, PO4`;
- ROI8 = mean of available channel-wise ratios for `PO3, POz, PO4, O1, Oz, O2, PO7, PO8`.

Finally:

- group ROI3 = mean of participant ROI3 ratios;
- group ROI8 = mean of participant ROI8 ratios.

There is no EEG-domain channel averaging before the TFR, no averaging of channel power before forming the ratio, and no ratio of group-level powers.

Missing/rejected sensors are not interpolated. The report records the exact channels used for each participant.

## Run

```bash
python G01_channel_power_ratio_ROI.py --subjects 102 103 104 107 110 112
```

Mac is the default platform. Use `--project-root` if needed.

## Output

Report:

`derivatives/reports/group/EEG_channel_power_ratio_ROI/`

Metadata:

`data/BIDS/derivatives/group/EEG_channel_power_ratio_ROI/`
