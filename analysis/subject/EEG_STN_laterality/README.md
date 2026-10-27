# Participant-level STN stimulation laterality analysis

This is a separate, participant-level analysis. It does not modify the existing grand-average pipeline.

## Goal

For each participant, describe how left-STN and right-STN stimulation are associated with posterior TFR power on the ipsilateral versus contralateral scalp. Left and right attention cue trials are combined within each block, because attention direction is not the target of this analysis.

- Left posterior ROI: `PO3, PO7, O1`
- Right posterior ROI: `PO4, PO8, O2`

The two OFF blocks are treated as a common reference and receive equal weight, regardless of accepted trial counts. Chronological effects are intentionally not modelled, as requested.

## Required confirmatory step — do not skip

Run the runner once in review-only mode:

```bash
python run_all_participants.py --subjects 102 103 104 107 110 112 113 114 115 116 118 119 120 121 122 123 --review-only
```

It reads each participant's BIDS `events.tsv`, identifies `block_onset` / `block_end` intervals, and shows the actual onset, offset, duration, left/right cue counts, event types, and the proposed sequence from the legacy stimulation mapping. The proposed mapping is not accepted automatically.

Open:

`derivatives/reports/group/EEG_STN_laterality/confirm_stimulation_blocks.html`

Then edit:

`derivatives/reports/group/EEG_STN_laterality/confirm_stimulation_blocks.csv`

For every block, enter `confirmed_condition` as exactly one of:

- `off`: an OFF block
- `stim_left`: left STN ON
- `stim_right`: right STN ON
- `exclude`: use this for all four rows if the entire participant is to be excluded (e.g. a non-STN stimulation case)

Set `confirmed` to `yes` only after manually checking the source record and BIDS time range. Each analysed participant must have exactly four confirmed blocks: two `off`, one `stim_left`, and one `stim_right`. Participants with unknown mappings or non-four block records are not analysed until resolved.

The CSV is regenerated on each runner start but existing confirmations are preserved only when subject, block index, onset and offset are identical. If those times change, reconfirm.

## Run the analysis

After reviewing and editing the CSV, run the same command without `--review-only`:

```bash
python run_all_participants.py --subjects 102 103 104 107 110 112 113 114 115 116 118 119 120 121 122 123
```

Mac is the default. Optional arguments: `--platform bluebear`, `--project-root /path/to/STN-in-PD`, `--session 01`, `--task SpAtt`, `--run 01`, `--n-jobs 4`.

Run from this folder, or from repository root use:

```bash
python analysis/subject/EEG_STN_laterality/run_all_participants.py --subjects 102 103 104 --review-only
```

## Outputs

Under `<PROJECT_ROOT>/derivatives/reports/group/EEG_STN_laterality/`:

- `confirm_stimulation_blocks.html`: readable BIDS block review
- `confirm_stimulation_blocks.csv`: user-confirmed block labels
- `sub-XXX_STN_laterality_report.pdf`: one report per participant
- `sub-XXX_STN_laterality_alpha.csv`: alpha-band numeric summaries
- `sub-XXX_STN_laterality_analysis.json`: parameters, blocks, channel availability and audit trail
- `runner_summary.csv`: success/failure for each participant

## Calculations

TFR: multitaper, 2–31.5 Hz in 0.5 Hz steps, `n_cycles = frequency / 2`, time-bandwidth 2, FFT enabled, decimation 2, averaged across trials within each block.

For each sensor and participant:
1. TFR power is calculated separately for each of the four blocks.
2. OFF reference power is the equal-weight average of the two OFF block TFRs.
3. Difference = raw linear `P_stim - P_OFF`.
4. Ratio = `(P_stim - P_OFF) / (P_stim + P_OFF)`.
5. Sensor-level contrast maps are averaged over the left or right posterior ROI to produce ipsilateral and contralateral results for each stimulated STN side.

Percent baseline correction (`-0.3 to -0.1 s`) is used only for descriptive condition-power displays. It is never used for difference or ratio. Numeric alpha summaries use unbaselined linear power in 8–12 Hz and 0.3–0.8 s after cue onset, preserving comparability with the existing project QC summary window.

## Statistical limitation

This report estimates and visualizes participant-specific effects. It does not claim conventional per-participant cluster-permutation significance: each participant has only one ON block per STN side and two OFF blocks, and stimulation side is fixed by block. Treating trials as freely permutable independent units would risk pseudoreplication. Block sequence is shown for confirmation but chronological effects are not modelled by design.
