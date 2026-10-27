# Participant-level STN stimulation laterality analysis

This is a separate, participant-level analysis. It does not modify the existing grand-average pipeline.

## Goal

For each participant, describe how left-STN and right-STN stimulation are associated with posterior TFR power on the ipsilateral versus contralateral scalp. Left and right attention cue trials are combined within each block, because attention direction is not the target of this analysis.

- Left posterior ROI: `PO3, PO7, O1`
- Right posterior ROI: `PO4, PO8, O2`

The two OFF blocks are treated as a common reference and receive equal weight, regardless of accepted trial counts. Chronological effects are intentionally not modelled, as requested.

## Interactive terminal confirmation and analysis

The BIDS files normally contain **eight task blocks**. Each consecutive pair has
the same STN condition: task blocks (1,2), (3,4), (5,6), and (7,8)
form four stimulation periods. The proposed order comes from the repository's
legacy stimulation sequence mapping, not from an automatically inferred stimulation
side. The terminal displays the proposed order, BIDS start/end times, and cue counts
for each participant.

Run:

```bash
python run_all_participants.py --subjects 101 102 103 104 105 106 107 108 110 111 112 113 114 115 116 118 119 120 121 122 123
```

For each participant, answer `y` to accept the proposed order, `n` to
correct it, `s` to skip, or `q` to stop. When correcting, type four labels,
e.g. `o o l r` for OFF, OFF, LEFT STN ON, RIGHT STN ON. The runner validates
that there are two OFF periods and one period per stimulation side. It saves
the confirmation to CSV, **runs that participant immediately**, and then
moves to the next participant. If an analysis fails, it records the error
and continues. To confirm labels without computing TFRs, add `--review-only`.

The CSV and HTML review remain available as an audit trail; you no longer need
to edit the CSV manually. Four-condition-period mapping is required; if a
participant has neither eight task blocks nor four condition periods in BIDS,
the runner flags and skips them instead of guessing.

**Special caution:** legacy records indicate that sub-110 may have VLM rather
than STN stimulation. Do not accept an STN mapping for this participant
without checking the source record.

Mac is the default platform. Optional arguments: `--platform bluebear`,
`--project-root /path/to/STN-in-PD`, `--session 01`, `--task SpAtt`,
`--run 01`, `--n-jobs 4`.

## Analysis of the paired task blocks

Each pair is treated as one stimulation-condition period. Cleaned cue epochs
are selected from both task blocks, then combined within the condition period.
The two OFF condition periods are averaged with equal weight for the reference.
The analysis uses the existing cleaned epochs and does not alter preprocessing.

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
