# Posterior-channel subject pipeline (run first)

**Run this pipeline for every participant before the all-channel pipeline.** It creates the raw BIDS recording and participant report used downstream.

From the repository root:

```bash
python analysis/subject/EEG_posterior_channels/run_subject_pipeline.py --subjects 101
```

For several participants use `--subjects 101 102 103`; for an inclusive range use `--range 101 110`. Choose **Mac** at the platform prompt for interactive preprocessing.

## Processing steps

1. **P01 BrainVision → BIDS:** locate/read the participant's BrainVision data, inspect/correct event annotations, write raw BIDS, and document recording/event information.
2. **P02 stimulation segmentation:** use saved or manually chosen stim/no-stim intervals, inspect raw/PSD and save condition-specific raw FIF files.
3. **P03 cue epoching:** filter 0.1–100 Hz, epoch cue onset (-0.5 to 1.6 s), inspect PyPREP channel suggestions/PSD, manually reject trials while viewing PO3/PO4/POz, and save cleaned epochs.
4. **A01 ERP:** average retained trials for each stimulation condition and cue/grating alignment, low-pass at 30 Hz, baseline-correct and save evoked results.
5. **A02 TFR:** multitaper 2–30 Hz (1-Hz spacing; cycles = frequency/2), combine left/right attention, compare posterior-channel stimulation conditions, and save TFR results/report figures.

Crop times are kept in `analysis/subject/stimulation_cropped_time.json`. Manual decisions require interactive MNE plots. This runner executes all five stages, not just BIDS conversion.

## Next: all-channel subject pipeline

After the posterior run finishes for a participant, run:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --subjects 101 --platform mac
```

This reuses the **raw BIDS recording**, not the three-channel cleaned epochs. It adds full-montage PyPREP/manual channel QC, optional average reference, ICA, manual epoch rejection, ERP and TFR. See `analysis/subject/EEG_all_channels/README.md`.

Group posterior analyses are in `analysis/group/EEG_posterior_channels/`.
