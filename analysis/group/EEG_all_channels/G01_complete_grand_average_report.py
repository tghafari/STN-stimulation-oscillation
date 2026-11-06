#!/usr/bin/env python
"""G01_complete_grand_average_report.py: group analysis overview

Read final cleaned all-channel cue epochs for every selected subject. In
addition, create a separate peak-aligned TFR report using participant-specific
OFF-derived alpha peaks. This reuses the TFRs computed for the standard G01
report and does not change the standard-frequency group report.

For each good EEG sensor, compute a subject-level ERP (30-Hz low-pass,
baseline -0.1 to 0 s) and multitaper TFR (2-31.5 Hz, 0.5-Hz steps,
n_cycles=f/2, time-bandwidth=2, decim=2). Display stim/no-stim TFR
with percent baseline (-0.3 to -0.1 s). Calculate stimulation minus no-stimulation absolute power difference and
normalized ratio from unbaselined power. Display the difference of the
condition-wise percent-baseline-corrected powers (percentage points), while
keeping the normalized ratio unbaselined.
Grand-average subjects independently per sensor, excluding missing/bad
channels, and save PDF figures, channel counts and analysis JSON.

Prepare subject inputs by running the posterior-channel subject pipeline first,
then the all-channel pipeline when analysing the full montage.
See this group's README for prerequisites, commands and output locations.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parents[1]
for p in (ANALYSIS_DIR / "subject" / "EEG_all_channels", ANALYSIS_DIR / "utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from pipeline_config import CONDITIONS, resolve_project_root, stage_path
from pdf_report import ParticipantPDF

ROI3 = ("PO3", "POz", "PO4")
ROI8 = ("PO3", "POz", "PO4", "O1", "Oz", "O2", "PO7", "PO8")
ERP_BASELINE = (-0.1, 0.0)
ERP_LP = 30.0
ERP_TMIN = -0.1
ERP_TMAX = 1.0
BASELINE = (-0.3, -0.1)
FREQS = np.arange(2.0, 32.0, 0.5)
N_CYCLES = FREQS / 2.0
TIME_BANDWIDTH = 2.0
DECIM = 2
ROBUST = 98.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="SpAtt")
    p.add_argument("--run", default="01")
    p.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    p.add_argument("--project-root", default=None)
    p.add_argument("--n-jobs", type=int, default=4)
    return p.parse_args()


def load_epochs(root, s, a):
    out = {}
    for c in CONDITIONS:
        f = stage_path(root, s, a.session, a.task, a.run, c, "clean", "epo")
        e = mne.read_epochs(f, preload=True, verbose=False)
        keys = [k for k in ("cue_onset_right", "cue_onset_left") if k in e.event_id]
        out[c] = e[keys] if keys else e
    return out


def good_channels(x):
    st, no = x["stim"], x["no-stim"]
    return [
        ch
        for ch in st.copy().pick("eeg").ch_names
        if ch in no.ch_names and ch not in st.info["bads"] and ch not in no.info["bads"]
    ]


def make_tfr(ep, ch, j):
    return (
        ep.copy()
        .pick([ch])
        .compute_tfr(
            method="multitaper",
            freqs=FREQS,
            n_cycles=N_CYCLES,
            time_bandwidth=TIME_BANDWIDTH,
            use_fft=True,
            zero_mean=True,
            return_itc=False,
            average=True,
            decim=DECIM,
            n_jobs=j,
            verbose=False,
        )
    )


def make_evoked(ep, ch):
    e = ep.copy().pick([ch]).average()
    e.filter(None, ERP_LP, verbose=False)
    e.apply_baseline(ERP_BASELINE)
    e.crop(ERP_TMIN, ERP_TMAX)
    return e


def baseline_percent(x):
    mask = (x.times >= BASELINE[0]) & (x.times <= BASELINE[1])
    base = x.data[0, :, mask].mean(-1, keepdims=True)
    return (
        100 * (x.data[0] - base) / np.where(np.abs(base) < np.finfo(float).eps, 1, base)
    )


def ratio(st, no):
    return (st - no) / (st + no + np.finfo(float).eps)


def scale(vals):
    z = np.concatenate(
        [np.asarray(v)[np.isfinite(v)] for v in vals if np.asarray(v).size]
    )
    m = float(np.percentile(np.abs(z), ROBUST))
    return (-m, m) if m > 0 else None


def roi(d, chs):
    av = [c for c in chs if c in d]
    return (np.mean([d[c][0] for c in av], 0), av) if av else (None, [])


def scalp(data, info, title, v):
    pos = info.get_montage().get_positions()["ch_pos"]
    chs = list(data)
    xy = {c: np.asarray(pos[c])[:2] for c in chs}
    xs = np.array([xy[c][0] for c in chs])
    ys = np.array([xy[c][1] for c in chs])
    nr = 11
    nc = 11
    free = {(r, c) for r in range(nr) for c in range(nc)}
    cells = {}
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    targets = {
        c: (
            (ymax - xy[c][1]) / max(ymax - ymin, 1e-12) * (nr - 1),
            (xy[c][0] - xmin) / max(xmax - xmin, 1e-12) * (nc - 1),
        )
        for c in chs
    }
    for ch in sorted(chs, key=lambda c: targets[c]):
        tr, tc = targets[ch]
        cell = min(free, key=lambda q: (q[0] - tr) ** 2 + (q[1] - tc) ** 2)
        cells[ch] = cell
        free.remove(cell)
    fig, axs = plt.subplots(nr, nc, figsize=(25, 23))
    [a.axis("off") for a in axs.ravel()]
    for ch in chs:
        ax = axs[cells[ch]]
        ax.axis("on")
        a, t = data[ch]
        ax.imshow(
            a,
            origin="lower",
            aspect="auto",
            extent=[t[0], t[-1], FREQS[0], FREQS[-1]],
            cmap="RdBu_r",
            vmin=v[0] if v else None,
            vmax=v[1] if v else None,
        )
        ax.axvline(0, color="k", ls="--", lw=0.4)
        ax.set_title(ch, fontsize=8)
        ax.tick_params(labelsize=5)
    fig.subplots_adjust(
        left=0.025, right=0.89, bottom=0.03, top=0.94, wspace=0.55, hspace=0.65
    )
    fig.suptitle(title)
    if v:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        cax = fig.add_axes([0.925, 0.2, 0.015, 0.6])
        sm = ScalarMappable(norm=Normalize(*v), cmap="RdBu_r")
        sm.set_array([])
        fig.colorbar(sm, cax=cax, label="Value")
    return fig


def roi_fig(a, t, title, v):
    f, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(
        a,
        origin="lower",
        aspect="auto",
        extent=[t[0], t[-1], FREQS[0], FREQS[-1]],
        cmap="RdBu_r",
        vmin=v[0] if v else None,
        vmax=v[1] if v else None,
    )
    ax.axvline(0, color="k", ls="--")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    f.colorbar(im, ax=ax)
    return f




# Individual-alpha-peak alignment
# ---------------------------------------------------------------------
# The peak definition matches analysis/quality_controls/
# Q01_participant_eight_posterior_TFR_QC.py:
#   - no-stimulation (DBS OFF) raw power;
#   - average across available good ROI8 channels;
#   - mean power from 0.3 to 0.8 s after cue onset;
#   - maximum within 8-12 Hz.
#
# The relative-frequency axis is expressed as f - individual_alpha_peak.
# Therefore zero is each participant's peak; negative values are below it
# and positive values are above it. The common range is +/- 6 Hz, which
# remains inside the available 2-31.5 Hz TFR range for peaks between
# 8 and 12 Hz.
# ---------------------------------------------------------------------

ALPHA_PEAK_SEARCH = (8.0, 12.0)
ALPHA_PEAK_TIME = (0.3, 0.8)
RELATIVE_FREQS = np.arange(-7.0, 18.0 + 0.25, 0.5)


def estimate_individual_alpha_peak(subject_tfr):
    """Estimate a participant's alpha peak from unbaselined no-stim power.

    Power is averaged over the available good ROI8 channels and 0.3-0.8 s.
    The maximum in the 8-12 Hz range is selected. This is the same peak
    definition used by the participant-level posterior TFR QC script.
    """
    available_channels = [
        ch for ch in ROI8 if ch in subject_tfr["no-stim"]
    ]
    if not available_channels:
        raise RuntimeError(
            "No good ROI8 channels are available for individual "
            "alpha-peak estimation."
        )

    example = subject_tfr["no-stim"][available_channels[0]]
    time_mask = (
        (example.times >= ALPHA_PEAK_TIME[0])
        & (example.times <= ALPHA_PEAK_TIME[1])
    )
    frequency_mask = (
        (example.freqs >= ALPHA_PEAK_SEARCH[0])
        & (example.freqs <= ALPHA_PEAK_SEARCH[1])
    )
    if not time_mask.any() or not frequency_mask.any():
        raise RuntimeError(
            "The alpha peak search frequency/time window is not present "
            "in the computed TFR."
        )

    channel_spectra = []
    for channel in available_channels:
        tfr = subject_tfr["no-stim"][channel]
        channel_spectra.append(
            tfr.data[0, :, time_mask].mean(axis=-1)
        )

    mean_spectrum = np.mean(channel_spectra, axis=0)
    candidates = np.flatnonzero(frequency_mask)
    peak_index = candidates[np.argmax(mean_spectrum[candidates])]
    peak_hz = float(example.freqs[peak_index])

    return peak_hz, available_channels


def align_tfr_to_individual_peak(data, frequencies, peak_hz):
    """Align a TFR to the participant's individual alpha peak.

    The frequency axis is expressed relative to the alpha peak.
    Values below the peak are negative; values above are positive.
    """
    target_frequencies = peak_hz + RELATIVE_FREQS

    # Allow very small floating-point differences at the boundaries.
    tolerance = 1e-6

    if (
        target_frequencies.min() < frequencies.min() - tolerance
        or target_frequencies.max() > frequencies.max() + tolerance
    ):
        raise ValueError(
            f"Alpha peak: {peak_hz:.2f} Hz\n"
            f"Requested absolute range: "
            f"{target_frequencies.min():.2f} to "
            f"{target_frequencies.max():.2f} Hz\n"
            f"Available TFR range: "
            f"{frequencies.min():.2f} to "
            f"{frequencies.max():.2f} Hz"
        )

    # Keep target frequencies inside the available range.
    target_frequencies = np.clip(
        target_frequencies,
        frequencies.min(),
        frequencies.max(),
    )

    aligned = np.empty(
        (len(RELATIVE_FREQS), data.shape[-1]),
        dtype=float,
    )

    for time_index in range(data.shape[-1]):
        aligned[:, time_index] = np.interp(
            target_frequencies,
            frequencies,
            data[:, time_index],
        )

    return aligned


def peak_aligned_scalp(data, info, times, title, vlim):
    """Plot peak-aligned TFRs in a sensor layout with one panel per channel."""
    montage = info.get_montage()
    if montage is None:
        raise RuntimeError(
            "A montage is required for the peak-aligned scalp report."
        )
    positions = montage.get_positions()["ch_pos"]
    channels = [ch for ch in data if ch in positions]
    if not channels:
        raise RuntimeError(
            "No plotted channels have positions in the EEG montage."
        )

    xy = {ch: np.asarray(positions[ch])[:2] for ch in channels}
    xs = np.array([xy[ch][0] for ch in channels])
    ys = np.array([xy[ch][1] for ch in channels])
    n_rows = 11
    n_cols = 11
    free_cells = {
        (row, col)
        for row in range(n_rows)
        for col in range(n_cols)
    }
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    targets = {
        ch: (
            (y_max - xy[ch][1]) / max(y_max - y_min, 1e-12) * (n_rows - 1),
            (xy[ch][0] - x_min) / max(x_max - x_min, 1e-12) * (n_cols - 1),
        )
        for ch in channels
    }

    cells = {}
    for channel in sorted(channels, key=lambda ch: targets[ch]):
        target_row, target_col = targets[channel]
        cell = min(
            free_cells,
            key=lambda position: (
                (position[0] - target_row) ** 2
                + (position[1] - target_col) ** 2
            ),
        )
        cells[channel] = cell
        free_cells.remove(cell)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(25, 23),
    )
    for axis in axes.ravel():
        axis.axis("off")

    image = None
    for channel in channels:
        axis = axes[cells[channel]]
        axis.axis("on")
        image = axis.imshow(
            data[channel],
            origin="lower",
            aspect="auto",
            extent=[
                times[0],
                times[-1],
                RELATIVE_FREQS[0],
                RELATIVE_FREQS[-1],
            ],
            cmap="RdBu_r",
            vmin=vlim[0] if vlim else None,
            vmax=vlim[1] if vlim else None,
        )
        axis.axhline(0, color="black", linestyle="--", linewidth=0.6)
        axis.axvline(0, color="black", linestyle=":", linewidth=0.4)
        axis.set_title(channel, fontsize=8)
        axis.tick_params(labelsize=5, length=2)

    fig.subplots_adjust(
        left=0.025,
        right=0.89,
        bottom=0.04,
        top=0.94,
        wspace=0.55,
        hspace=0.65,
    )
    fig.suptitle(title)
    if image is not None:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        color_axis = fig.add_axes([0.925, 0.2, 0.015, 0.6])
        scalar_map = ScalarMappable(
            norm=Normalize(*vlim) if vlim else None,
            cmap="RdBu_r",
        )
        scalar_map.set_array([])
        fig.colorbar(scalar_map, cax=color_axis, label="Power / change")
    return fig


def peak_aligned_roi_figure(data, times, title, vlim):
    """Plot a peak-aligned ROI TFR with relative frequency on the y-axis."""
    fig, axis = plt.subplots(
        figsize=(10, 5),
        constrained_layout=True,
    )
    image = axis.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[
            times[0],
            times[-1],
            RELATIVE_FREQS[0],
            RELATIVE_FREQS[-1],
        ],
        cmap="RdBu_r",
        vmin=vlim[0] if vlim else None,
        vmax=vlim[1] if vlim else None,
    )
    axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
    axis.axvline(0, color="black", linestyle=":", linewidth=0.6)
    axis.set_title(title)
    axis.set_xlabel("Time from cue onset (s)")
    axis.set_ylabel("Frequency relative to individual alpha peak (Hz)")
    fig.colorbar(image, ax=axis, label="Power / change")
    return fig


def build_peak_aligned_report(
    report_dir,
    figure_dir,
    derivative_dir,
    report_id,
    subjects,
    good_channels_by_subject,
    subject_tfrs,
    times,
    montage_info,
):
    """Create a separate grand-average report with participant-specific peaks.

    Condition TFRs and their difference are percent-baseline corrected at
    participant/sensor level before alignment. The normalized ratio is
    calculated from original, unbaselined power before alignment.
    """
    peak_report = ParticipantPDF(
        str(report_dir),
        report_id + "_peak_aligned",
    )
    peak_figure_dir = figure_dir / "peak_aligned"
    peak_figure_dir.mkdir(parents=True, exist_ok=True)

    peaks = {}
    peak_channels = {}
    for subject in subjects:
        peak_hz, channels = estimate_individual_alpha_peak(
            subject_tfrs[subject]
        )
        peaks[subject] = peak_hz
        peak_channels[subject] = channels

    peak_report.add_text(
        "Methods: individual-alpha-peak-aligned TFR grand averages",
        (
            "INPUT AND PREPROCESSING\\n"
            "Final cleaned, cue-locked epochs were loaded separately for "
            "stimulation ON and OFF. Attention-left and attention-right "
            "trials were combined. Only EEG channels present and marked "
            "good in both conditions contributed for each participant.\\n\\n"
            "SUBJECT-LEVEL TFR\\n"
            "For each participant, condition, and eligible sensor, trial-"
            "averaged multitaper power was computed using MNE: frequencies "
            f"{FREQS[0]:g}-{FREQS[-1]:g} Hz in "
            f"{FREQS[1] - FREQS[0]:g}-Hz steps; n_cycles = frequency/2; "
            f"time-bandwidth = {TIME_BANDWIDTH:g}; FFT enabled; "
            f"decimation = {DECIM}; ITC not calculated. "
            "The TFRs are aligned to cue onset (time 0 s).\\n\\n"
            "INDIVIDUAL ALPHA PEAK\\n"
            "Each participant's peak was selected independently from "
            "original UNBASELINED stimulation-OFF power, averaged over "
            "0.3-0.8 s after cue onset and available good sensors from "
            "PO3, POz, PO4, O1, Oz, O2, PO7 and PO8. The maximum-power "
            "frequency between 8 and 12 Hz was selected. The same peak "
            "was used for both stimulation conditions and both contrasts. "
            "This is a maximum-power estimate, not a 1/f-adjusted "
            "oscillatory peak.\\n\\n"
            "FREQUENCY ALIGNMENT\\n"
            "For each participant, relative frequency was defined as "
            "absolute frequency minus that participant's alpha peak. "
            "Thus 0 Hz on the y-axis is the individual alpha peak, "
            "negative values are below it, and positive values are "
            "above it. Each subject's frequency-by-time maps were "
            "interpolated onto the shared relative-frequency grid "
            f"{RELATIVE_FREQS[0]:+g} to {RELATIVE_FREQS[-1]:+g} Hz "
            f"in {RELATIVE_FREQS[1] - RELATIVE_FREQS[0]:g}-Hz steps "
            "BEFORE group averaging. This axis is not absolute Hz.\\n\\n"
            "BASELINE AND CONTRAST DEFINITIONS\\n"
            f"Stimulation ON and OFF maps were separately normalized "
            f"to {BASELINE[0]:g} to {BASELINE[1]:g} s using percent "
            "change from baseline. The displayed difference is "
            "ON percent change minus OFF percent change (percentage "
            "points), computed within participant/sensor. The ratio "
            "(ON - OFF)/(ON + OFF) was computed from original "
            "UNBASELINED power; no baseline correction was applied "
            "to the ratio.\\n\\n"
            "GRAND AVERAGING\\n"
            "Peak-aligned maps were averaged with equal participant "
            "weight. Whole-scalp sensor plots use each sensor's own "
            "eligible participant set (N can differ by sensor). "
            "For ROI3 and ROI8, available good channels were first "
            "averaged within each participant and then participants "
            "were averaged equally. No missing channels were "
            "interpolated for group averaging. The individual peak "
            "frequencies and channel membership are recorded below "
            "and in the accompanying analysis JSON."
        ),
        "Peak-aligned TFR methods",
    )

    peak_lines = []
    for subject in subjects:
        peak_lines.append(
            f"sub-{subject}: peak={peaks[subject]:.1f} Hz; "
            f"peak channels={', '.join(peak_channels[subject])}"
        )
    peak_report.add_text(
        "Participant alpha peaks",
        "\n".join(peak_lines),
        "Peak-aligned TFR",
    )

    aligned = {
        "no-stim": {},
        "stim": {},
        "difference": {},
        "ratio": {},
    }
    membership = {
        key: {}
        for key in aligned
    }

    for subject in subjects:
        peak_hz = peaks[subject]
        for channel in good_channels_by_subject[subject]:
            if (
                channel not in subject_tfrs[subject]["stim"]
                or channel not in subject_tfrs[subject]["no-stim"]
            ):
                continue

            stim_raw = subject_tfrs[subject]["stim"][channel]
            no_stim_raw = subject_tfrs[subject]["no-stim"][channel]
            stim_percent = baseline_percent(stim_raw)
            no_stim_percent = baseline_percent(no_stim_raw)

            subject_arrays = {
                "stim": stim_percent,
                "no-stim": no_stim_percent,
                "difference": stim_percent - no_stim_percent,
                "ratio": ratio(
                    stim_raw.data[0],
                    no_stim_raw.data[0],
                ),
            }
            for key, values in subject_arrays.items():
                aligned_values = align_tfr_to_individual_peak(
                    values,
                    stim_raw.freqs,
                    peak_hz,
                )
                aligned[key].setdefault(channel, []).append(aligned_values)
                membership[key].setdefault(channel, []).append(subject)

    group_data = {
        key: {
            channel: np.mean(subject_arrays, axis=0)
            for channel, subject_arrays in channel_data.items()
        }
        for key, channel_data in aligned.items()
    }

    # Use one shared scale for each contrast report across all channels.
    difference_vlim = scale(
        [
            data.ravel()
            for data in group_data["difference"].values()
        ]
    )
    ratio_vlim = scale(
        [
            data.ravel()
            for data in group_data["ratio"].values()
        ]
    )
    condition_vlim = scale(
        [
            data.ravel()
            for condition in ("stim", "no-stim")
            for data in group_data[condition].values()
        ]
    )

    report_specs = (
        (
            "no-stim",
            "No stimulation",
            condition_vlim,
            "Condition-wise percent baseline power change "
            "(-0.3 to -0.1 s), aligned to individual alpha peak.",
        ),
        (
            "stim",
            "Stimulation",
            condition_vlim,
            "Condition-wise percent baseline power change "
            "(-0.3 to -0.1 s), aligned to individual alpha peak.",
        ),
        (
            "difference",
            "Stimulation minus no stimulation",
            difference_vlim,
            "Difference of condition-wise percent-baseline-corrected "
            "power (percentage points); both conditions corrected over "
            "-0.3 to -0.1 s before subtraction.",
        ),
        (
            "ratio",
            "(Stimulation - no stimulation) / (Stimulation + no stimulation)",
            ratio_vlim,
            "Normalized ratio from original unbaselined power; no "
            "baseline correction. Ratio = (stim - no-stim) / "
            "(stim + no-stim).",
        ),
    )

    # Whole-scalp layouts: every channel uses only subjects with that
    # channel present and good in both conditions.
    for key, title, vlim, description in report_specs:
        data = group_data[key]
        if not data:
            continue
        channel_counts = {
            channel: len(membership[key][channel])
            for channel in data
        }
        figure = peak_aligned_scalp(
            data,
            montage_info,
            times,
            title + " — individual-alpha-peak-aligned TFR",
            vlim,
        )
        peak_report.add_figure(
            figure,
            str(peak_figure_dir / f"peak_aligned_{key}_scalp.png"),
            title + " — peak-aligned scalp TFR",
            (
                description
                + " Y-axis: frequency relative to participant alpha peak; "
                "0 Hz is the individual peak. Per-channel group N: "
                + ", ".join(
                    f"{channel}={count}"
                    for channel, count in channel_counts.items()
                )
            ),
            "Peak-aligned TFR",
        )
        plt.close(figure)

        # Posterior ROIs: average available sensors within each participant,
        # then average participant maps equally.
        for roi_channels, roi_name in (
            (ROI3, "ROI3"),
            (ROI8, "ROI8"),
        ):
            participant_maps = []
            participants = []
            channel_details = {}
            for subject in subjects:
                available = [
                    channel
                    for channel in roi_channels
                    if channel in good_channels_by_subject[subject]
                    and channel in aligned[key]
                    and subject in membership[key].get(channel, [])
                ]
                if not available:
                    continue
                participant_maps.append(
                    np.mean(
                        [
                            aligned[key][channel][
                                membership[key][channel].index(subject)
                            ]
                            for channel in available
                        ],
                        axis=0,
                    )
                )
                participants.append(subject)
                channel_details["sub-" + subject] = available

            if not participant_maps:
                continue
            roi_group = np.mean(participant_maps, axis=0)
            figure = peak_aligned_roi_figure(
                roi_group,
                times,
                f"{title} — {roi_name} peak-aligned TFR",
                vlim,
            )
            peak_report.add_figure(
                figure,
                str(
                    peak_figure_dir
                    / f"peak_aligned_{key}_{roi_name}.png"
                ),
                f"{title} — {roi_name}",
                (
                    description
                    + f" Equal-weight participant grand average, n={len(participants)}. "
                    + "Available channels per participant: "
                    + "; ".join(
                        f"{subject}: {', '.join(channels)}"
                        for subject, channels in channel_details.items()
                    )
                ),
                "Peak-aligned TFR",
            )
            plt.close(figure)

    peak_summary = {
        "subjects": subjects,
        "alpha_peak_method": (
            "Unbaselined no-stimulation power; average available good ROI8 "
            "channels and 0.3-0.8 s; maximum 8-12 Hz; 0.5-Hz bins."
        ),
        "relative_frequency_hz": RELATIVE_FREQS.tolist(),
        "condition_baseline": {
            "window_seconds": BASELINE,
            "mode": "percent",
        },
        "difference": (
            "stim percent-baseline change minus no-stim percent-baseline "
            "change (percentage points)"
        ),
        "ratio": (
            "(stim - no-stim) / (stim + no-stim) from unbaselined power"
        ),
        "participant_alpha_peaks_hz": peaks,
        "channels_used_for_peak": peak_channels,
        "channel_subject_membership": membership,
    }
    summary_path = derivative_dir / (
        report_id + "_peak_aligned_analysis.json"
    )
    summary_path.write_text(
        json.dumps(peak_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    peak_report.add_text(
        "Saved analysis details",
        str(summary_path),
        "Peak-aligned TFR",
    )
    print(f"Peak-aligned TFR report: {peak_report.pdf_fname}")


def main():
    a = parse_args()
    subs = [s.removeprefix("sub-") for s in a.subjects]
    root = resolve_project_root(a.platform, a.project_root)
    out = (
        root
        / "derivatives"
        / "reports"
        / "group"
        / "EEG_all_channels_complete_grand_average"
    )
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    deriv = (
        root
        / "data"
        / "BIDS"
        / "derivatives"
        / "group"
        / "EEG_all_channels_complete_grand_average"
    )
    deriv.mkdir(parents=True, exist_ok=True)
    rid = "complete_grand_average_" + "_".join(subs)
    report = ParticipantPDF(str(out), rid)
    ep = {s: load_epochs(root, s, a) for s in subs}
    good = {s: good_channels(ep[s]) for s in subs}
    chs = []
    for s in subs:
        for ch in good[s]:
            if ch not in chs:
                chs.append(ch)
    by = {ch: [s for s in subs if ch in good[s]] for ch in chs}
    info = ep[subs[0]]["stim"].copy().pick("eeg").info
    report.add_text(
        "Subjects included", ", ".join("sub-" + s for s in subs), "Overview"
    )
    report.add_text(
        "Subjects by channel",
        "\n".join(f"{ch} (n={len(by[ch])}): {', '.join(by[ch])}" for ch in chs),
        "Overview",
    )
    logic = "Power is computed separately for Stim and No-stim for every participant/sensor. Difference and Ratio are then computed for that participant/sensor from ORIGINAL UNBASELINED power. Participants are averaged independently per sensor. For ROI TFRs, average good sensors within each participant before averaging participants equally. Plotted difference: Stim percent-baseline power minus No-stim percent-baseline power (percentage points), each normalized to -0.3 to -0.1 s at participant/sensor level. Absolute difference QC and normalized ratio retain ORIGINAL UNBASELINED power. Ratio baseline: NONE."
    report.add_text("Final analysis logic", logic, "Overview")
    # ERP ROI summaries retained; ERP baseline is independent of TFR contrasts.
    erp = {c: {} for c in CONDITIONS}
    for c in CONDITIONS:
        for ch in chs:
            items = [make_evoked(ep[s][c], ch) for s in by[ch]]
            erp[c][ch] = (np.mean([x.data[0] for x in items], 0), items[0].times)
    for R, name in ((ROI3, "PO3/POz/PO4"), (ROI8, "8 posterior channels")):
        av = [c for c in R if c in erp["stim"]]
        no = np.mean([erp["no-stim"][c][0] for c in av], 0)
        st = np.mean([erp["stim"][c][0] for c in av], 0)
        f, ax = plt.subplots(figsize=(10, 5))
        ax.plot(erp["stim"][av[0]][1], no * 1e6, label="No stimulation")
        ax.plot(erp["stim"][av[0]][1], st * 1e6, label="Stimulation")
        ax.legend()
        ax.set_title("ERP mean " + name)
        report.add_figure(
            f,
            str(figs / f"ERP_mean{len(av)}.png"),
            "ERP mean " + name,
            "ERP baseline -0.1 to 0 s.",
            "ERP",
        )
    tfr = {s: {c: {} for c in CONDITIONS} for s in subs}
    for s in subs:
        for c in CONDITIONS:
            print(f"Computing TFR sub-{s} {c}")
            for ch in good[s]:
                tfr[s][c][ch] = make_tfr(ep[s][c], ch, a.n_jobs)
    first = next(tfr[s]["stim"][c] for s in subs for c in tfr[s]["stim"])
    times = first.times
    display = {"stim": {}, "no-stim": {}}
    diff = {}
    rat = {}
    absolute_checks = {}
    fm = (FREQS >= 8) & (FREQS <= 12)
    tm = (times >= 0.3) & (times <= 0.8)
    if not fm.any() or not tm.any():
        raise RuntimeError("Alpha QC window absent from TFR axes.")
    for ch in chs:
        eligible = [
            sub for sub in subs if ch in tfr[sub]["stim"] and ch in tfr[sub]["no-stim"]
        ]
        display["stim"][ch] = (
            np.mean([baseline_percent(tfr[sub]["stim"][ch]) for sub in eligible], 0),
            eligible,
        )
        display["no-stim"][ch] = (
            np.mean([baseline_percent(tfr[sub]["no-stim"][ch]) for sub in eligible], 0),
            eligible,
        )
        d = [
            tfr[sub]["stim"][ch].data[0] - tfr[sub]["no-stim"][ch].data[0]
            for sub in eligible
        ]
        r = [
            ratio(tfr[sub]["stim"][ch].data[0], tfr[sub]["no-stim"][ch].data[0])
            for sub in eligible
        ]
        # Difference of condition-wise baseline-normalized powers. Do not
        # normalize the raw difference by its own potentially near-zero baseline.
        baseline_d = [
            baseline_percent(tfr[sub]["stim"][ch])
            - baseline_percent(tfr[sub]["no-stim"][ch])
            for sub in eligible
        ]
        diff[ch] = (np.mean(baseline_d, 0), eligible)
        rat[ch] = (np.mean(r, 0), eligible)
        per_subject = {}
        for sub, di, ra in zip(eligible, d, r):
            st = tfr[sub]["stim"][ch].data[0]
            no = tfr[sub]["no-stim"][ch].data[0]
            np.testing.assert_allclose(di, st - no)
            np.testing.assert_allclose(ra, ratio(st, no))
            st_mean = float(st[np.ix_(fm, tm)].mean())
            no_mean = float(no[np.ix_(fm, tm)].mean())
            per_subject["sub-" + sub] = {
                "stim": st_mean,
                "no_stim": no_mean,
                "difference": st_mean - no_mean,
            }
            print(
                f"sub-{sub} {ch}: alpha absolute stim={st_mean:.6e}, no-stim={no_mean:.6e}, diff={st_mean-no_mean:.6e}"
            )
        st_group = float(np.mean([v["stim"] for v in per_subject.values()]))
        no_group = float(np.mean([v["no_stim"] for v in per_subject.values()]))
        absolute_checks[ch] = {
            "n_subjects": len(eligible),
            "stim": st_group,
            "no_stim": no_group,
            "difference": st_group - no_group,
            "per_subject": per_subject,
        }
    report.add_text(
        "Absolute alpha power: stimulation vs no stimulation",
        "Unbaselined multitaper power; 8-12 Hz, 0.3-0.8 s after cue. "
        "Each channel uses eligible participants with equal weight. "
        "Difference = stimulation - no stimulation.\\n\\n"
        + "\\n".join(
            f"{ch} (n={v['n_subjects']}): stim={v['stim']:.6e}, "
            f"no-stim={v['no_stim']:.6e}, diff={v['difference']:.6e}"
            for ch, v in absolute_checks.items()
        ),
        "TFR absolute-power QC",
    )
    results = [
        ("no-stim", "No stimulation", display["no-stim"], True),
        ("stim", "Stimulation", display["stim"], True),
        ("difference", "Stimulation - no stimulation", diff, True),
        (
            "ratio",
            "(Stimulation - no stimulation)/(Stimulation + no stimulation)",
            rat,
            False,
        ),
    ]
    # ROI TFRs: first average available good sensors within each participant,
    # then average participant-level ROI values with equal participant weight.
    # Scalp plots retain their independent per-sensor group averages.
    def participant_roi(key, channels):
        participant_values = []
        participants = []
        channel_membership = {}
        for sub in subs:
            available = [ch for ch in channels if ch in good[sub]]
            if not available:
                continue
            if key == "stim" or key == "no-stim":
                arrays = [baseline_percent(tfr[sub][key][ch]) for ch in available]
            elif key == "difference":
                arrays = [
                    baseline_percent(tfr[sub]["stim"][ch])
                    - baseline_percent(tfr[sub]["no-stim"][ch])
                    for ch in available
                ]
            else:
                arrays = [
                    ratio(tfr[sub]["stim"][ch].data[0],
                          tfr[sub]["no-stim"][ch].data[0])
                    for ch in available
                ]
            participant_values.append(np.mean(arrays, axis=0))
            participants.append(sub)
            channel_membership["sub-" + sub] = available
        if not participant_values:
            return None, participants, channel_membership
        return np.mean(participant_values, axis=0), participants, channel_membership

    for key, title, d, bl in results:
        data = {c: (x[0], times) for c, x in d.items()}
        v = scale([x[0] for x in d.values()])
        detail = (
            ("Percent baseline -0.3 to -0.1 s; condition-wise percent power change. "
             "Difference is Stim percent change minus No-stim percent change "
             "(percentage points), calculated per participant/sensor."
             if key == "difference" else
             "Percent baseline -0.3 to -0.1 s; descriptive condition only.")
            if bl
            else "NO BASELINE CORRECTION. Normalized ratio formed from raw participant/sensor power before participant averaging."
        )
        report.add_text("Analysis details", detail, "TFR - " + title)
        report.add_figure(
            scalp(data, info, title, v),
            str(figs / f"TFR_{key}_scalp.png"),
            title + " scalp",
            "Montage-informed unique-cell layout; panels cannot overlap; colorbar is outside the grid. "
            + detail,
            "TFR - " + title,
        )
        for R, name, n in ((ROI3, "PO3/POz/PO4", 3), (ROI8, "8 posterior channels", 8)):
            x, participants, membership = participant_roi(key, R)
            if x is not None:
                report.add_figure(
                    roi_fig(x, times, title + " mean " + name, scale([x])),
                    str(figs / f"TFR_{key}_mean{n}.png"),
                    title + " mean " + name,
                    "First average available good ROI sensors within each "
                    "participant, then average participants equally "
                    f"(n={len(participants)}). "
                    + detail,
                    "TFR - " + title,
                )
                report.add_text(
                    "ROI participant/channel inclusion: " + title + " " + name,
                    "\\n".join(
                        f"sub-{sub}: {', '.join(membership['sub-' + sub])}"
                        for sub in participants
                    ),
                    "TFR - " + title,
                )
    # Create a separate report with participant-specific alpha-peak alignment.
    build_peak_aligned_report(
        report_dir=out,
        figure_dir=figs,
        derivative_dir=deriv,
        report_id=rid,
        subjects=subs,
        good_channels_by_subject=good,
        subject_tfrs=tfr,
        times=times,
        montage_info=info,
    )

    manuscript = f"Cue-locked TFR power was estimated separately for each participant and retained EEG sensor in stimulation and no-stimulation conditions. For each participant and sensor, the absolute difference P_stim-P_no-stim and normalized difference (P_stim-P_no-stim)/(P_stim+P_no-stim) were calculated directly from original unbaselined power. Participant-level sensor contrasts were then averaged across eligible participants independently for each sensor. Posterior ROI TFR summaries first average available good sensors within each participant, then average the participant ROI results equally, across PO3, POz and PO4 (ROI3) or PO3, POz, PO4, O1, Oz, O2, PO7 and PO8 (ROI8). Individual-sensor scalp TFRs continue to average participants independently per sensor. For display, Stim and No-stim power were separately percent-baseline corrected over -0.3 to -0.1 s at participant/sensor level, and the displayed difference was Stim percent change minus No-stim percent change (percentage points). The normalized ratio and absolute-power QC remained unbaselined."
    report.add_text("Analysis report - manuscript style", manuscript, "Analysis")
    (deriv / f"{rid}_analysis.json").write_text(
        json.dumps(
            {
                "subjects": subs,
                "subjects_by_channel": by,
                "contrast_order": "TFR scalp: participant/sensor baseline or contrast -> mean participants per sensor; TFR ROI: participant/sensor baseline or contrast -> mean available sensors per participant -> mean participants equally",
                "difference_baseline": "stim percent change minus no-stim percent change, each relative to -0.3 to -0.1 s (percentage points)",
                "ratio_baseline": "none",
                "absolute_difference_qc_baseline": "none",
                "stim_nostim_baseline": "percent -0.3 to -0.1 s",
                "absolute_alpha_power_checks": absolute_checks,
            },
            indent=2,
        )
        + "\n"
    )
    print(report.pdf_fname)


if __name__ == "__main__":
    main()
