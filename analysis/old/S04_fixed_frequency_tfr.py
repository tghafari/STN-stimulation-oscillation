"""Fixed-frequency posterior TFR sensitivity analyses.

These routines do not use individual alpha peaks. They compute
trial-averaged unbaselined linear multitaper power for each participant,
average available good ROI sensors within condition, and form the
dimensionless (ON-OFF)/(ON+OFF) TFR. All participants share absolute
frequency and cue-relative time axes.

The signed test uses two-sided spatiotemporal cluster permutations over
the prespecified 0.3-0.8 s attention interval and all 2-31.5 Hz bins.
The magnitude test uses whole-period label permutations with a mean
absolute normalised effect across participants, frequency and time.
"""
from __future__ import annotations

import numpy as np
import mne
from mne.stats import permutation_cluster_1samp_test, combine_adjacency

FREQS = np.arange(2.0, 32.0, 0.5)
WINDOW = (0.3, 0.8)
ROI8 = ("PO3", "POz", "PO4", "O1", "Oz", "O2", "PO7", "PO8")
ROI3 = ("PO3", "POz", "PO4")


def power_epochs(epochs, channels, n_jobs, average=True):
    """Return linear power and times, without baseline correction."""
    tfr = epochs.copy().pick(channels).compute_tfr(
        method="multitaper", freqs=FREQS, n_cycles=FREQS / 2,
        time_bandwidth=2.0, use_fft=True, zero_mean=True,
        return_itc=False, average=average, decim=2,
        n_jobs=n_jobs, verbose=False,
    )
    return tfr.data, tfr.times


def normalise(on, off):
    """Normalise unbaselined linear powers after condition averaging."""
    denom = on + off
    if not np.isfinite(denom).all() or np.any(denom <= 0):
        raise ValueError("Invalid ON+OFF denominator")
    return (on - off) / denom


def participant_map(epochs, channels, n_jobs):
    """Compute sensor-averaged ON and OFF maps for one participant."""
    on, times = power_epochs(epochs["stim"], channels, n_jobs)
    off, off_times = power_epochs(epochs["no-stim"], channels, n_jobs)
    if not np.array_equal(times, off_times):
        raise ValueError("Condition TFR time axes do not match")
    return normalise(on.mean(axis=0), off.mean(axis=0)), times


def signed_cluster_test(maps, times, permutations, seed):
    """Two-sided frequency-by-time cluster permutation on group maps.

    Cluster correction is across the complete frequency x attention-time
    search space. This is not a cluster onset localisation test.
    """
    attention = (times >= WINDOW[0]) & (times <= WINDOW[1])
    values = maps[:, :, attention]
    # MNE expects samples x observations; adjacency matches freq x time.
    adjacency = combine_adjacency(len(FREQS), int(attention.sum()))
    stat, clusters, pvals, _ = permutation_cluster_1samp_test(
        values, n_permutations=permutations,
        threshold=None, tail=0, adjacency=adjacency,
        out_type="mask", seed=seed, verbose=False,
    )
    significant = np.zeros(values.shape[1:], dtype=bool)
    summaries = []
    for index, (mask, pval) in enumerate(zip(clusters, pvals)):
        if pval <= 0.05:
            significant |= mask
            frequency_indices, time_indices = np.where(mask)
            t = times[attention]
            summaries.append({
                "cluster": index,
                "p": float(pval),
                "f_low": float(FREQS[frequency_indices].min()),
                "f_high": float(FREQS[frequency_indices].max()),
                "t_start": float(t[time_indices].min()),
                "t_end": float(t[time_indices].max()),
                "n_bins": int(mask.sum()),
            })
    full_mask = np.zeros(maps.shape[1:], dtype=bool)
    full_mask[:, attention] = significant
    return full_mask, summaries


def plot_map(mean, times, title, mask=None):
    """Display absolute-frequency normalised grand-average TFR."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    limit = max(float(np.nanpercentile(np.abs(mean), 99)), 1e-8)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    image = ax.pcolormesh(
        times, FREQS, mean, shading="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    if mask is not None and mask.any():
        ax.contour(
            times, FREQS, mask.astype(float),
            levels=[0.5], colors="black", linewidths=2,
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set(
        title=title, xlabel="Time from cue onset (s)",
        ylabel="Absolute frequency (Hz)",
    )
    fig.colorbar(image, ax=ax, label="(ON - OFF) / (ON + OFF)")
    return fig
