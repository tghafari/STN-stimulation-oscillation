#!/usr/bin/env python
"""Compatibility/synchronization entry point for all-channel group grand averages.

This wrapper keeps G01's channel-wise grand-average logic but synchronizes display
and analysis constants with the finalized subject-level ERP/TFR pipeline and works
around MNE plotting incompatibilities in the local environment.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import G01_grand_average_report as pipeline

# Match finalized subject-level analysis.
pipeline.POSTERIOR = ("PO3", "POz", "PO4", "O1", "Oz", "O2", "PO7", "PO8")
pipeline.ERP_BASELINE = (-0.1, 0.0)
pipeline.ERP_LP_HZ = 30.0
pipeline.ERP_TMIN = -0.1
pipeline.ERP_TMAX = 0.5
pipeline.BASELINE = (-0.3, -0.1)
pipeline.FREQS = np.arange(2.0, 31.0, 1.0)
pipeline.N_CYCLES = pipeline.FREQS / 2.0
pipeline.TIME_BANDWIDTH = 2.0
pipeline.DECIM = 2
pipeline.PLOT_TMIN = -0.3
pipeline.PLOT_TMAX = 1.4
pipeline.ROBUST_PERCENTILE = 98.0

# Comparative TFRs match subject-level A02: difference and normalized ratio are
# calculated from unbaselined power. Do not ask for an alternative group rule.
pipeline.get_difference_baseline_choice = lambda: False


def scalp_tfr_mne_compatible(tfr, vlim):
    """MNE-version-compatible TFR scalp layout with shared image limits."""
    fig = tfr.plot_topo(
        tmin=pipeline.PLOT_TMIN, tmax=pipeline.PLOT_TMAX,
        fmin=2, fmax=30, baseline=None, mode=None,
        cmap="RdBu_r", show=False,
    )
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        if vlim is not None and None not in vlim:
            for image in ax.images:
                image.set_clim(*vlim)
    if vlim is not None and None not in vlim:
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(*vlim))
        sm.set_array([])
        cax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_ticks(np.linspace(vlim[0], vlim[1], 5))
        cb.set_label("TFR power")
    return fig


pipeline.scalp_tfr = scalp_tfr_mne_compatible

# MNE's plot_compare_evokeds(..., axes='topo') fails in the installed version
# with an empty internal time vector. Replace only that topo case; all ordinary
# plot_compare_evokeds calls are delegated unchanged.
_original_compare = pipeline.mne.viz.plot_compare_evokeds


def _erp_scalp_layout(evokeds, picks, show=False, **kwargs):
    """Plot each requested ERP sensor at its 2-D electrode position."""
    first = next(iter(evokeds.values()))
    picks = [ch for ch in picks if ch in first.ch_names]
    pos = {}
    montage = first.get_montage()
    if montage is not None:
        ch_pos = montage.get_positions().get("ch_pos", {})
        for ch in picks:
            if ch in ch_pos:
                xyz = np.asarray(ch_pos[ch], float)
                pos[ch] = xyz[:2]
    # Fall back to a regular grid only for channels lacking usable montage positions.
    if len(pos) != len(picks):
        ncol = int(np.ceil(np.sqrt(len(picks))))
        for i, ch in enumerate(picks):
            pos.setdefault(ch, np.array([i % ncol, -(i // ncol)], float))
    xy = np.vstack([pos[ch] for ch in picks])
    xmin, ymin = xy.min(axis=0); xmax, ymax = xy.max(axis=0)
    dx = max(xmax - xmin, 1e-9); dy = max(ymax - ymin, 1e-9)
    norm = {ch: ((pos[ch][0]-xmin)/dx, (pos[ch][1]-ymin)/dy) for ch in picks}
    fig = plt.figure(figsize=(13, 10), facecolor="white")
    w, h = 0.13, 0.105
    for ch in picks:
        x, y = norm[ch]
        ax = fig.add_axes([0.05 + x*(0.86-w), 0.06 + y*(0.88-h), w, h])
        for label, ev in evokeds.items():
            e = ev.copy().pick([ch])
            ax.plot(e.times, e.data[0] * 1e6, label=label, linewidth=1)
        ax.axvline(0, color="k", linestyle="--", linewidth=.6)
        ax.set_title(ch, fontsize=8); ax.tick_params(labelsize=6)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles: fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Grand-average ERP: stimulation vs no stimulation — sensors in scalp layout")
    if show: plt.show()
    return fig


def compare_evokeds_compatible(evokeds, *args, **kwargs):
    if kwargs.get("axes") == "topo":
        picks = kwargs.get("picks")
        if picks is None and args:
            picks = args[0]
        return _erp_scalp_layout(evokeds, picks, show=kwargs.get("show", False))
    return _original_compare(evokeds, *args, **kwargs)


pipeline.mne.viz.plot_compare_evokeds = compare_evokeds_compatible
pipeline.main()
