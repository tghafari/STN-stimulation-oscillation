#!/usr/bin/env python
"""Entry point for the complete all-channel grand-average report.

Applies a compatibility replacement for G04's ERP scalp-layout function because
MNE plot_compare_evokeds(..., axes='topo') fails in the project's installed MNE
version with an empty internal time vector. The replacement plots the same grand
ERP data directly with Matplotlib and never changes the analysis values.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import G04_complete_grand_average_report as pipeline


def scalp_erp_compatible(compare, channels):
    """Non-overlapping scalp-like ERP layout using electrode positions."""
    first = next(iter(compare.values()))
    channels = [ch for ch in channels if ch in first.ch_names]
    montage = first.get_montage()
    ch_pos = montage.get_positions().get('ch_pos', {}) if montage is not None else {}
    # Preserve left/right and anterior/posterior ordering, but assign every sensor
    # to a unique grid cell so nearby electrodes can never overlap.
    coords = {}
    for i, ch in enumerate(channels):
        if ch in ch_pos:
            xyz = np.asarray(ch_pos[ch], float); coords[ch] = (float(xyz[0]), float(xyz[1]))
        else:
            coords[ch] = (float(i % 9), float(-(i // 9)))
    xs = np.array([coords[ch][0] for ch in channels]); ys = np.array([coords[ch][1] for ch in channels])
    # Nine columns gives enough horizontal separation for a typical 64-channel cap.
    ncols = 9
    x_order = {ch: rank for rank, ch in enumerate(sorted(channels, key=lambda c: coords[c][0]))}
    y_sorted = sorted(channels, key=lambda c: coords[c][1], reverse=True)
    # Greedy unique-cell assignment: nearest scalp-like row/column, then nearest free cell.
    nrows = int(np.ceil(len(channels) / ncols))
    free = {(r, c) for r in range(nrows) for c in range(ncols)}; cells = {}
    xmin,xmax=(xs.min(),xs.max()) if len(xs)>1 else (0.,1.); ymin,ymax=(ys.min(),ys.max()) if len(ys)>1 else (0.,1.)
    for ch in y_sorted:
        x,y=coords[ch]; tc=int(round((x-xmin)/max(xmax-xmin,1e-9)*(ncols-1))); tr=int(round((ymax-y)/max(ymax-ymin,1e-9)*(nrows-1)))
        cell=min(free,key=lambda rc:(rc[0]-tr)**2+(rc[1]-tc)**2);cells[ch]=cell;free.remove(cell)
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 3.0*nrows), squeeze=False)
    for ax in axes.ravel(): ax.axis('off')
    for ch in channels:
        r,c=cells[ch];ax=axes[r,c];ax.axis('on')
        for label, ev in compare.items():
            e=ev.copy().pick([ch]);ax.plot(e.times,e.data[0]*1e6,label=label,linewidth=1)
        ax.axvline(0,color='k',linestyle='--',linewidth=.6);ax.set_title(ch,fontsize=9);ax.tick_params(labelsize=6);ax.set_xlim(pipeline.ERP_TMIN,pipeline.ERP_TMAX)
    handles,labels=next(ax for ax in axes.ravel() if ax.has_data()).get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc='upper right')
    fig.suptitle('Grand-average ERP: stimulation vs no stimulation - scalp layout',fontsize=16)
    fig.subplots_adjust(left=.04,right=.96,bottom=.04,top=.94,wspace=.55,hspace=.75)
    return fig

pipeline.scalp_erp = scalp_erp_compatible
pipeline.main()
