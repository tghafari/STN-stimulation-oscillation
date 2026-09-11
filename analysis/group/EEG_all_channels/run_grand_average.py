#!/usr/bin/env python
"""MNE-compatible entry point for the all-channel group grand-average analysis.

The installed MNE version used by this project does not accept ``vlim`` in
AverageTFR.plot_topo(). This entry point applies the same compatibility fix as
the current subject-level all-channel A02_TFR.py: create the scalp-layout plot
first, then set the color limits on each image. All analysis logic and
parameters remain in G01_grand_average_report.py.
"""
from __future__ import annotations

import G01_grand_average_report as pipeline


def scalp_tfr_mne_compatible(tfr, vlim):
    fig = tfr.plot_topo(
        tmin=pipeline.PLOT_TMIN,
        tmax=pipeline.PLOT_TMAX,
        fmin=2,
        fmax=31.5,
        baseline=None,
        mode=None,
        cmap="RdBu_r",
        show=False,
    )
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        if vlim is not None and None not in vlim:
            for image in ax.images:
                image.set_clim(*vlim)
    return fig


pipeline.scalp_tfr = scalp_tfr_mne_compatible
pipeline.main()
