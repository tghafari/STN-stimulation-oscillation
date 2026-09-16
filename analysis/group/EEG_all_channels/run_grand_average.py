#!/usr/bin/env python
"""Run G04 with a safe descriptive TFR percent-baseline helper.

The patch affects ONLY descriptive Stim/No-stim plots. Difference and ratio remain
computed inside G04 from original unbaselined participant/sensor TFR power.
"""
import numpy as np
import G04_complete_grand_average_report as pipeline

def baseline_percent_safe(x):
    """Percent baseline along the TIME axis of single-channel AverageTFR data."""
    data=np.asarray(x.data[0])  # (n_freqs, n_times)
    if data.ndim!=2:raise ValueError(f'Expected (freq,time) TFR data, got {data.shape}')
    mask=(x.times>=pipeline.BASELINE[0])&(x.times<=pipeline.BASELINE[1])
    if mask.sum()==0:raise ValueError(f'No TFR samples in baseline {pipeline.BASELINE}')
    # Avoid NumPy mixed advanced/basic indexing, which can transpose dimensions.
    base=data[:,mask].mean(axis=1,keepdims=True)  # (n_freqs,1)
    denom=np.where(np.abs(base)<np.finfo(float).eps,1.0,base)
    return 100.0*(data-base)/denom

pipeline.baseline_percent=baseline_percent_safe
if __name__=='__main__':pipeline.main()
