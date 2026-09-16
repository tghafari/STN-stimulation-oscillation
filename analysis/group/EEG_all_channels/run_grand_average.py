#!/usr/bin/env python
"""Entry point for the final all-channel grand-average report.

Adds stable ERP compatibility helpers plus montage-based scalp plotting. Scalp
panels use actual electrode montage coordinates; a dedicated colorbar area sits to
the right of all TFR panels and cannot cover a channel plot.
"""
import numpy as np
import matplotlib.pyplot as plt
import G04_complete_grand_average_report as pipeline
from scalp_layout import anatomical_tfr_scalp,anatomical_erp_scalp

def grand_arrays(items):
    if not items:raise ValueError('grand_arrays requires at least one subject-level object')
    times=np.asarray(items[0].times).copy();arrays=[]
    for item in items:
        if len(item.times)!=len(times) or not np.allclose(item.times,times):raise ValueError('Subject-level objects have inconsistent time axes')
        d=np.asarray(item.data);arrays.append(d[0] if d.ndim==2 and d.shape[0]==1 else np.squeeze(d))
    return np.mean(arrays,axis=0),times

def roi_erp_compatible(no,st,times,title):
    no=np.asarray(no).squeeze();st=np.asarray(st).squeeze();times=np.asarray(times).squeeze()
    if no.ndim!=1 or st.ndim!=1 or times.ndim!=1 or len(times)!=len(no) or len(times)!=len(st):raise ValueError(f'ROI ERP shape mismatch: times={times.shape}, no={no.shape}, stim={st.shape}')
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);ax.plot(times,no*1e6,label='No stimulation');ax.plot(times,st*1e6,label='Stimulation');ax.axvline(0,color='k',ls='--',lw=.7);ax.set_xlim(pipeline.ERP_TMIN,pipeline.ERP_TMAX);ax.set_xlabel('Time (s)');ax.set_ylabel('Amplitude (uV)');ax.set_title(title);ax.legend();return fig

def tfr_scalp_compatible(data,channels,info,title,vlim):return anatomical_tfr_scalp(data,channels,info,title,vlim,pipeline.FREQS)
def scalp_erp_compatible(group_erp,channels,info):return anatomical_erp_scalp(group_erp,channels,info,pipeline.ERP_TMIN,pipeline.ERP_TMAX)
pipeline.grand_arrays=grand_arrays;pipeline.roi_erp=roi_erp_compatible;pipeline.tfr_scalp=tfr_scalp_compatible;pipeline.scalp_erp=scalp_erp_compatible
if __name__=='__main__':pipeline.main()
