#!/usr/bin/env python
"""Compatibility/synchronization entry point for all-channel group grand averages."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import G01_grand_average_report as pipeline
pipeline.POSTERIOR=("PO3","POz","PO4","O1","Oz","O2","PO7","PO8");pipeline.ERP_BASELINE=(-0.1,0.0);pipeline.ERP_LP_HZ=30.;pipeline.ERP_TMIN=-0.1;pipeline.ERP_TMAX=0.5;pipeline.BASELINE=(-0.3,-0.1);pipeline.FREQS=np.arange(2.,31.,1.);pipeline.N_CYCLES=pipeline.FREQS/2.;pipeline.TIME_BANDWIDTH=2.;pipeline.DECIM=2;pipeline.PLOT_TMIN=-0.3;pipeline.PLOT_TMAX=1.4;pipeline.ROBUST_PERCENTILE=98.;pipeline.get_difference_baseline_choice=lambda:False

def scalp_tfr_mne_compatible(tfr,vlim):
 fig=tfr.plot_topo(tmin=pipeline.PLOT_TMIN,tmax=pipeline.PLOT_TMAX,fmin=2,fmax=30,baseline=None,mode=None,cmap='RdBu_r',show=False);fig.patch.set_facecolor('white')
 for ax in fig.axes:
  ax.set_facecolor('white')
  if vlim is not None and None not in vlim:
   for image in ax.images:image.set_clim(*vlim)
 if vlim is not None and None not in vlim:
  sm=plt.cm.ScalarMappable(cmap='RdBu_r',norm=plt.Normalize(*vlim));sm.set_array([]);cax=fig.add_axes([.92,.18,.018,.64]);cb=fig.colorbar(sm,cax=cax);cb.set_ticks(np.linspace(vlim[0],vlim[1],5));cb.set_label('TFR power')
 return fig
pipeline.scalp_tfr=scalp_tfr_mne_compatible

# Eight ROI channels: force a readable 2 x 4 layout instead of one long row.
def posterior_tfr_two_rows(tfr,posterior,title,vlim):
 n=len(posterior);ncol=4;nrow=int(np.ceil(n/ncol));fig,axes=plt.subplots(nrow,ncol,figsize=(16,4.5*nrow),constrained_layout=True);axes=np.atleast_1d(axes).ravel()
 for ax in axes[n:]:ax.axis('off')
 for ax,ch in zip(axes,posterior):
  tfr.plot(picks=ch,tmin=pipeline.PLOT_TMIN,tmax=pipeline.PLOT_TMAX,fmin=2,fmax=30,baseline=None,mode=None,axes=ax,show=False,colorbar=True,vlim=vlim,cmap='RdBu_r');ax.set_title(ch)
 fig.suptitle(title);return fig
pipeline.posterior_tfr=posterior_tfr_two_rows

_original_compare=pipeline.mne.viz.plot_compare_evokeds

def _erp_scalp_layout(evokeds,picks,show=False,**kwargs):
 """Non-overlapping scalp-like ERP layout using collision-aware grid cells."""
 first=next(iter(evokeds.values()));picks=[ch for ch in picks if ch in first.ch_names];montage=first.get_montage();ch_pos=montage.get_positions().get('ch_pos',{}) if montage is not None else {};coords={}
 for ch in picks:
  if ch in ch_pos:coords[ch]=np.asarray(ch_pos[ch],float)[:2]
 # If montage positions are incomplete, use a regular grid for all channels.
 if len(coords)!=len(picks):
  ncol=int(np.ceil(np.sqrt(len(picks))));fig,axes=plt.subplots(int(np.ceil(len(picks)/ncol)),ncol,figsize=(16,12),constrained_layout=True);axes=np.atleast_1d(axes).ravel()
  for ax in axes[len(picks):]:ax.axis('off')
  assigned=dict(zip(picks,axes))
 else:
  xy=np.vstack([coords[ch] for ch in picks]);xmin,ymin=xy.min(axis=0);xmax,ymax=xy.max(axis=0);dx=max(xmax-xmin,1e-9);dy=max(ymax-ymin,1e-9)
  # Quantize electrode positions to a sufficiently large grid. Each sensor gets a unique cell,
  # preserving approximate scalp position while guaranteeing that axes never overlap.
  ncol=max(7,int(np.ceil(np.sqrt(len(picks))*1.7)));nrow=ncol
  desired={ch:(int(round((coords[ch][0]-xmin)/dx*(ncol-1))),int(round((coords[ch][1]-ymin)/dy*(nrow-1)))) for ch in picks};occupied=set();cells={}
  for ch in picks:
   cx,cy=desired[ch];candidates=sorted(((abs(x-cx)+abs(y-cy),x,y) for y in range(nrow) for x in range(ncol) if (x,y) not in occupied));_,x,y=candidates[0];cells[ch]=(x,y);occupied.add((x,y))
  fig=plt.figure(figsize=(18,14),facecolor='white');gs=fig.add_gridspec(nrow,ncol,left=.04,right=.96,bottom=.05,top=.93,wspace=.55,hspace=.75);assigned={ch:fig.add_subplot(gs[nrow-1-cells[ch][1],cells[ch][0]]) for ch in picks}
 for ch,ax in assigned.items():
  for label,ev in evokeds.items():
   e=ev.copy().pick([ch]);ax.plot(e.times,e.data[0]*1e6,label=label,linewidth=1)
  ax.axvline(0,color='k',linestyle='--',linewidth=.6);ax.set_title(ch,fontsize=8);ax.tick_params(labelsize=6);ax.margins(x=.02)
 handles,labels=next(iter(assigned.values())).get_legend_handles_labels()
 if handles:fig.legend(handles,labels,loc='upper right')
 fig.suptitle('Grand-average ERP: stimulation vs no stimulation — sensors in scalp layout');
 if show:plt.show()
 return fig

def compare_evokeds_compatible(evokeds,*args,**kwargs):
 if kwargs.get('axes')=='topo':
  picks=kwargs.get('picks');picks=picks if picks is not None else (args[0] if args else next(iter(evokeds.values())).ch_names);return _erp_scalp_layout(evokeds,picks,show=kwargs.get('show',False))
 return _original_compare(evokeds,*args,**kwargs)
pipeline.mne.viz.plot_compare_evokeds=compare_evokeds_compatible
pipeline.main()
