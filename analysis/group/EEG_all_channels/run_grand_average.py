#!/usr/bin/env python
"""Compatibility/synchronization entry point for all-channel group grand averages.

This wrapper customizes the all-channel scalp TFR and ERP-topo views without globally
monkey-patching MNE's plot_compare_evokeds. Avoiding that global patch is important in
interactive/IPython sessions, where re-running this script could otherwise save the
previous wrapper as the 'original' function and recurse indefinitely.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import G01_grand_average_report as pipeline

pipeline.POSTERIOR=("PO3","POz","PO4","O1","Oz","O2","PO7","PO8")
pipeline.ERP_BASELINE=(-0.1,0.0);pipeline.ERP_LP_HZ=30.;pipeline.ERP_TMIN=-0.1;pipeline.ERP_TMAX=0.5
pipeline.BASELINE=(-0.3,-0.1);pipeline.FREQS=np.arange(2.,31.,1.);pipeline.N_CYCLES=pipeline.FREQS/2.
pipeline.TIME_BANDWIDTH=2.;pipeline.DECIM=2;pipeline.PLOT_TMIN=-0.3;pipeline.PLOT_TMAX=1.4
pipeline.ROBUST_PERCENTILE=98.;pipeline.get_difference_baseline_choice=lambda:False


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


def posterior_tfr_two_rows(tfr,posterior,title,vlim):
 n=len(posterior);ncol=4;nrow=int(np.ceil(n/ncol));fig,axes=plt.subplots(nrow,ncol,figsize=(16,4.5*nrow),constrained_layout=True);axes=np.atleast_1d(axes).ravel()
 for ax in axes[n:]:ax.axis('off')
 for ax,ch in zip(axes,posterior):
  tfr.plot(picks=ch,tmin=pipeline.PLOT_TMIN,tmax=pipeline.PLOT_TMAX,fmin=2,fmax=30,baseline=None,mode=None,axes=ax,show=False,colorbar=True,vlim=vlim,cmap='RdBu_r');ax.set_title(ch)
 fig.suptitle(title);return fig
pipeline.posterior_tfr=posterior_tfr_two_rows


def _erp_scalp_layout(evokeds,picks,show=False,**kwargs):
 """Readable, non-overlapping scalp-like ERP layout without patching MNE globally."""
 first=next(iter(evokeds.values()));picks=[ch for ch in picks if ch in first.ch_names];montage=first.get_montage();ch_pos=montage.get_positions().get('ch_pos',{}) if montage is not None else {};coords={}
 for ch in picks:
  if ch in ch_pos:coords[ch]=np.asarray(ch_pos[ch],float)[:2]
 if len(coords)!=len(picks):
  ncol=int(np.ceil(np.sqrt(len(picks))));nrow=int(np.ceil(len(picks)/ncol));fig,axes=plt.subplots(nrow,ncol,figsize=(24,18),constrained_layout=True);axes=np.atleast_1d(axes).ravel()
  for ax in axes[len(picks):]:ax.axis('off')
  assigned=dict(zip(picks,axes))
 else:
  xy=np.vstack([coords[ch] for ch in picks]);xmin,ymin=xy.min(axis=0);xmax,ymax=xy.max(axis=0);dx=max(xmax-xmin,1e-9);dy=max(ymax-ymin,1e-9);ncol=max(7,int(np.ceil(np.sqrt(len(picks))*1.7)));nrow=ncol
  desired={ch:(int(round((coords[ch][0]-xmin)/dx*(ncol-1))),int(round((coords[ch][1]-ymin)/dy*(nrow-1)))) for ch in picks};occupied=set();cells={}
  for ch in picks:
   cx,cy=desired[ch];candidates=sorted((abs(x-cx)+abs(y-cy),x,y) for y in range(nrow) for x in range(ncol) if (x,y) not in occupied);_,x,y=candidates[0];cells[ch]=(x,y);occupied.add((x,y))
  fig=plt.figure(figsize=(28,22),facecolor='white');gs=fig.add_gridspec(nrow,ncol,left=.035,right=.965,bottom=.04,top=.94,wspace=.35,hspace=.48);assigned={ch:fig.add_subplot(gs[nrow-1-cells[ch][1],cells[ch][0]]) for ch in picks}
 for ch,ax in assigned.items():
  for label,ev in evokeds.items():
   e=ev.copy().pick([ch]);ax.plot(e.times,e.data[0]*1e6,label=label,linewidth=1.2)
  ax.axvline(0,color='k',linestyle='--',linewidth=.6);ax.set_title(ch,fontsize=9);ax.tick_params(labelsize=7);ax.margins(x=.02)
 handles,labels=next(iter(assigned.values())).get_legend_handles_labels()
 if handles:fig.legend(handles,labels,loc='upper right')
 fig.suptitle('Grand-average ERP: stimulation vs no stimulation — sensors in scalp layout')
 if show:plt.show()
 return fig

# Patch only the pipeline's scalp ERP helper. Ordinary calls to
# mne.viz.plot_compare_evokeds (e.g. PO3/POz/PO4 panels) remain the genuine MNE
# function, so repeated notebook/script execution cannot create wrapper recursion.
def scalp_erp_compatible(compare,channels):
 return _erp_scalp_layout(compare,channels,show=False)

# G01 creates its all-channel ERP directly inside main rather than through a helper.
# Give it a small proxy namespace whose plot_compare_evokeds intercepts ONLY axes='topo'
# and delegates all other calls to the genuine MNE function captured from the module.
_real_compare=pipeline.mne.viz.plot_compare_evokeds
class _VizProxy:
 def __getattr__(self,name):return getattr(pipeline.mne.viz,name)
 def plot_compare_evokeds(self,evokeds,*args,**kwargs):
  if kwargs.get('axes')=='topo':
   picks=kwargs.get('picks') or next(iter(evokeds.values())).ch_names
   return _erp_scalp_layout(evokeds,picks,show=kwargs.get('show',False))
  return _real_compare(evokeds,*args,**kwargs)
class _MNEProxy:
 def __getattr__(self,name):return getattr(pipeline.mne,name)

# Do not assign into mne.viz itself. Replace only G01's module-level mne reference
# with a proxy whose viz attribute is isolated from the real global MNE module.
_mne_proxy=_MNEProxy();_mne_proxy.viz=_VizProxy();pipeline.mne=_mne_proxy
pipeline.main()
