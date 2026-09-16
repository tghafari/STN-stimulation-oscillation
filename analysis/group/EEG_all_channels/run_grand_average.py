#!/usr/bin/env python
"""Run G04 with report-only detailed sensor views. Numerical analysis unchanged."""
import numpy as np
import matplotlib.pyplot as plt
import G04_complete_grand_average_report as pipeline

def baseline_percent_safe(x):
 data=np.asarray(x.data[0]);mask=(x.times>=pipeline.BASELINE[0])&(x.times<=pipeline.BASELINE[1]);base=data[:,mask].mean(axis=1,keepdims=True);return 100.*(data-base)/np.where(np.abs(base)<np.finfo(float).eps,1.,base)
pipeline.baseline_percent=baseline_percent_safe
_orig_evoked=pipeline.make_evoked;_erp_calls=[]
def make_evoked_collect(ep,ch):x=_orig_evoked(ep,ch);_erp_calls.append((ch,x));return x
pipeline.make_evoked=make_evoked_collect
_orig_scalp=pipeline.scalp;_last_scalp=None
def scalp_capture(data,info,title,v):
 global _last_scalp;_last_scalp=(data,info,title,v);return _orig_scalp(data,info,title,v)
pipeline.scalp=scalp_capture

def posterior_tfr(data,title,v):
 fig,axs=plt.subplots(2,4,figsize=(20,9));im=None
 for ax,ch in zip(axs.ravel(),pipeline.ROI8):
  if ch not in data:ax.axis('off');ax.set_title(ch+' unavailable');continue
  arr,t=data[ch];im=ax.imshow(arr,origin='lower',aspect='auto',extent=[t[0],t[-1],pipeline.FREQS[0],pipeline.FREQS[-1]],cmap='RdBu_r',vmin=v[0] if v else None,vmax=v[1] if v else None);ax.axvline(0,color='k',ls='--',lw=.5);ax.set_title(ch)
 fig.subplots_adjust(left=.06,right=.88,bottom=.08,top=.90,wspace=.32,hspace=.35);fig.suptitle(title+': eight posterior sensors')
 if im is not None:cax=fig.add_axes([.91,.18,.015,.62]);fig.colorbar(im,cax=cax,label='Value')
 return fig

def build_erp_groups():
 if not _erp_calls:return None
 n=len(_erp_calls)//2;halves=[_erp_calls[:n],_erp_calls[n:]];out={pipeline.CONDITIONS[0]:{},pipeline.CONDITIONS[1]:{}}
 for cond,calls in zip(pipeline.CONDITIONS,halves):
  bucket={}
  for ch,x in calls:bucket.setdefault(ch,[]).append(x)
  for ch,xs in bucket.items():out[cond][ch]=(np.mean([z.data[0] for z in xs],0),xs[0].times)
 return out

def _positions(chs,info):
 """Return positions for all plotted channels; fill montage gaps from standard_1020."""
 pos={}
 montage=info.get_montage()
 if montage is not None:pos.update(montage.get_positions().get('ch_pos',{}))
 missing=[c for c in chs if c not in pos]
 if missing:
  try:
   std=pipeline.mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
   for c in missing:
    if c in std:pos[c]=std[c]
  except Exception:pass
 # A channel without any known position cannot be anatomically placed. Exclude only
 # from this visualization; it remains fully included in every numerical analysis.
 return pos

def erp_scalp(erp,info):
 allchs=[c for c in erp['stim'] if c in erp['no-stim']];pos=_positions(allchs,info);chs=[c for c in allchs if c in pos]
 if not chs:raise RuntimeError('No EEG channel positions available for ERP scalp visualization')
 xy={c:np.asarray(pos[c])[:2] for c in chs};xs=np.array([xy[c][0] for c in chs]);ys=np.array([xy[c][1] for c in chs]);nr=11;nc=11;free={(r,c) for r in range(nr) for c in range(nc)};cells={};xmin,xmax=xs.min(),xs.max();ymin,ymax=ys.min(),ys.max();targets={c:((ymax-xy[c][1])/max(ymax-ymin,1e-12)*(nr-1),(xy[c][0]-xmin)/max(xmax-xmin,1e-12)*(nc-1)) for c in chs}
 for ch in sorted(chs,key=lambda c:targets[c]):tr,tc=targets[ch];q=min(free,key=lambda z:(z[0]-tr)**2+(z[1]-tc)**2);cells[ch]=q;free.remove(q)
 fig,axs=plt.subplots(nr,nc,figsize=(25,23));[a.axis('off') for a in axs.ravel()]
 for ch in chs:
  ax=axs[cells[ch]];ax.axis('on');st,t=erp['stim'][ch];no,_=erp['no-stim'][ch];ax.plot(t,no*1e6,lw=.8,label='No stimulation');ax.plot(t,st*1e6,lw=.8,label='Stimulation');ax.axvline(0,color='k',ls='--',lw=.4);ax.set_xlim(pipeline.ERP_TMIN,pipeline.ERP_TMAX);ax.set_title(ch,fontsize=8);ax.tick_params(labelsize=5)
 fig.subplots_adjust(left=.025,right=.94,bottom=.03,top=.94,wspace=.55,hspace=.65);fig.suptitle('Grand-average ERP: all sensors');return fig

def posterior_erp(erp):
 fig,axs=plt.subplots(2,4,figsize=(20,9),constrained_layout=True)
 for ax,ch in zip(axs.ravel(),pipeline.ROI8):
  if ch not in erp['stim'] or ch not in erp['no-stim']:ax.axis('off');continue
  st,t=erp['stim'][ch];no,_=erp['no-stim'][ch];ax.plot(t,no*1e6,label='No stimulation');ax.plot(t,st*1e6,label='Stimulation');ax.axvline(0,color='k',ls='--',lw=.5);ax.set_xlim(pipeline.ERP_TMIN,pipeline.ERP_TMAX);ax.set_title(ch)
 axs[0,0].legend();fig.suptitle('Grand-average ERP: eight posterior sensors');return fig

_BasePDF=pipeline.ParticipantPDF
class DetailPDF(_BasePDF):
 def __init__(self,*a,**k):super().__init__(*a,**k);self._erp_inserted=False
 def add_figure(self,fig,png,title,caption,section):
  global _last_scalp
  if section=='ERP' and title.startswith('ERP mean') and not self._erp_inserted:
   e=build_erp_groups()
   if e:
    info=_erp_calls[0][1].info
    super().add_figure(erp_scalp(e,info),str(png).replace('ERP_mean','ERP_all_sensors_scalp'),'ERP: all sensors scalp layout','Grand-average ERP per sensor. Missing montage coordinates are filled from MNE standard_1020 for plotting only; numerical data are unchanged.','ERP')
    super().add_figure(posterior_erp(e),str(png).replace('ERP_mean','ERP_8posterior'),'ERP: eight posterior sensors','PO3, POz, PO4, O1, Oz, O2, PO7, PO8 in a 2 x 4 layout.','ERP')
   self._erp_inserted=True
  super().add_figure(fig,png,title,caption,section)
  if section.startswith('TFR - ') and title.endswith(' scalp') and _last_scalp is not None:
   data,info,stitle,v=_last_scalp;super().add_figure(posterior_tfr(data,stitle,v),str(png).replace('_scalp.png','_8posterior.png'),stitle+': eight posterior sensors','Eight posterior sensors in a 2 x 4 layout using the exact same already-computed group arrays and color scale.',section)
pipeline.ParticipantPDF=DetailPDF
if __name__=='__main__':pipeline.main()
