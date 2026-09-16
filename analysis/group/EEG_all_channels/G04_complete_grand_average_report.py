#!/usr/bin/env python
"""Single authoritative complete all-channel grand-average report.

FINAL TFR CONTRAST ORDER
1. Stim and No-stim TFR power separately for each participant and sensor.
2. From ORIGINAL UNBASELINED power for that same participant/sensor:
   Difference = Pstim - Pnostim
   Ratio = (Pstim-Pnostim)/(Pstim+Pnostim)
3. Average participant contrasts independently for each sensor.
4. LAST: average channel-level group contrasts over ROI3 or ROI8.

Percent baseline (-0.3,-0.1 s) is used ONLY for descriptive Stim/No-stim TFRs.
It never enters Difference or Ratio.
"""
import argparse,json,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import mne
HERE=Path(__file__).resolve().parent;ANALYSIS_DIR=HERE.parents[1]
for p in (ANALYSIS_DIR/'subject'/'EEG_all_channels',ANALYSIS_DIR/'utils'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from pipeline_config import CONDITIONS,resolve_project_root,stage_path
from pdf_report import ParticipantPDF
ROI3=('PO3','POz','PO4');ROI8=('PO3','POz','PO4','O1','Oz','O2','PO7','PO8')
ERP_BASELINE=(-.1,0.);ERP_LP=30.;ERP_TMIN=-.1;ERP_TMAX=1.;BASELINE=(-.3,-.1);FREQS=np.arange(2.,32.,.5);N_CYCLES=FREQS/2.;TIME_BANDWIDTH=2.;DECIM=2;ROBUST=98.
def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subjects',nargs='+',required=True);p.add_argument('--session',default='01');p.add_argument('--task',default='SpAtt');p.add_argument('--run',default='01');p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);p.add_argument('--n-jobs',type=int,default=4);return p.parse_args()
def load_epochs(root,s,a):
 out={}
 for c in CONDITIONS:
  f=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo');e=mne.read_epochs(f,preload=True,verbose=False);keys=[k for k in ('cue_onset_right','cue_onset_left') if k in e.event_id];out[c]=e[keys] if keys else e
 return out
def good_channels(x):
 st,no=x['stim'],x['no-stim'];return[ch for ch in st.copy().pick('eeg').ch_names if ch in no.ch_names and ch not in st.info['bads'] and ch not in no.info['bads']]
def make_tfr(ep,ch,j):return ep.copy().pick([ch]).compute_tfr(method='multitaper',freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,zero_mean=True,return_itc=False,average=True,decim=DECIM,n_jobs=j,verbose=False)
def make_evoked(ep,ch):
 e=ep.copy().pick([ch]).average();e.filter(None,ERP_LP,verbose=False);e.apply_baseline(ERP_BASELINE);e.crop(ERP_TMIN,ERP_TMAX);return e
def baseline_percent(x):
 mask=(x.times>=BASELINE[0])&(x.times<=BASELINE[1]);base=x.data[0,:,mask].mean(-1,keepdims=True);return 100*(x.data[0]-base)/np.where(np.abs(base)<np.finfo(float).eps,1,base)
def ratio(st,no):return(st-no)/(st+no+np.finfo(float).eps)
def scale(vals):
 z=np.concatenate([np.asarray(v)[np.isfinite(v)] for v in vals if np.asarray(v).size]);m=float(np.percentile(np.abs(z),ROBUST));return(-m,m) if m>0 else None
def roi(d,chs):
 av=[c for c in chs if c in d];return(np.mean([d[c][0] for c in av],0),av) if av else(None,[])
def scalp(data,info,title,v):
 pos=info.get_montage().get_positions()['ch_pos'];chs=list(data);xy={c:np.asarray(pos[c])[:2] for c in chs};xs=np.array([xy[c][0] for c in chs]);ys=np.array([xy[c][1] for c in chs]);nr=11;nc=11;free={(r,c) for r in range(nr) for c in range(nc)};cells={};xmin,xmax=xs.min(),xs.max();ymin,ymax=ys.min(),ys.max()
 targets={c:((ymax-xy[c][1])/max(ymax-ymin,1e-12)*(nr-1),(xy[c][0]-xmin)/max(xmax-xmin,1e-12)*(nc-1)) for c in chs}
 for ch in sorted(chs,key=lambda c:targets[c]):tr,tc=targets[ch];cell=min(free,key=lambda q:(q[0]-tr)**2+(q[1]-tc)**2);cells[ch]=cell;free.remove(cell)
 fig,axs=plt.subplots(nr,nc,figsize=(25,23));[a.axis('off') for a in axs.ravel()]
 for ch in chs:
  ax=axs[cells[ch]];ax.axis('on');a,t=data[ch];ax.imshow(a,origin='lower',aspect='auto',extent=[t[0],t[-1],FREQS[0],FREQS[-1]],cmap='RdBu_r',vmin=v[0] if v else None,vmax=v[1] if v else None);ax.axvline(0,color='k',ls='--',lw=.4);ax.set_title(ch,fontsize=8);ax.tick_params(labelsize=5)
 fig.subplots_adjust(left=.025,right=.89,bottom=.03,top=.94,wspace=.55,hspace=.65);fig.suptitle(title)
 if v:
  from matplotlib.cm import ScalarMappable
  from matplotlib.colors import Normalize
  cax=fig.add_axes([.925,.2,.015,.6]);sm=ScalarMappable(norm=Normalize(*v),cmap='RdBu_r');sm.set_array([]);fig.colorbar(sm,cax=cax,label='Value')
 return fig
def roi_fig(a,t,title,v):
 f,ax=plt.subplots(figsize=(10,5),constrained_layout=True);im=ax.imshow(a,origin='lower',aspect='auto',extent=[t[0],t[-1],FREQS[0],FREQS[-1]],cmap='RdBu_r',vmin=v[0] if v else None,vmax=v[1] if v else None);ax.axvline(0,color='k',ls='--');ax.set_title(title);ax.set_xlabel('Time (s)');ax.set_ylabel('Frequency (Hz)');f.colorbar(im,ax=ax);return f
def main():
 a=parse_args();subs=[s.removeprefix('sub-') for s in a.subjects];root=resolve_project_root(a.platform,a.project_root);out=root/'derivatives'/'reports'/'group'/'EEG_all_channels_complete_grand_average';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True);deriv=root/'data'/'BIDS'/'derivatives'/'group'/'EEG_all_channels_complete_grand_average';deriv.mkdir(parents=True,exist_ok=True);rid='complete_grand_average_'+'_'.join(subs);report=ParticipantPDF(str(out),rid)
 ep={s:load_epochs(root,s,a) for s in subs};good={s:good_channels(ep[s]) for s in subs};chs=[]
 for s in subs:
  for ch in good[s]:
   if ch not in chs:chs.append(ch)
 by={ch:[s for s in subs if ch in good[s]] for ch in chs};info=ep[subs[0]]['stim'].copy().pick('eeg').info;report.add_text('Subjects included',', '.join('sub-'+s for s in subs),'Overview');report.add_text('Subjects by channel','\n'.join(f"{ch} (n={len(by[ch])}): {', '.join(by[ch])}" for ch in chs),'Overview')
 logic='Power is computed separately for Stim and No-stim for every participant/sensor. Difference and Ratio are then computed for that participant/sensor from ORIGINAL UNBASELINED power. Participants are averaged independently per sensor. Only as the final step are sensor-level group contrasts averaged over ROI3 or ROI8. Difference baseline: NONE. Ratio baseline: NONE. Percent baseline -0.3 to -0.1 s applies ONLY to descriptive Stim and No-stim plots.';report.add_text('Final analysis logic',logic,'Overview')
 # ERP ROI summaries retained; ERP baseline is independent of TFR contrasts.
 erp={c:{} for c in CONDITIONS}
 for c in CONDITIONS:
  for ch in chs:
   items=[make_evoked(ep[s][c],ch) for s in by[ch]];erp[c][ch]=(np.mean([x.data[0] for x in items],0),items[0].times)
 for R,name in ((ROI3,'PO3/POz/PO4'),(ROI8,'8 posterior channels')):
  av=[c for c in R if c in erp['stim']];no=np.mean([erp['no-stim'][c][0] for c in av],0);st=np.mean([erp['stim'][c][0] for c in av],0);f,ax=plt.subplots(figsize=(10,5));ax.plot(erp['stim'][av[0]][1],no*1e6,label='No stimulation');ax.plot(erp['stim'][av[0]][1],st*1e6,label='Stimulation');ax.legend();ax.set_title('ERP mean '+name);report.add_figure(f,str(figs/f'ERP_mean{len(av)}.png'),'ERP mean '+name,'ERP baseline -0.1 to 0 s.','ERP')
 tfr={s:{c:{} for c in CONDITIONS} for s in subs}
 for s in subs:
  for c in CONDITIONS:
   print(f'Computing TFR sub-{s} {c}')
   for ch in good[s]:tfr[s][c][ch]=make_tfr(ep[s][c],ch,a.n_jobs)
 first=next(tfr[s]['stim'][c] for s in subs for c in tfr[s]['stim']);times=first.times;display={'stim':{},'no-stim':{}};diff={};rat={}
 for ch in chs:
  eligible=[s for s in subs if ch in tfr[s]['stim'] and ch in tfr[s]['no-stim']]
  display['stim'][ch]=(np.mean([baseline_percent(tfr[s]['stim'][ch]) for s in eligible],0),eligible);display['no-stim'][ch]=(np.mean([baseline_percent(tfr[s]['no-stim'][ch]) for s in eligible],0),eligible)
  # These two lines are the authoritative contrast definitions. NO baseline arrays are referenced.
  d=[tfr[s]['stim'][ch].data[0]-tfr[s]['no-stim'][ch].data[0] for s in eligible];r=[ratio(tfr[s]['stim'][ch].data[0],tfr[s]['no-stim'][ch].data[0]) for s in eligible];diff[ch]=(np.mean(d,0),eligible);rat[ch]=(np.mean(r,0),eligible)
 results=[('no-stim','No stimulation',display['no-stim'],True),('stim','Stimulation',display['stim'],True),('difference','Stimulation - no stimulation',diff,False),('ratio','(Stimulation - no stimulation)/(Stimulation + no stimulation)',rat,False)]
 for key,title,d,bl in results:
  data={c:(x[0],times) for c,x in d.items()};v=scale([x[0] for x in d.values()]);detail=('Percent baseline -0.3 to -0.1 s; descriptive condition only.' if bl else 'NO BASELINE CORRECTION. Contrast formed from raw participant/sensor power before participant averaging.');report.add_text('Analysis details',detail,'TFR - '+title);report.add_figure(scalp(data,info,title,v),str(figs/f'TFR_{key}_scalp.png'),title+' scalp','Montage-informed unique-cell layout; panels cannot overlap; colorbar is outside the grid. '+detail,'TFR - '+title)
  for R,name,n in ((ROI3,'PO3/POz/PO4',3),(ROI8,'8 posterior channels',8)):
   x,av=roi(d,R)
   if x is not None:report.add_figure(roi_fig(x,times,title+' mean '+name,scale([x])),str(figs/f'TFR_{key}_mean{n}.png'),title+' mean '+name,'LAST STEP: mean of channel-level group results across '+', '.join(av)+'. '+detail,'TFR - '+title)
 manuscript=f'Cue-locked TFR power was estimated separately for each participant and retained EEG sensor in stimulation and no-stimulation conditions. For each participant and sensor, the absolute difference P_stim-P_no-stim and normalized difference (P_stim-P_no-stim)/(P_stim+P_no-stim) were calculated directly from original unbaselined power. Participant-level sensor contrasts were then averaged across eligible participants independently for each sensor. Posterior ROI summaries were calculated only after sensor-level group averaging, across PO3, POz and PO4 (ROI3) or PO3, POz, PO4, O1, Oz, O2, PO7 and PO8 (ROI8). No baseline correction was applied to either difference or ratio. Percent baseline correction from -0.3 to -0.1 s was used only for descriptive Stim and No-stim plots.';report.add_text('Analysis report - manuscript style',manuscript,'Analysis');(deriv/f'{rid}_analysis.json').write_text(json.dumps({'subjects':subs,'subjects_by_channel':by,'contrast_order':'raw power participant/sensor -> contrast participant/sensor -> mean participants/sensor -> mean sensors','difference_baseline':'none','ratio_baseline':'none','stim_nostim_baseline':'percent -0.3 to -0.1 s'},indent=2)+'\n');print(report.pdf_fname)
if __name__=='__main__':main()
