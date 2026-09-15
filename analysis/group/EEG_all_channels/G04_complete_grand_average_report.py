#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G04: complete cue-locked all-channel EEG grand-average report.

This is the comprehensive group report. It uses final cleaned all-channel cue epochs,
never concatenates epochs across participants, and averages each participant first.
For every individual EEG sensor, only participants retaining that sensor in BOTH
stimulation conditions contribute to its group average. Missing sensors are not
interpolated at group level.

Report order
------------
1. Group overview: subjects included and exact subjects contributing to each sensor.
2. ERP: all-channel non-overlapping scalp layout; 8 posterior sensors in 2x4 form;
   8-sensor posterior ROI mean; PO3/POz/PO4 ROI mean.
3. TFR no stimulation: same four views.
4. TFR stimulation: same four views.
5. TFR stimulation - no stimulation: same four views.
6. TFR (stimulation - no stimulation)/(stimulation + no stimulation): same views.
7. Manuscript-style analysis report.

ROI means are computed within participant first from available ROI sensors, then
participant ROI averages are grand-averaged. Thus participants contribute equally.

TFR parameters match the existing posterior grand-average analysis:
2-31.5 Hz in 0.5-Hz steps, multitaper, n_cycles=f/2, time-bandwidth=2,
decim=2, FFT=True, zero_mean=True, ITC=False, trial average=True.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np

HERE=Path(__file__).resolve().parent
ANALYSIS_DIR=HERE.parents[1]
for p in (ANALYSIS_DIR/'subject'/'EEG_all_channels', ANALYSIS_DIR/'utils'):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from pipeline_config import CONDITIONS, resolve_project_root, stage_path
from pdf_report import ParticipantPDF

ROI8=("PO3","POz","PO4","O1","Oz","O2","PO7","PO8")
ROI3=("PO3","POz","PO4")
ERP_BASELINE=(-0.1,0.0); ERP_LP=30.; ERP_TMIN=-0.1; ERP_TMAX=1.0
BASELINE=(-0.3,-0.1); FREQS=np.arange(2.,32.,0.5); N_CYCLES=FREQS/2.
TIME_BANDWIDTH=2.; DECIM=2; TFR_TMIN=-0.5; TFR_TMAX=1.5; ROBUST=98.

def args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--subjects',nargs='+',required=True); p.add_argument('--session',default='01'); p.add_argument('--task',default='SpAtt'); p.add_argument('--run',default='01'); p.add_argument('--platform',choices=['mac','bluebear'],default='mac'); p.add_argument('--project-root',default=None); p.add_argument('--n-jobs',type=int,default=4); return p.parse_args()
def diff_baseline_choice():
 while True:
  x=input('\nApply percent baseline (-0.3,-0.1 s) before calculating stim - no-stim TFR? (y/n): ').strip().lower()
  if x in {'y','yes'}: return True
  if x in {'n','no'}: return False
  print("Please enter 'y' or 'n'.")
def load_epochs(root,s,a):
 out={}
 for c in CONDITIONS:
  p=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo')
  if not p.exists(): raise FileNotFoundError(f'Missing final cleaned epochs: {p}')
  ep=mne.read_epochs(p,preload=True); keep=[k for k in ('cue_onset_right','cue_onset_left') if k in ep.event_id]; out[c]=ep[keep] if keep else ep
 return out
def good_channels(pair):
 s=pair['stim']; n=pair['no-stim']; return [ch for ch in s.copy().pick('eeg').ch_names if ch in n.ch_names and ch not in s.info['bads'] and ch not in n.info['bads']]
def evoked(ep,picks):
 x=ep.copy().pick(picks).average(); x.filter(None,ERP_LP); x.apply_baseline(ERP_BASELINE); x.crop(ERP_TMIN,ERP_TMAX); return x
def tfr(ep,picks,jobs):
 return ep.copy().pick(picks).compute_tfr(method='multitaper',freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,zero_mean=True,return_itc=False,average=True,decim=DECIM,n_jobs=jobs)
def template_info(epochs,channels):
 for pair in epochs.values():
  info=pair['stim'].copy().pick('eeg').info
  if all(ch in info.ch_names for ch in channels): return mne.pick_info(info,[info.ch_names.index(ch) for ch in channels],copy=True)
 first=next(iter(epochs.values()))['stim']; info=mne.create_info(channels,first.info['sfreq'],'eeg')
 try: info.set_montage(first.get_montage(),on_missing='ignore')
 except Exception: pass
 return info
def channel_grand(subject_objects,by_ch,info,channels,condition,is_tfr=False):
 rows=[]
 for ch in channels:
  rows.append(np.mean([subject_objects[s][condition].copy().pick([ch]).data[0] for s in by_ch[ch]],axis=0))
 first=subject_objects[by_ch[channels[0]][0]][condition]
 if is_tfr: return mne.time_frequency.AverageTFRArray(info.copy(),np.asarray(rows),first.times,first.freqs,nave=len(subject_objects),comment=f'grand {condition}')
 return mne.EvokedArray(np.asarray(rows),info.copy(),tmin=float(first.times[0]),nave=len(subject_objects),comment=f'grand {condition}')
def roi_evoked(subject_evokeds,subjects,roi,condition):
 xs=[]; used=[]; sensors={}
 for s in subjects:
  available=[ch for ch in roi if ch in subject_evokeds[s]['stim'].ch_names and ch in subject_evokeds[s]['no-stim'].ch_names]
  if not available: continue
  x=subject_evokeds[s][condition].copy().pick(available); data=x.data.mean(axis=0,keepdims=True); info=mne.create_info(['ROI_mean'],x.info['sfreq'],'eeg'); xs.append(mne.EvokedArray(data,info,tmin=x.times[0])); used.append(s); sensors[s]=available
 if not xs:return None,[],{}
 g=xs[0].copy(); g.data=np.mean([x.data for x in xs],axis=0); g.nave=len(xs); return g,used,sensors
def roi_tfr(subject_tfrs,subjects,roi,result,diff_bl):
 xs=[]; used=[]; sensors={}
 for s in subjects:
  available=[ch for ch in roi if ch in subject_tfrs[s]['stim'].ch_names and ch in subject_tfrs[s]['no-stim'].ch_names]
  if not available: continue
  st=subject_tfrs[s]['stim'].copy().pick(available); no=subject_tfrs[s]['no-stim'].copy().pick(available)
  if result=='stim': x=st.apply_baseline(BASELINE,mode='percent')
  elif result=='no-stim': x=no.apply_baseline(BASELINE,mode='percent')
  elif result=='difference':
   if diff_bl: st.apply_baseline(BASELINE,mode='percent'); no.apply_baseline(BASELINE,mode='percent')
   x=st; x.data=st.data-no.data
  elif result=='ratio': x=st; x.data=(st.data-no.data)/(st.data+no.data+np.finfo(float).eps)
  x.data=x.data.mean(axis=0,keepdims=True); x.info=mne.create_info(['ROI_mean'],x.info['sfreq'],'eeg'); xs.append(x); used.append(s); sensors[s]=available
 if not xs:return None,[],{}
 g=xs[0].copy(); g.data=np.mean([x.data for x in xs],axis=0); g.nave=len(xs); return g,used,sensors
def vlim(x):
 ti=(x.times>=TFR_TMIN)&(x.times<=TFR_TMAX); z=np.asarray(x.data)[...,ti]; z=z[np.isfinite(z)]
 if not z.size:return (None,None)
 m=float(np.percentile(np.abs(z),ROBUST)); return (-m,m) if np.isfinite(m) and m>0 else (None,None)
def style(fig):
 fig.patch.set_facecolor('white')
 for ax in fig.axes: ax.set_facecolor('white')
 return fig
def scalp_erp(compare,channels):
 # MNE topo axes can overlap for dense montages; increase figure substantially and
 # shrink every sensor axis around its centre to guarantee visible separation.
 fig=mne.viz.plot_compare_evokeds(compare,picks=channels,combine=None,axes='topo',show=False,ci=False,truncate_xaxis=False,truncate_yaxis=False,legend=True)
 fig=fig[0] if isinstance(fig,list) else fig; fig.set_size_inches(18,14,forward=True)
 for ax in fig.axes:
  if not ax.get_position().width: continue
  box=ax.get_position(); cx=box.x0+box.width/2; cy=box.y0+box.height/2; w=box.width*.72; h=box.height*.72; ax.set_position([cx-w/2,cy-h/2,w,h])
 return style(fig)
def scalp_tfr(x,lim):
 # plot_topo does not accept vlim in the installed MNE version. Set image clim after plotting.
 fig=x.plot_topo(tmin=TFR_TMIN,tmax=TFR_TMAX,fmin=2,fmax=31.5,baseline=None,mode=None,cmap='RdBu_r',show=False); fig.set_size_inches(18,14,forward=True)
 for ax in fig.axes:
  box=ax.get_position(); cx=box.x0+box.width/2; cy=box.y0+box.height/2; w=box.width*.72; h=box.height*.72; ax.set_position([cx-w/2,cy-h/2,w,h])
  if None not in lim:
   for im in ax.images: im.set_clim(*lim)
 return style(fig)
def eight_erp(compare,available,counts):
 fig,axs=plt.subplots(2,4,figsize=(18,8),constrained_layout=True)
 for ax,ch in zip(axs.ravel(),ROI8):
  if ch not in available: ax.axis('off'); ax.set_title(f'{ch}: unavailable'); continue
  mne.viz.plot_compare_evokeds(compare,picks=ch,axes=ax,show=False,ci=False,truncate_xaxis=False,truncate_yaxis=False); ax.axvline(0,color='k',ls='--',lw=.8); ax.set_title(f'{ch} (n={counts[ch]})')
 return fig
def eight_tfr(x,available,counts,lim):
 fig,axs=plt.subplots(2,4,figsize=(20,9),constrained_layout=True)
 for ax,ch in zip(axs.ravel(),ROI8):
  if ch not in available: ax.axis('off'); ax.set_title(f'{ch}: unavailable'); continue
  kw=dict(picks=[ch],tmin=TFR_TMIN,tmax=TFR_TMAX,fmin=2,fmax=31.5,baseline=None,mode=None,axes=ax,show=False,colorbar=True,cmap='RdBu_r');
  if None not in lim: kw['vlim']=lim
  x.plot(**kw); ax.axvline(0,color='k',ls='--',lw=.8); ax.set_title(f'{ch} (n={counts[ch]})')
 return fig
def roi_erp_plot(no,st,title):
 fig=mne.viz.plot_compare_evokeds({'No stimulation':no,'Stimulation':st},picks=['ROI_mean'],show=False,ci=False,truncate_xaxis=False,truncate_yaxis=False); fig=fig[0] if isinstance(fig,list) else fig; fig.axes[0].axvline(0,color='k',ls='--'); fig.axes[0].set_title(title); return fig
def roi_tfr_plot(x,title,lim):
 kw=dict(picks=['ROI_mean'],tmin=TFR_TMIN,tmax=TFR_TMAX,fmin=2,fmax=31.5,baseline=None,mode=None,show=False,colorbar=True,cmap='RdBu_r');
 if None not in lim:kw['vlim']=lim
 fig=x.plot(**kw); fig=fig[0] if isinstance(fig,list) else fig; fig.axes[0].set_title(title); return fig
def add_erp(report,figs,grand,subject_evokeds,subjects,by_ch,channels):
 compare={'No stimulation':grand['no-stim'],'Stimulation':grand['stim']}; counts={ch:len(by_ch[ch]) for ch in channels}; available8=[ch for ch in ROI8 if ch in channels]
 report.add_text('ERP analysis details',f'Cue-locked ERP; cue-left/right combined. Each subject was trial-averaged first. Evoked low-pass={ERP_LP:g} Hz; baseline={ERP_BASELINE}; display={ERP_TMIN} to {ERP_TMAX} s. Each sensor grand average includes only subjects retaining that sensor in both conditions.','ERP')
 report.add_figure(scalp_erp(compare,channels),str(figs/'ERP_all_channels_scalp.png'),'ERP: all channels in non-overlapping scalp layout','Stimulation and no stimulation are overlaid at every available sensor. Enlarged canvas and reduced sensor axes prevent overlap.','ERP')
 report.add_figure(eight_erp(compare,available8,counts),str(figs/'ERP_eight_posterior_2x4.png'),'ERP: eight posterior sensors (2 x 4)','PO3, POz, PO4, O1, Oz, O2, PO7 and PO8; unavailable channels are labelled.','ERP')
 for roi,name in ((ROI8,'8 posterior sensors'),(ROI3,'PO3/POz/PO4')):
  no,u,_=roi_evoked(subject_evokeds,subjects,roi,'no-stim'); st,_,_=roi_evoked(subject_evokeds,subjects,roi,'stim')
  if no is not None and st is not None: report.add_figure(roi_erp_plot(no,st,f'Grand-average ERP: mean {name}'),str(figs/f"ERP_mean_{len(roi)}_posterior.png"),f'ERP: mean {name}',f'ROI sensors were averaged within participant first, then participant ROI ERPs were grand-averaged (n={len(u)}).','ERP')
def add_tfr_result(report,figs,x,subject_tfrs,subjects,by_ch,channels,result,stem,title,details,diff_bl):
 lim=vlim(x); counts={ch:len(by_ch[ch]) for ch in channels}; available8=[ch for ch in ROI8 if ch in channels]; scale=f'Robust symmetric scale {lim[0]:.4g} to {lim[1]:.4g}.' if None not in lim else 'Automatic scale.'
 section=f'TFR - {title}'
 report.add_text('Analysis details',details+' '+scale+' Individual sensors include only subjects retaining that channel in both conditions. ROI averages are formed within participant first, then across participants.',section)
 report.add_figure(scalp_tfr(x,lim),str(figs/f'{stem}_all_channels_scalp.png'),f'{title}: all channels in non-overlapping scalp layout',details+' '+scale,section)
 report.add_figure(eight_tfr(x,available8,counts,lim),str(figs/f'{stem}_eight_posterior_2x4.png'),f'{title}: eight posterior sensors (2 x 4)',details+' '+scale,section)
 for roi,name in ((ROI8,'8 posterior sensors'),(ROI3,'PO3/POz/PO4')):
  r,u,_=roi_tfr(subject_tfrs,subjects,roi,result,diff_bl)
  if r is not None:
   rl=vlim(r); report.add_figure(roi_tfr_plot(r,f'{title}: mean {name}',rl),str(figs/f'{stem}_mean_{len(roi)}_posterior.png'),f'{title}: mean {name}',details+f' ROI computed within participant first and then grand-averaged (n={len(u)}).',section)
def main():
 a=args(); subjects=[s.removeprefix('sub-') for s in a.subjects]; diff_bl=diff_baseline_choice(); root=resolve_project_root(a.platform,a.project_root)
 out=root/'derivatives'/'reports'/'group'/'EEG_all_channels_complete_grand_average'; figs=out/'figures'; figs.mkdir(parents=True,exist_ok=True); deriv=root/'data'/'BIDS'/'derivatives'/'group'/'EEG_all_channels_complete_grand_average'; deriv.mkdir(parents=True,exist_ok=True); rid='complete_grand_average_'+'_'.join(subjects); report=ParticipantPDF(str(out),rid)
 epochs={s:load_epochs(root,s,a) for s in subjects}; good={s:good_channels(epochs[s]) for s in subjects}; channels=[]
 for s in subjects:
  for ch in good[s]:
   if ch not in channels:channels.append(ch)
 by_ch={ch:[s for s in subjects if ch in good[s]] for ch in channels}; info=template_info(epochs,channels); channels=info.ch_names; by_ch={ch:by_ch[ch] for ch in channels}
 report.add_text('Subjects included',f'n={len(subjects)}\n'+', '.join('sub-'+s for s in subjects),'Group overview'); report.add_text('Subjects contributing to each EEG channel','\n'.join(f"{ch} (n={len(by_ch[ch])}): "+', '.join('sub-'+s for s in by_ch[ch]) for ch in channels),'Group overview')
 csvp=deriv/f'{rid}_subjects_by_channel.csv'
 with csvp.open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f); w.writerow(['channel','n_subjects','subjects']); [w.writerow([ch,len(by_ch[ch]),';'.join('sub-'+s for s in by_ch[ch])]) for ch in channels]
 sev={s:{c:evoked(epochs[s][c],good[s]) for c in CONDITIONS} for s in subjects}; gev={c:channel_grand(sev,by_ch,info,channels,c,False) for c in CONDITIONS}; add_erp(report,figs,gev,sev,subjects,by_ch,channels)
 stfr={}
 for s in subjects:
  stfr[s]={}
  for c in CONDITIONS: print(f'Computing TFR sub-{s} {c}'); stfr[s][c]=tfr(epochs[s][c],good[s],a.n_jobs)
 raw={c:channel_grand(stfr,by_ch,info,channels,c,True) for c in CONDITIONS}; no=raw['no-stim'].copy().apply_baseline(BASELINE,mode='percent'); st=raw['stim'].copy().apply_baseline(BASELINE,mode='percent')
 sd=raw['stim'].copy(); nd=raw['no-stim'].copy()
 if diff_bl: sd.apply_baseline(BASELINE,mode='percent'); nd.apply_baseline(BASELINE,mode='percent')
 diff=sd.copy(); diff.data=sd.data-nd.data; ratio=raw['stim'].copy(); ratio.data=(raw['stim'].data-raw['no-stim'].data)/(raw['stim'].data+raw['no-stim'].data+np.finfo(float).eps)
 common=f'Multitaper 2-31.5 Hz in 0.5-Hz steps; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; FFT=True; zero_mean=True; ITC=False; trial-average=True; decim={DECIM}; cue onset=0 s.'
 add_tfr_result(report,figs,no,stfr,subjects,by_ch,channels,'no-stim','TFR_no_stim','No stimulation TFR',common+f' Percent baseline {BASELINE}.',diff_bl)
 add_tfr_result(report,figs,st,stfr,subjects,by_ch,channels,'stim','TFR_stim','Stimulation TFR',common+f' Percent baseline {BASELINE}.',diff_bl)
 add_tfr_result(report,figs,diff,stfr,subjects,by_ch,channels,'difference','TFR_difference','Stimulation - no stimulation TFR',common+f' Difference baseline correction={diff_bl}.',diff_bl)
 add_tfr_result(report,figs,ratio,stfr,subjects,by_ch,channels,'ratio','TFR_ratio','(Stimulation - no stimulation) / (stimulation + no stimulation)',common+' Calculated from original unbaselined power; no baseline correction.',diff_bl)
 manuscript=(f'EEG analysis was performed on final cleaned cue-locked epochs from {len(subjects)} participants. Cue-left and cue-right trials were combined within stimulation condition. Epochs were not concatenated across participants. Each participant was averaged separately before group averaging, ensuring equal participant weighting. For each EEG sensor, only participants retaining that sensor in both stimulation conditions contributed to its sensor-specific grand average; missing sensors were not interpolated at group level. Cue-locked ERPs were low-pass filtered at {ERP_LP:g} Hz and baseline corrected from {ERP_BASELINE[0]:g} to {ERP_BASELINE[1]:g} s. Time-frequency power was estimated using multitaper decomposition from 2 to 31.5 Hz in 0.5-Hz steps, with n_cycles equal to frequency/2, time-bandwidth {TIME_BANDWIDTH:g}, FFT enabled, zero-mean tapers, no ITC, trial averaging, and decimation by {DECIM}. Stimulation and no-stimulation TFRs were expressed as percent change relative to the {BASELINE[0]:g} to {BASELINE[1]:g} s baseline. The stimulation-minus-no-stimulation contrast was calculated with baseline correction set to {diff_bl}. The normalized contrast (stimulation - no stimulation)/(stimulation + no stimulation) was calculated from original unbaselined power. In addition to sensor-wise scalp-layout results, posterior activity was summarized using an eight-sensor ROI (PO3, POz, PO4, O1, Oz, O2, PO7, PO8) and a three-sensor ROI (PO3, POz, PO4). For each ROI, available sensors were averaged within each participant before participant-level ROI values were grand-averaged. This preserved equal participant weighting while allowing participants with individual rejected ROI sensors to contribute without interpolation.')
 report.add_text('Analysis report - manuscript style',manuscript,'Manuscript-style analysis report'); report.add_text('Exact reproducibility parameters',f'Subjects: '+', '.join('sub-'+s for s in subjects)+f'\nCue epochs only; attention directions combined.\nERP: low-pass={ERP_LP:g} Hz; baseline={ERP_BASELINE}; display={ERP_TMIN} to {ERP_TMAX} s.\nTFR frequencies={FREQS[0]:g}-{FREQS[-1]:g} Hz; step=0.5 Hz; multitaper; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; decim={DECIM}; baseline={BASELINE} percent for condition plots.\nDifference baseline correction={diff_bl}; ratio uses unbaselined power.\nROI8={ROI8}; ROI3={ROI3}.','Manuscript-style analysis report')
 (deriv/f'{rid}_analysis.json').write_text(json.dumps({'subjects':['sub-'+s for s in subjects],'subjects_by_channel':{ch:['sub-'+s for s in by_ch[ch]] for ch in channels},'ROI8':ROI8,'ROI3':ROI3,'difference_baseline':diff_bl,'frequencies_hz':FREQS.tolist()},indent=2)+'\n'); print(f'Complete group report: {report.pdf_fname}')
if __name__=='__main__':main()
