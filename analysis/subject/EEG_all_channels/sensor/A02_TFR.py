"""A02: TFR analysis with all-sensor scalp layout and user-reviewed eight-channel posterior/occipital ROI."""
from __future__ import annotations
import argparse,json
import matplotlib.pyplot as plt
import mne
import numpy as np
from pipeline_config import CONDITIONS,qc_dir,resolve_project_root,stage_path,subject_deriv_dir
from all_channel_report import participant_report,figure_dir,fmt_channels
ROI_CANDIDATES=('O7','O3','PO3','POz','Oz','PO4','O4','O8');BASELINE=(-0.3,-0.1);FREQS=np.arange(2.,31.);N_CYCLES=FREQS/2.;TIME_BANDWIDTH=2.;DECIM=2;PLOT_TMIN=-0.3;PLOT_TMAX=1.4;ROBUST_PERCENTILE=98.
def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',required=True);p.add_argument('--session',default='01');p.add_argument('--task',default='SpAtt');p.add_argument('--run',default='01');p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);p.add_argument('--n-jobs',type=int,default=4);return p.parse_args()
def load(root,s,a):
 d={}
 for c in CONDITIONS:
  p=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo');ep=mne.read_epochs(p,preload=True);keep=[k for k in ('cue_onset_right','cue_onset_left') if k in ep.event_id];d[c]=ep[keep] if keep else ep
 return d
def common_good(d):return [ch for ch in d['stim'].copy().pick('eeg').ch_names if ch in d['no-stim'].ch_names and ch not in d['stim'].info['bads'] and ch not in d['no-stim'].info['bads']]
def compute(ep,j):return ep.compute_tfr(method='multitaper',freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,return_itc=False,average=True,decim=DECIM,n_jobs=j)
def vlim(t):
 x=np.asarray(t.data);x=x[np.isfinite(x)];v=float(np.percentile(np.abs(x),ROBUST_PERCENTILE)) if x.size else 0.;return (-v,v) if v>0 else None
def scalp(t,lim):
 fig=t.plot_topo(tmin=PLOT_TMIN,tmax=PLOT_TMAX,baseline=None,mode=None,cmap='RdBu_r',show=False);fig.patch.set_facecolor('white')
 for ax in fig.axes:
  ax.set_facecolor('white')
  for im in ax.images:
   if lim:im.set_clim(*lim)
 if lim:
  sm=plt.cm.ScalarMappable(cmap='RdBu_r',norm=plt.Normalize(*lim));sm.set_array([]);cax=fig.add_axes([.92,.18,.018,.64]);cb=fig.colorbar(sm,cax=cax);cb.set_ticks(np.linspace(lim[0],lim[1],5));cb.set_label('TFR power')
 return fig
def separate(t,chs,title,lim):
 fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True);axes=axes.ravel()
 for ax in axes[len(chs):]:ax.axis('off')
 for ax,ch in zip(axes,chs):t.plot(picks=ch,tmin=PLOT_TMIN,tmax=PLOT_TMAX,baseline=None,mode=None,vlim=lim,cmap='RdBu_r',axes=ax,show=False,colorbar=True);ax.set_title(ch)
 fig.suptitle(title);return fig
def roi_mean(t,chs,title,lim):
 roi=t.copy().pick(chs);roi.data=roi.data.mean(axis=0,keepdims=True);roi.info=mne.create_info(['ROI_mean'],sfreq=roi.info['sfreq'],ch_types=['eeg']);fig=roi.plot(picks='ROI_mean',tmin=PLOT_TMIN,tmax=PLOT_TMAX,baseline=None,mode=None,vlim=lim,cmap='RdBu_r',show=False,colorbar=True);fig=fig[0] if isinstance(fig,list) else fig;fig.axes[0].set_title(title);return fig
def choose(candidates):
 print('\nInspect the all-sensor scalp-layout TFRs before defining the posterior/occipital ROI.');print('Available candidates:',', '.join(candidates))
 while True:
  ex=input('Channels to EXCLUDE from TFR ROI mean (space/comma separated, Enter for none): ').replace(',',' ').split();bad=[x for x in ex if x not in candidates]
  if not bad:return [x for x in candidates if x not in ex],sorted(set(ex))
  print('Invalid candidate(s):',bad)
def add(report,figs,t,stem,title,caption,candidates,roi=None,mean=False):
 lim=vlim(t);report.add_figure(scalp(t,lim),str(figs/f'{stem}_sensor_topography.png'),f'{title}: all sensors separately in scalp layout',caption+f' Shared robust scale: {lim}.','Time-frequency analysis');report.add_figure(separate(t,candidates,title,lim),str(figs/f'{stem}_ROI_candidates_separate.png'),f'{title}: eight posterior/occipital candidates separately',caption+f' Candidates: {fmt_channels(candidates)}.','Time-frequency analysis')
 if mean and roi:report.add_figure(roi_mean(t,roi,f'{title}: ROI mean',lim),str(figs/f'{stem}_ROI_mean.png'),f'{title}: user-reviewed ROI mean',caption+f' Final ROI: {fmt_channels(roi)}.','Time-frequency analysis')
 return lim
def main():
 a=parse_args();s=a.subject.removeprefix('sub-');root=resolve_project_root(a.platform,a.project_root);report=participant_report(root,s);figs=figure_dir(root,s);deriv=subject_deriv_dir(root,s);eps=load(root,s,a);common=common_good(eps);candidates=[x for x in ROI_CANDIDATES if x in common]
 if not candidates:raise RuntimeError('No predefined ROI candidates are good in both conditions.')
 raw={c:compute(eps[c].copy().pick(common),a.n_jobs) for c in CONDITIONS};display={c:raw[c].copy().apply_baseline(BASELINE,mode='percent') for c in CONDITIONS};scales={}
 # First generate scalp layouts and separate candidate panels for ALL analyses; no ROI mean yet.
 for c in CONDITIONS:
  name='Stimulation' if c=='stim' else 'No stimulation';scales[c]=add(report,figs,display[c],f'A02_{c}_TFR',f'{name}: combined attention-left/right TFR',f'Percent baseline {BASELINE}; multitaper 2-30 Hz.',candidates)
 diff=raw['stim'].copy();diff.data=raw['stim'].data-raw['no-stim'].data;scales['difference']=add(report,figs,diff,'A02_stim_minus_no_stim_TFR','TFR: stimulation - no stimulation','Unbaselined power; NO baseline correction.',candidates)
 ratio=raw['stim'].copy();ratio.data=(raw['stim'].data-raw['no-stim'].data)/(raw['stim'].data+raw['no-stim'].data+np.finfo(float).eps);scales['ratio']=add(report,figs,ratio,'A02_stim_normalized_difference_TFR','TFR: (stim - no-stim) / (stim + no-stim)','Unbaselined power; NO baseline correction.',candidates)
 plt.show(block=False);roi,excluded=choose(candidates)
 if not roi:raise RuntimeError('ROI cannot be empty.')
 # ROI mean for every TFR analysis, using the same reviewed channel set.
 for key,t,title in [('no-stim',display['no-stim'],'No stimulation: combined attention-left/right TFR'),('stim',display['stim'],'Stimulation: combined attention-left/right TFR'),('difference',diff,'TFR: stimulation - no stimulation'),('ratio',ratio,'TFR: (stim - no-stim) / (stim + no-stim)')]:report.add_figure(roi_mean(t,roi,f'{title}: ROI mean',scales[key]),str(figs/f'A02_{key}_TFR_ROI_mean.png'),f'{title}: user-reviewed posterior/occipital ROI mean',f'Initial candidates: {fmt_channels(candidates)}. Excluded after scalp-layout review: {fmt_channels(excluded)}. Final ROI: {fmt_channels(roi)}.','Time-frequency analysis')
 for c in CONDITIONS:raw[c].save(deriv/f'sub-{s}_ses-{a.session}_task-{a.task}_run-{a.run}_{c}_desc-allchannels_tfr.h5',overwrite=True)
 details={'subject':f'sub-{s}','method':'multitaper','frequencies_hz':FREQS.tolist(),'n_cycles':'frequency / 2','time_bandwidth':2,'decim':2,'condition_baseline_s':list(BASELINE),'condition_baseline_mode':'percent','comparative_baseline':None,'roi_candidates_predefined':list(ROI_CANDIDATES),'roi_candidates_available':candidates,'roi_excluded_by_user':excluded,'roi_final':roi,'vlims':scales};(qc_dir(root,s)/'A02_tfr_analysis.json').write_text(json.dumps(details,indent=2)+'\n');report.add_text('TFR ROI decision',f'All-sensor scalp-layout and eight candidate-channel plots were generated before defining the ROI mean. Initial candidates: {fmt_channels(candidates)}. User exclusions: {fmt_channels(excluded)}. Final ROI used identically for all TFR ROI means: {fmt_channels(roi)}.','Time-frequency analysis');print(f'TFR complete for sub-{s}.')
if __name__=='__main__':main()
