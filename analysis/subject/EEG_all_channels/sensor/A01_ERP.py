"""A01: ERP analysis with scalp layout, eight posterior/occipital sensors, and user-defined ROI mean."""
from __future__ import annotations
import argparse,json
import matplotlib.pyplot as plt
import mne
from pipeline_config import CONDITIONS,qc_dir,resolve_project_root,stage_path
from all_channel_report import participant_report,figure_dir,fmt_channels
ROI_CANDIDATES=('PO3','POz','PO4','O1','Oz','O2','PO7','PO8')
ERP_BASELINE=(-0.1,0.0);ERP_LP_HZ=30.;ERP_TMIN=-0.1;ERP_TMAX=0.5
def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',required=True);p.add_argument('--session',default='01');p.add_argument('--task',default='SpAtt');p.add_argument('--run',default='01');p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);return p.parse_args()
def load(root,s,a):
 d={}
 for c in CONDITIONS:
  p=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo');ep=mne.read_epochs(p,preload=True);keep=[k for k in ('cue_onset_right','cue_onset_left') if k in ep.event_id];d[c]=ep[keep] if keep else ep
 return d
def common_good(d):return [ch for ch in d['stim'].copy().pick('eeg').ch_names if ch in d['no-stim'].ch_names and ch not in d['stim'].info['bads'] and ch not in d['no-stim'].info['bads']]
def evoked(ep,picks):
 x=ep.copy().pick(picks).average();x.filter(None,ERP_LP_HZ);x.apply_baseline(ERP_BASELINE);x.crop(ERP_TMIN,ERP_TMAX);return x
def choose_roi(candidates):
 print('\nInspect the eight posterior/occipital ERP channels shown in the figure, then close the figure to continue.');print('Available ROI candidates:',', '.join(candidates))
 while True:
  exclude=input('Channels to EXCLUDE from ROI mean (space/comma separated, Enter for none): ').replace(',',' ').split();unknown=[x for x in exclude if x not in candidates]
  if not unknown:return [x for x in candidates if x not in exclude],sorted(set(exclude))
  print('Invalid candidate(s):',unknown)
def main():
 a=parse_args();s=a.subject.removeprefix('sub-');root=resolve_project_root(a.platform,a.project_root);report=participant_report(root,s);figs=figure_dir(root,s);eps=load(root,s,a);common=common_good(eps)
 if not common:raise RuntimeError('No common good EEG sensors.')
 ev={c:evoked(eps[c],common) for c in CONDITIONS};compare={'no stimulation':ev['no-stim'],'stimulation':ev['stim']}
 topo=mne.viz.plot_compare_evokeds(compare,picks=common,combine=None,axes='topo',show=False,ci=False,truncate_xaxis=False,truncate_yaxis=False,legend=True);topo=topo[0] if isinstance(topo,list) else topo;report.add_figure(topo,str(figs/'A01_ERP_stim_vs_no_stim_sensor_topography.png'),'ERP: stimulation vs no stimulation, all sensors in scalp layout',f'Each of {len(common)} common good EEG sensors is displayed separately.','ERP analysis')
 candidates=[ch for ch in ROI_CANDIDATES if ch in common]
 if not candidates:raise RuntimeError('None of the eight posterior/occipital ROI candidates are good in both conditions.')
 fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True);axes=axes.ravel()
 for ax in axes[len(candidates):]:ax.axis('off')
 for ax,ch in zip(axes,candidates):mne.viz.plot_compare_evokeds(compare,picks=ch,combine=None,axes=ax,show=False,ci=False,truncate_xaxis=False,truncate_yaxis=False);ax.axvline(0,color='k',linestyle='--',linewidth=1);ax.set_xlim(ERP_TMIN,ERP_TMAX);ax.set_title(ch)
 report.add_figure(fig,str(figs/'A01_ERP_stim_vs_no_stim_ROI_candidates_separate.png'),'ERP: eight posterior/occipital ROI candidates separately',f'Candidate channels displayed separately: {fmt_channels(candidates)}.','ERP analysis')
 # ROI decision is based only on this eight-channel figure, not the all-sensor scalp layout.
 plt.figure(fig.number);plt.show(block=True)
 roi,excluded=choose_roi(candidates)
 if not roi:raise RuntimeError('ROI cannot be empty.')
 fm=mne.viz.plot_compare_evokeds(compare,picks=roi,combine='mean',show=False,ci=False,truncate_xaxis=False,truncate_yaxis=False);fm=fm[0] if isinstance(fm,list) else fm;fm.axes[0].axvline(0,color='k',linestyle='--',linewidth=1);fm.axes[0].set_xlim(ERP_TMIN,ERP_TMAX);fm.axes[0].set_title(f'sub-{s}: posterior/occipital ROI mean')
 report.add_figure(fm,str(figs/'A01_ERP_stim_vs_no_stim_ROI_mean.png'),'ERP: posterior/occipital ROI mean',f'Final channels contributing to ROI mean: {fmt_channels(roi)}.','ERP analysis')
 for c in CONDITIONS:mne.write_evokeds(stage_path(root,s,a.session,a.task,a.run,c,'erp','ave'),ev[c],overwrite=True)
 details={'subject':f'sub-{s}','epoch_original_window_s':[-0.5,1.6],'erp_display_window_s':[ERP_TMIN,ERP_TMAX],'baseline_s':list(ERP_BASELINE),'evoked_low_pass_hz':ERP_LP_HZ,'roi_candidates_predefined':list(ROI_CANDIDATES),'roi_candidates_available':candidates,'roi_excluded_by_user':excluded,'roi_final':roi,'roi_decision_view':'eight separate posterior/occipital ERP channels only','common_good_eeg_sensors':common};(qc_dir(root,s)/'A01_erp_analysis.json').write_text(json.dumps(details,indent=2)+'\n')
 report.add_text('ERP analysis details',f'Attention-left/right trials combined. Trial-average ERP; 30-Hz low-pass; baseline -0.1 to 0 s; display -0.1 to 0.5 s. Eight separate posterior/occipital channels: {fmt_channels(ROI_CANDIDATES)}. ROI exclusion decisions were made from the eight-channel figure.\nFinal channels contributing to ROI mean: {fmt_channels(roi)}.','ERP analysis');print(f'ERP complete for sub-{s}.')
if __name__=='__main__':main()
