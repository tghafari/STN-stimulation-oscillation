"""P01: 1-100 Hz filtering -> PyPREP/manual channel QC -> PSD review -> segmentation -> epochs -> reference.

Continuous EEG is band-pass filtered 1-100 Hz before PyPREP. After PyPREP and
manual raw-browser channel QC, rejected-channel PSDs can be used to rescue channels.
The PSD of the remaining good channels is then displayed and the user gets one final
opportunity to mark additional channels bad based on that PSD before segmentation.
Average EEG rereferencing is ON by default; use --rereference none to retain original reference.
"""
from __future__ import annotations
import argparse,json
import matplotlib.pyplot as plt
import mne
from mne_bids import read_raw_bids
from pyprep.find_noisy_channels import NoisyChannels
from pipeline_config import CONDITIONS,base_bids_path,crop_table_path,qc_dir,resolve_project_root,stage_path
from all_channel_report import participant_report,figure_dir,fmt_channels
EVENT_DICT={'cue_onset_right':1,'cue_onset_left':2,'trial_onset':3,'stim_onset':4,'catch_onset':5,'dot_onset_right':6,'dot_onset_left':7,'response_press_onset':8,'block_onset':20,'block_end':21,'experiment_end':30,'new_stim_segment':99999}
def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',required=True);p.add_argument('--session',default='01');p.add_argument('--task',default='SpAtt');p.add_argument('--run',default='01');p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);p.add_argument('--rereference',choices=['avg','none'],default='avg');return p.parse_args()
def pyprep_reasons(raw):
 eeg=raw.copy().pick('eeg');
 if eeg.get_montage() is None:eeg.set_montage('standard_1020',on_missing='warn')
 noisy=NoisyChannels(eeg,random_state=42);errors=[]
 for label,func in [('deviation',noisy.find_bad_by_deviation),('high-frequency noise',noisy.find_bad_by_hfnoise),('correlation',noisy.find_bad_by_correlation),('RANSAC',noisy.find_bad_by_ransac)]:
  try:func()
  except Exception as exc:errors.append(f'{label}: {type(exc).__name__}: {exc}')
 amap={'bad_by_deviation':'deviation','bad_by_hf_noise':'high-frequency noise','bad_by_correlation':'correlation','bad_by_ransac':'RANSAC','bad_by_nan':'NaN/flat data','bad_by_SNR':'poor signal-to-noise ratio'};reasons={}
 for attr,why in amap.items():
  for ch in getattr(noisy,attr,[]) or []:reasons.setdefault(str(ch),[]).append(why)
 for ch in noisy.get_bads():reasons.setdefault(str(ch),[]).append('PyPREP overall decision')
 return {k:sorted(set(v)) for k,v in reasons.items()},errors
def make_segment(raw,times):
 pieces=[raw.copy().crop(tmin=float(times[0]),tmax=float(times[1]))]
 if len(times)==4:pieces.append(raw.copy().crop(tmin=float(times[2]),tmax=float(times[3])))
 return pieces[0] if len(pieces)==1 else mne.concatenate_raws(pieces,on_mismatch='warn')
def define_epochs(raw):
 events,ids=mne.events_from_annotations(raw,event_id=EVENT_DICT);cue={k:ids[k] for k in ('cue_onset_right','cue_onset_left') if k in ids}
 if not cue:raise RuntimeError('No cue onset events found after stimulation segmentation.')
 return mne.Epochs(raw,events,cue,tmin=-0.5,tmax=1.6,baseline=None,detrend=1,proj=True,picks='all',reject=None,reject_by_annotation=False,preload=True,event_repeated='merge')
def main():
 a=parse_args();s=a.subject.removeprefix('sub-');root=resolve_project_root(a.platform,a.project_root);report=participant_report(root,s);figs=figure_dir(root,s);reref='avg' if a.rereference=='avg' else None;label='average EEG reference' if reref else 'none';raw=read_raw_bids(base_bids_path(root,s,a.session,a.task,a.run),verbose=True,extra_params={'preload':True})
 if raw.get_montage() is None:raw.set_montage('standard_1020',on_missing='warn')
 print('\nBand-pass filtering continuous EEG: 1-100 Hz before PyPREP.');raw.filter(l_freq=1.0,h_freq=100.0)
 reasons,errors=pyprep_reasons(raw);suggested=sorted(reasons);raw.info['bads']=sorted(set(raw.info['bads'])|set(suggested));print('PyPREP suggested:',suggested or 'None')
 before=set(raw.info['bads']);raw.plot(n_channels=min(30,len(raw.ch_names)),duration=20.,block=True,title=f'sub-{s}: post-PyPREP manual channel QC');after=set(raw.info['bads']);manual=sorted(after-before);browser_rescues=sorted(before-after);manual_reasons={}
 for ch in manual:
  while not (r:=input(f'Reason for manually rejecting {ch}: ').strip()):print('Please enter a reason.');
  manual_reasons[ch]=r
 rejected=sorted(after);rescue=[]
 if rejected:
  rr=raw.copy().pick(rejected);rr.info['bads']=[];spec=rr.compute_psd(fmin=1.,fmax=min(100.,rr.info['sfreq']/2.));f=spec.plot(show=False);report.add_figure(f,str(figs/'P01_rejected_channels_PSD.png'),'PSD of PyPREP + manually rejected channels',f'1-100 Hz filtered data. Proposed rejection: {fmt_channels(rejected)}.','All-channel preprocessing');plt.close(f);spec.plot(show=True);plt.show(block=True);rescue=input('Rejected channels that are actually GOOD (space-separated, Enter for none): ').strip().split()
 unknown=[ch for ch in rescue if ch not in rejected]
 if unknown:raise ValueError(f'Can only rescue channels shown in PSD: {unknown}')
 final=sorted(set(rejected)-set(rescue));raw.info['bads']=final;good=[ch for ch in raw.copy().pick('eeg').ch_names if ch not in final]
 if not good:raise RuntimeError('No good EEG channels remain.')
 gr=raw.copy().pick(good);gr.info['bads']=[];gs=gr.compute_psd(fmin=1.,fmax=min(100.,gr.info['sfreq']/2.));gf=gs.plot(show=False);report.add_figure(gf,str(figs/'P01_final_good_channels_PSD.png'),'PSD of remaining good EEG channels before final PSD review',f'1-100 Hz filtered retained EEG before segmentation: {fmt_channels(good)}.','All-channel preprocessing');plt.close(gf);gs.plot(show=True);plt.show(block=True)
 # Final optional rejection is deliberately placed here, immediately after the PSD
 # of the remaining channels, before segmentation/epoching/rereferencing.
 while True:
  psd_bad=input('Additional EEG channels to mark BAD based on the remaining-channel PSD (space/comma separated, Enter for none): ').replace(',',' ').split();unknown=[ch for ch in psd_bad if ch not in good]
  if not unknown:break
  print('Can only reject currently good EEG channels:',unknown)
 psd_reasons={}
 for ch in psd_bad:
  while not (r:=input(f'Reason for rejecting {ch} based on PSD: ').strip()):print('Please enter a reason.')
  psd_reasons[ch]=r
 final=sorted(set(final)|set(psd_bad));raw.info['bads']=final;good=[ch for ch in raw.copy().pick('eeg').ch_names if ch not in final]
 if not good:raise RuntimeError('No good EEG channels remain after PSD review.')
 if psd_bad:
  gr2=raw.copy().pick(good);gr2.info['bads']=[];gs2=gr2.compute_psd(fmin=1.,fmax=min(100.,gr2.info['sfreq']/2.));gf2=gs2.plot(show=False);report.add_figure(gf2,str(figs/'P01_final_good_channels_PSD_after_PSD_rejection.png'),'Final PSD after PSD-based channel rejection',f'Additional PSD-based rejection: {fmt_channels(psd_bad)}. Remaining EEG: {fmt_channels(good)}.','All-channel preprocessing');plt.close(gf2)
 table=json.loads(crop_table_path().read_text());key=f'sub-{s}';audit={'subject':key,'high_pass_hz':1.0,'low_pass_hz':100.0,'filter':'1-100 Hz band-pass on continuous EEG before PyPREP','rereference':reref,'rereference_requested':a.rereference,'pyprep_reasons':reasons,'detector_errors':errors,'pyprep_suggested_bad_channels':suggested,'manual_raw_browser_additions':manual,'manual_raw_browser_reasons':manual_reasons,'raw_browser_rescued_pyprep_channels':browser_rescues,'channels_rescued_after_rejected_psd':rescue,'additional_bad_channels_from_remaining_channel_psd':psd_bad,'additional_bad_channel_psd_reasons':psd_reasons,'bad_channels':final,'final_good_eeg_channels':good,'conditions':{}}
 for c in CONDITIONS:
  times=table[key][c];seg=make_segment(raw,times);seg.info['bads']=[x for x in final if x in seg.ch_names];ep=define_epochs(seg);ep.info['bads']=[x for x in final if x in ep.ch_names]
  if reref=='avg':ep.set_eeg_reference(ref_channels='average',projection=False)
  out=stage_path(root,s,a.session,a.task,a.run,c,'reref','epo');ep.save(out,overwrite=True);audit['conditions'][c]={'crop_times_sec':times,'n_epochs':len(ep),'rereference':label,'output_file':str(out)};report.add_text(f'{c}: stimulation segmentation',f'Continuous EEG was already band-pass filtered 1-100 Hz.\nKept ranges (s): {times}\nCue epochs: -0.5 to +1.6 s; baseline=None; detrend=1.\nRereferencing: {label}.\nEpochs: {len(ep)}','Stimulation segmentation')
 report.add_text('PyPREP, manual channel QC, PSD review, filtering and reference',f'Continuous EEG was band-pass filtered 1-100 Hz before PyPREP.\nPyPREP suggestions: {fmt_channels(suggested)}\nUser-rejected channels in raw browser: {fmt_channels(manual)}\nChannels rescued after QC: {fmt_channels(sorted(set(browser_rescues)|set(rescue)))}\nAdditional channels rejected after remaining-channel PSD review: {fmt_channels(psd_bad)}\nPSD-based reasons: '+('; '.join(f'{k}: {v}' for k,v in psd_reasons.items()) or 'None')+f'\nFINAL bad channels: {fmt_channels(final)}\nRereferencing: {label}.','All-channel preprocessing');(qc_dir(root,s)/'P01_pyprep_segment_epoch_reref.json').write_text(json.dumps(audit,indent=2)+'\n')
if __name__=='__main__':main()
