"""P01: PyPREP/manual channel QC -> segmentation -> epochs -> optional rereference.

Rereferencing is OFF by default. To apply an average EEG reference, run with
``--rereference avg``. All channel-QC decisions occur before stimulation segmentation.
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
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--subject',required=True); p.add_argument('--session',default='01'); p.add_argument('--task',default='SpAtt'); p.add_argument('--run',default='01'); p.add_argument('--platform',choices=['mac','bluebear'],default='mac'); p.add_argument('--project-root',default=None)
 p.add_argument('--rereference',choices=['avg'],default=None,help='Optional rereferencing. Omit for NO rereferencing; use "--rereference avg" for average EEG reference.')
 return p.parse_args()

def pyprep_reasons(raw):
 eeg=raw.copy().pick('eeg')
 if eeg.get_montage() is None:eeg.set_montage('standard_1020',on_missing='warn')
 noisy=NoisyChannels(eeg,random_state=42); errors=[]
 for label,func in [('deviation',noisy.find_bad_by_deviation),('high-frequency noise',noisy.find_bad_by_hfnoise),('correlation',noisy.find_bad_by_correlation),('RANSAC',noisy.find_bad_by_ransac)]:
  try:func()
  except Exception as exc:errors.append(f'{label}: {type(exc).__name__}: {exc}')
 amap={'bad_by_deviation':'deviation','bad_by_hf_noise':'high-frequency noise','bad_by_correlation':'correlation','bad_by_ransac':'RANSAC','bad_by_nan':'NaN/flat data','bad_by_SNR':'poor signal-to-noise ratio'}; reasons={}
 for attr,why in amap.items():
  for ch in getattr(noisy,attr,[]) or []:reasons.setdefault(str(ch),[]).append(why)
 for ch in noisy.get_bads():reasons.setdefault(str(ch),[]).append('PyPREP overall decision')
 return {k:sorted(set(v)) for k,v in reasons.items()},errors

def make_segment(raw,times):
 if len(times) not in (2,4):raise ValueError(f'Crop times must contain 2 or 4 values, got {times}')
 pieces=[raw.copy().crop(tmin=float(times[0]),tmax=float(times[1]))]
 if len(times)==4:pieces.append(raw.copy().crop(tmin=float(times[2]),tmax=float(times[3])))
 return pieces[0] if len(pieces)==1 else mne.concatenate_raws(pieces,on_mismatch='warn')

def define_epochs(raw):
 events,event_ids=mne.events_from_annotations(raw,event_id=EVENT_DICT); cue={k:event_ids[k] for k in ('cue_onset_right','cue_onset_left') if k in event_ids}
 if not cue:raise RuntimeError('No cue onset events found after stimulation segmentation.')
 return mne.Epochs(raw,events,cue,tmin=-0.5,tmax=1.6,baseline=None,detrend=1,proj=True,picks='all',reject=None,reject_by_annotation=False,preload=True,event_repeated='merge')

def ask_reasons(channels):
 out={}
 for ch in channels:
  while True:
   reason=input(f'Reason for manually rejecting {ch}: ').strip()
   if reason:out[ch]=reason;break
   print('Please enter a reason so the manual decision is auditable.')
 return out

def main():
 a=parse_args(); s=a.subject.removeprefix('sub-'); root=resolve_project_root(a.platform,a.project_root); report=participant_report(root,s); figs=figure_dir(root,s)
 reref_label='average EEG reference' if a.rereference=='avg' else 'none'
 print(f'\nRereferencing requested: {reref_label}')
 raw=read_raw_bids(base_bids_path(root,s,a.session,a.task,a.run),verbose=True,extra_params={'preload':True})
 if raw.get_montage() is None:raw.set_montage('standard_1020',on_missing='warn')
 raw.filter(l_freq=None,h_freq=100.0)
 reasons,errors=pyprep_reasons(raw); suggested=sorted(reasons); raw.info['bads']=sorted(set(raw.info['bads'])|set(suggested))
 print('\nPyPREP suggested bad channels:',suggested or 'None')
 for ch,why in sorted(reasons.items()):print(f'  {ch}: {", ".join(why)}')
 for err in errors:print('PyPREP detector error: '+err)

 before_browser=set(raw.info['bads']); print('\nOpening continuous raw data for manual channel QC. PyPREP channels are already marked bad.')
 raw.plot(n_channels=min(30,len(raw.ch_names)),duration=20.0,block=True,title=f'sub-{s}: post-PyPREP manual channel QC')
 after_browser=set(raw.info['bads']); manual_additions=sorted(after_browser-before_browser); browser_rescues=sorted(before_browser-after_browser); manual_reasons=ask_reasons(manual_additions)

 rejected_before_psd=sorted(after_browser)
 if rejected_before_psd:
  rr=raw.copy().pick(rejected_before_psd); rr.info['bads']=[]; spec=rr.compute_psd(fmin=0.5,fmax=min(100.,rr.info['sfreq']/2.)); f=spec.plot(show=False)
  report.add_figure(f,str(figs/'P01_rejected_channels_PSD.png'),'PSD of PyPREP + manually rejected channels',f'Channels proposed for rejection: {fmt_channels(rejected_before_psd)}.','All-channel preprocessing'); plt.close(f)
  print('\nOpening PSD of rejected channels ONLY.'); spec.plot(show=True); plt.show(block=True)
  rescue_psd=input('Rejected channels that are actually GOOD and should be rescued (space-separated, Enter for none): ').strip().split()
 else: rescue_psd=[]
 unknown=[ch for ch in rescue_psd if ch not in rejected_before_psd]
 if unknown:raise ValueError(f'Can only rescue channels shown in rejected-channel PSD: {unknown}')
 final=sorted(set(rejected_before_psd)-set(rescue_psd)); raw.info['bads']=final; print('FINAL bad channels:',final or 'None')

 good_eeg=[ch for ch in raw.copy().pick('eeg').ch_names if ch not in final]
 if not good_eeg:raise RuntimeError('No good EEG channels remain after channel QC.')
 gr=raw.copy().pick(good_eeg); gr.info['bads']=[]; gs=gr.compute_psd(fmin=0.5,fmax=min(100.,gr.info['sfreq']/2.)); gf=gs.plot(show=False)
 report.add_figure(gf,str(figs/'P01_final_good_channels_PSD.png'),'Final PSD of remaining good EEG channels',f'Final retained EEG channels before segmentation: {fmt_channels(good_eeg)}.','All-channel preprocessing'); plt.close(gf)
 print('\nOpening FINAL PSD of remaining GOOD EEG channels. Close to continue to segmentation.'); gs.plot(show=True); plt.show(block=True)

 table=json.loads(crop_table_path().read_text(encoding='utf-8')); key=f'sub-{s}'
 if key not in table:raise KeyError(f'No stimulation crop times for {key} in {crop_table_path()}')
 audit={'subject':key,'low_pass_hz':100,'rereference':a.rereference,'rereference_applied':a.rereference is not None,'pyprep_reasons':reasons,'detector_errors':errors,'pyprep_suggested_bad_channels':suggested,'manual_raw_browser_additions':manual_additions,'manual_raw_browser_reasons':manual_reasons,'raw_browser_rescued_pyprep_channels':browser_rescues,'rejected_channels_shown_in_psd':rejected_before_psd,'channels_rescued_after_rejected_psd':rescue_psd,'bad_channels':final,'final_good_eeg_channels':good_eeg,'conditions':{}}
 for condition in CONDITIONS:
  times=table[key][condition]; segment=make_segment(raw,times); segment.info['bads']=[ch for ch in final if ch in segment.ch_names]
  epochs=define_epochs(segment); epochs.info['bads']=[ch for ch in final if ch in epochs.ch_names]
  if a.rereference=='avg': epochs.set_eeg_reference(ref_channels='average',projection=False)
  out=stage_path(root,s,a.session,a.task,a.run,condition,'reref','epo'); epochs.save(out,overwrite=True)
  audit['conditions'][condition]={'crop_times_sec':times,'n_epochs':len(epochs),'rereference':reref_label}
  report.add_text(f'{condition}: stimulation segmentation',f'Kept ranges (s): {times}\nCue epochs: -0.5 to +1.6 s; baseline=None; detrend=1.\nRereferencing: {reref_label}.\nEpochs: {len(epochs)}','Stimulation segmentation')
 reason_text='\n'.join(f'{ch}: {", ".join(v)}' for ch,v in sorted(reasons.items())) or 'None'; manual_text='\n'.join(f'{ch}: {manual_reasons[ch]}' for ch in manual_additions) or 'None'
 report.add_text('PyPREP, manual channel QC, and reference',f'PyPREP was run on the full continuous EEG after 100-Hz low-pass.\nPyPREP suggestions:\n{reason_text}\nUser-rejected channels and reasons:\n{manual_text}\nChannels rescued in raw browser: {fmt_channels(browser_rescues)}\nChannels rescued after rejected-channel PSD: {fmt_channels(rescue_psd)}\nFINAL bad channels: {fmt_channels(final)}\nFinal good EEG channels: {fmt_channels(good_eeg)}\nRereferencing: {reref_label}. Rereferencing is not performed unless explicitly requested with --rereference avg.','All-channel preprocessing')
 (qc_dir(root,s)/'P01_pyprep_segment_epoch_reref.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
