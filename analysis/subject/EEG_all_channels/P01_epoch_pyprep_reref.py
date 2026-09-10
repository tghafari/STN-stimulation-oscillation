"""Stage 1: cue epoch definition, PyPREP bad-channel QC, and average rereference.

PyPREP is run on continuous pre-reference EEG. Cue epochs (-0.5..1.6 s) are
defined before cleaning. PyPREP suggestions are reviewed by the researcher; the
union of final bad channels across stim/no-stim is then marked in both conditions.
Bad channels are not interpolated or dropped. Average reference therefore uses
only retained good EEG electrodes. Text, PSD figures, decisions and output paths
are appended to the participant PDF.
"""
from __future__ import annotations
import argparse, json
import mne
from pyprep.find_noisy_channels import NoisyChannels
from pipeline_config import CONDITIONS,qc_dir,resolve_project_root,segmented_raw_path,stage_path
from all_channel_report import participant_report,figure_dir,fmt_channels
EVENT_DICT={'cue_onset_right':1,'cue_onset_left':2,'trial_onset':3,'stim_onset':4,'catch_onset':5,'dot_onset_right':6,'dot_onset_left':7,'response_press_onset':8,'block_onset':20,'block_end':21,'experiment_end':30,'new_stim_segment':99999}

def parse_args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--subject',required=True); p.add_argument('--session',default='01'); p.add_argument('--task',default='SpAtt'); p.add_argument('--run',default='01'); p.add_argument('--platform',choices=['mac','bluebear'],default='mac'); p.add_argument('--project-root',default=None); p.add_argument('--line-freq',type=float,default=50.); return p.parse_args()

def pyprep_reasons(raw):
 eeg=raw.copy().pick('eeg');
 if eeg.get_montage() is None: eeg.set_montage('standard_1020',on_missing='warn')
 noisy=NoisyChannels(eeg,random_state=42); reasons={}; errors=[]
 for label,func in [('deviation',noisy.find_bad_by_deviation),('high-frequency noise',noisy.find_bad_by_hfnoise),('correlation',noisy.find_bad_by_correlation),('RANSAC',noisy.find_bad_by_ransac)]:
  try: func()
  except Exception as exc: errors.append(f'{label}: {type(exc).__name__}: {exc}')
 for attr,why in {'bad_by_deviation':'deviation','bad_by_hf_noise':'high-frequency noise','bad_by_correlation':'correlation','bad_by_ransac':'RANSAC','bad_by_nan':'NaN/flat data','bad_by_SNR':'poor signal-to-noise ratio'}.items():
  for ch in getattr(noisy,attr,[]) or []: reasons.setdefault(str(ch),[]).append(why)
 for ch in noisy.get_bads(): reasons.setdefault(str(ch),[]).append('PyPREP overall decision')
 if errors: reasons['__detector_errors__']=errors
 return {ch:sorted(set(v)) for ch,v in reasons.items()}

def define_epochs(raw):
 events,ids=mne.events_from_annotations(raw,event_id=EVENT_DICT); cue={k:ids[k] for k in ('cue_onset_right','cue_onset_left') if k in ids}
 if not cue: raise RuntimeError('No cue events found.')
 return mne.Epochs(raw,events,cue,tmin=-.5,tmax=1.6,baseline=None,detrend=1,proj=True,picks='all',reject=None,reject_by_annotation=False,preload=True,event_repeated='merge')

def main():
 a=parse_args(); subject=a.subject.removeprefix('sub-'); root=resolve_project_root(a.platform,a.project_root); report=participant_report(root,subject); figs=figure_dir(root,subject)
 raws={}; eps={}; audit={'subject':f'sub-{subject}','conditions':{},'common_bad_channels':[]}; union=set()
 report.add_text('P01 method: epoching, PyPREP and rereferencing','Cue-locked epochs are defined from -0.5 to 1.6 s with cue onset at 0 s, no baseline correction, linear detrending, and no automatic epoch rejection. PyPREP noisy-channel detectors (deviation, high-frequency noise, correlation and RANSAC) are run on continuous pre-rereference EEG. The researcher reviews the suggestions. The union of bad channels across stimulation conditions is applied to both conditions. Bad channels remain marked and are not interpolated/dropped. Average reference is calculated from the remaining good EEG channels.','All-channel EEG preprocessing')
 for condition in CONDITIONS:
  infile=segmented_raw_path(root,subject,a.session,a.task,a.run,condition)
  if not infile.exists(): raise FileNotFoundError(f'Missing input: {infile}')
  raw=mne.io.read_raw_fif(infile,preload=True,verbose=True); raw.info['line_freq']=raw.info.get('line_freq') or a.line_freq
  if raw.get_montage() is None: raw.set_montage('standard_1020',on_missing='warn')
  ep=define_epochs(raw); reasons=pyprep_reasons(raw); errors=reasons.pop('__detector_errors__',[]); suggested=sorted(reasons); raw.info['bads']=sorted(set(raw.info['bads'])|set(suggested))
  psd=raw.compute_psd(fmin=.5,fmax=min(100.,raw.info['sfreq']/2)).plot(show=False); report.add_figure(psd,str(figs/f'P01_{condition}_pre_reref_PSD.png'),f'{condition}: PSD before rereferencing',f'Continuous EEG before rereferencing. PyPREP suggestions are marked bad: {fmt_channels(suggested)}.','All-channel EEG preprocessing')
  raw.compute_psd(fmin=.5,fmax=min(100.,raw.info['sfreq']/2)).plot(); print(f'\n{condition}: PyPREP suggests {suggested or "None"}')
  additions=input('Additional bad EEG channels (space-separated, Enter for none): ').strip().split(); removals=input('PyPREP channels you believe are GOOD (space-separated, Enter for none): ').strip().split()
  final=(set(raw.info['bads'])|set(additions)|set(suggested))-set(removals); final &= set(raw.ch_names); union.update(final)
  reason_text='\n'.join(f'{ch}: {", ".join(vals)}' for ch,vals in sorted(reasons.items())) or 'None'
  report.add_text(f'{condition}: PyPREP and manual channel decision',f'Number of cue epochs defined: {len(ep)}\nPyPREP suggested channels: {fmt_channels(suggested)}\nPyPREP reasons:\n{reason_text}\nDetector errors: {"; ".join(errors) if errors else "None"}\nManual additions: {fmt_channels(additions)}\nManual removals from PyPREP suggestions: {fmt_channels(removals)}\nFinal bad channels for this condition: {fmt_channels(sorted(final))}', 'All-channel EEG preprocessing')
  audit['conditions'][condition]={'input':str(infile),'n_epochs_defined':len(ep),'pyprep_reasons':reasons,'pyprep_detector_errors':errors,'manual_additions':additions,'manual_removals':removals,'condition_bad_channels':sorted(final)}; raws[condition]=raw; eps[condition]=ep
 common=sorted(union); audit['common_bad_channels']=common; report.add_text('Common bad-channel set used for both conditions',f'Common bad channels: {fmt_channels(common)}\nThese channels are marked bad in both stim and no-stim. They are NOT interpolated or dropped. This preserves channel-specific missingness for later group averages.','All-channel EEG preprocessing')
 for condition in CONDITIONS:
  raw,ep=raws[condition],eps[condition]; present=[ch for ch in common if ch in raw.ch_names]; raw.info['bads']=present; ep.info['bads']=[ch for ch in present if ch in ep.ch_names]; raw.set_eeg_reference(ref_channels='average',projection=False); ep.set_eeg_reference(ref_channels='average',projection=False)
  psd=raw.compute_psd(fmin=.5,fmax=min(100.,raw.info['sfreq']/2)).plot(show=False); report.add_figure(psd,str(figs/f'P01_{condition}_post_reref_PSD.png'),f'{condition}: PSD after average rereference',f'Average reference calculated excluding marked bad channels: {fmt_channels(present)}.','All-channel EEG preprocessing')
  ro=stage_path(root,subject,a.session,a.task,a.run,condition,'reref','raw'); eo=stage_path(root,subject,a.session,a.task,a.run,condition,'reref','epo'); raw.save(ro,overwrite=True); ep.save(eo,overwrite=True); audit['conditions'][condition].update({'rereferenced_raw':str(ro),'rereferenced_epochs':str(eo)})
 audit_file=qc_dir(root,subject)/'P01_bad_channels_and_rereference.json'; audit_file.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8'); report.add_text('P01 saved outputs',f'QC audit: {audit_file}\nRereferenced raw and epoch FIF files are listed above in the audit JSON.','All-channel EEG preprocessing'); print(f'Updated PDF: {report.pdf_fname}')
if __name__=='__main__': main()
