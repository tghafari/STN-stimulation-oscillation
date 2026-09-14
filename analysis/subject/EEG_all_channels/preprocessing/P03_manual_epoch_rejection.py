"""P03: manual bad-trial rejection, optional additional bad-channel marking, and spectral QC.

During manual epoch inspection the user may identify additional bad EEG channels.
Those channels are marked bad (not dropped or interpolated) in BOTH stimulation
conditions so downstream ERP/TFR comparisons use a consistent sensor set. Final
PSDs use Welch. The stimulation-artifact spectral comparison is diagnostic only.
"""
from __future__ import annotations
import argparse,json
import matplotlib.pyplot as plt
import mne
import numpy as np
from pipeline_config import CONDITIONS,qc_dir,resolve_project_root,stage_path
from all_channel_report import participant_report,figure_dir,fmt_channels

def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',required=True);p.add_argument('--session',default='01');p.add_argument('--task',default='SpAtt');p.add_argument('--run',default='01');p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);p.add_argument('--n-channels',type=int,default=20);return p.parse_args()
def welch_psd(epochs):
 n_fft=min(int(2*epochs.info['sfreq']),len(epochs.times));return epochs.compute_psd(fmin=0.1,fmax=min(100.,epochs.info['sfreq']/2.),method='welch',n_fft=n_fft),n_fft
def mean_linear_psd(spectrum):
 data=spectrum.get_data()
 if data.ndim!=3:raise RuntimeError(f'Expected (epochs, channels, frequencies), got {data.shape}.')
 return data.mean(axis=(0,1))
def ask_channels(prompt,valid):
 while True:
  vals=input(prompt).strip().replace(',',' ').split();unknown=[x for x in vals if x not in valid]
  if not unknown:return sorted(set(vals))
  print(f'Unknown/non-EEG channels: {unknown}. Please enter EEG channel names exactly as shown.')
def ask_reasons(channels):
 reasons={}
 for ch in channels:
  while not (r:=input(f'Reason for marking {ch} bad at manual epoch QC: ').strip()):print('Please enter a reason so this decision is auditable.')
  reasons[ch]=r
 return reasons
def stimulation_artifact_qc(clean_epochs,report,figs,subject):
 stim=clean_epochs['stim'];nostim=clean_epochs['no-stim'];common=[ch for ch in stim.copy().pick('eeg').ch_names if ch in nostim.ch_names and ch not in stim.info['bads'] and ch not in nostim.info['bads']]
 if not common:raise RuntimeError('No common good EEG channels available for stimulation-artifact PSD QC.')
 ss,_=welch_psd(stim.copy().pick(common));ns,_=welch_psd(nostim.copy().pick(common))
 if not np.allclose(ss.freqs,ns.freqs):raise RuntimeError('Stim and no-stim PSD frequency bins do not match.')
 freqs=ss.freqs;sp=mean_linear_psd(ss);npow=mean_linear_psd(ns);eps=np.finfo(float).tiny;sdb=10*np.log10(np.maximum(sp,eps));ndb=10*np.log10(np.maximum(npow,eps));ratio=10*np.log10(np.maximum(sp,eps)/np.maximum(npow,eps))
 fig,(a1,a2)=plt.subplots(2,1,figsize=(10,8),sharex=True);a1.plot(freqs,ndb,label='no-stim');a1.plot(freqs,sdb,label='stim');a1.set_ylabel('Mean EEG PSD (dB)');a1.set_title(f'sub-{subject}: stimulation artifact QC');a1.legend();a1.grid(True,alpha=.25);a2.plot(freqs,ratio);a2.axhline(0,linestyle='--',linewidth=1);a2.set_xlabel('Frequency (Hz)');a2.set_ylabel('Stim / no-stim (dB)');a2.grid(True,alpha=.25);mt=int(np.floor(freqs.max()/10.)*10);a2.set_xticks(np.arange(0,mt+1,10));a2.set_xlim(0,min(100.,float(freqs.max())));fig.tight_layout()
 report.add_figure(fig,str(figs/'P03_stimulation_artifact_QC.png'),'Stimulation artifact QC: stim versus no-stim PSD',f'Mean Welch PSD across epochs and the same {len(common)} good EEG channels; bottom is 10*log10(stim/no-stim). Diagnostic only.','Stimulation artifact QC')
 return {'common_good_eeg_channels':common,'n_common_good_eeg_channels':len(common),'ratio_definition':'10*log10(mean_stim_linear_power / mean_no_stim_linear_power)','qc_only_no_correction':True}
def main():
 a=parse_args();s=a.subject.removeprefix('sub-');root=resolve_project_root(a.platform,a.project_root);report=participant_report(root,s);figs=figure_dir(root,s);audit={'subject':f'sub-{s}','conditions':{}}
 epochs={}
 for c in CONDITIONS:
  p=stage_path(root,s,a.session,a.task,a.run,c,'ica','epo')
  if not p.exists():raise FileNotFoundError(f'Missing ICA-cleaned epochs: {p}')
  epochs[c]=mne.read_epochs(p,preload=True)
 # Additional channel decisions are made once and propagated to both conditions.
 all_eeg=epochs[CONDITIONS[0]].copy().pick('eeg').ch_names;existing=sorted(set().union(*(set(epochs[c].info['bads']) for c in CONDITIONS)))
 print('\nExisting bad channels from P01:',existing or 'None')
 print('During P03 you may add channels that only became clearly bad while inspecting epochs.')
 # First browse each condition using currently good channels and reject bad trials.
 before_counts={}
 for c in CONDITIONS:
  before_counts[c]=len(epochs[c]);good=[ch for ch in epochs[c].copy().pick('eeg').ch_names if ch not in epochs[c].info['bads']]
  print('\n'+'='*72);print(f'sub-{s} / {c}: MANUAL BAD-TRIAL REJECTION');print('You may also note any consistently bad channels for the prompt after both browsers close.');print('='*72)
  epochs[c].plot(picks=good,n_channels=min(a.n_channels,len(good)),block=True,title=f'sub-{s} {c}: manual bad-trial QC')
 additional=ask_channels('\nAdditional EEG channels to mark BAD after epoch inspection (space/comma separated, Enter for none): ',all_eeg);new_add=[ch for ch in additional if ch not in existing];reasons=ask_reasons(new_add)
 final_bads=sorted(set(existing)|set(additional));print('FINAL bad channels after P03:',final_bads or 'None')
 clean={}
 for c in CONDITIONS:
  epochs[c].info['bads']=[ch for ch in final_bads if ch in epochs[c].ch_names];after=len(epochs[c]);out=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo');epochs[c].save(out,overwrite=True);clean[c]=epochs[c].copy();spec,n_fft=welch_psd(epochs[c]);fig=spec.plot(show=False);report.add_figure(fig,str(figs/f'P03_{c}_final_epoch_PSD.png'),f'{c}: PSD of final cleaned epochs',f'Welch PSD after manual trial QC and final bad-channel marking. Retained epochs: {after}; rejected: {before_counts[c]-after}.','All-channel EEG preprocessing');plt.close(fig)
  audit['conditions'][c]={'input_epochs':str(stage_path(root,s,a.session,a.task,a.run,c,'ica','epo')),'final_epochs':str(out),'n_epochs_before_manual_rejection':before_counts[c],'n_epochs_after_manual_rejection':after,'n_epochs_manually_rejected':before_counts[c]-after,'final_bad_channels':list(epochs[c].info['bads']),'final_psd':{'method':'welch','fmin_hz':0.1,'fmax_hz':min(100.,epochs[c].info['sfreq']/2.),'n_fft':n_fft}}
 report.add_text('P03 manual epoch and channel QC',f'ICA-cleaned cue epochs were visually inspected and bad trials were rejected manually.\nBad channels entering P03: {fmt_channels(existing)}\nAdditional channels marked bad during epoch QC: {fmt_channels(new_add)}\nReasons: '+('; '.join(f'{ch}: {why}' for ch,why in reasons.items()) or 'None')+f'\nFinal bad channels applied identically to stim and no-stim: {fmt_channels(final_bads)}. Channels were marked bad, not interpolated or physically dropped.','All-channel EEG preprocessing')
 audit['additional_bad_channels_at_epoch_qc']=new_add;audit['additional_bad_channel_reasons']=reasons;audit['final_bad_channels']=final_bads;audit['stimulation_artifact_qc']=stimulation_artifact_qc(clean,report,figs,s);(qc_dir(root,s)/'P03_manual_epoch_rejection.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');report.add_text('Preprocessing complete','All-channel preprocessing completed through manual bad-trial and final bad-channel QC.','All-channel EEG preprocessing');print(f'\nUpdated PDF: {report.pdf_fname}')
if __name__=='__main__':main()
