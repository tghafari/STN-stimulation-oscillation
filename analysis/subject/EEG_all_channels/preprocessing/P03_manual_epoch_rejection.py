"""P03: manual bad-trial rejection, optional final bad-channel marking, and final PSD QC."""
from __future__ import annotations
import argparse,json
import matplotlib.pyplot as plt
import mne
import numpy as np
from pipeline_config import CONDITIONS,qc_dir,resolve_project_root,stage_path
from all_channel_report import participant_report,figure_dir,fmt_channels
def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',required=True);p.add_argument('--session',default='01');p.add_argument('--task',default='SpAtt');p.add_argument('--run',default='01');p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);p.add_argument('--n-channels',type=int,default=20);return p.parse_args()
def welch(ep):
 n=min(int(2*ep.info['sfreq']),len(ep.times));return ep.compute_psd(fmin=0.1,fmax=min(100.,ep.info['sfreq']/2.),method='welch',n_fft=n),n
def mean_psd(spec):return spec.get_data().mean(axis=(0,1))
def stimulation_qc(clean,report,figs,s):
 stim,nostim=clean['stim'],clean['no-stim'];common=[ch for ch in stim.copy().pick('eeg').ch_names if ch in nostim.ch_names and ch not in stim.info['bads'] and ch not in nostim.info['bads']]
 ss,_=welch(stim.copy().pick(common));ns,_=welch(nostim.copy().pick(common));f=ss.freqs;eps=np.finfo(float).tiny;sdb=10*np.log10(np.maximum(mean_psd(ss),eps));ndb=10*np.log10(np.maximum(mean_psd(ns),eps));fig,ax=plt.subplots(figsize=(10,5));ax.plot(f,ndb,label='no-stim');ax.plot(f,sdb,label='stim');ax.set_xlabel('Frequency (Hz)');ax.set_ylabel('Mean EEG PSD (dB)');ax.set_title(f'sub-{s}: stimulation artifact QC');ax.legend();ax.grid(True,alpha=.25);ax.set_xticks(np.arange(0,int(np.floor(f.max()/10))*10+1,10));ax.set_xlim(0,min(100.,float(f.max())));fig.tight_layout();report.add_figure(fig,str(figs/'P03_stimulation_artifact_QC.png'),'Stimulation artifact QC: stim versus no-stim PSD',f'Mean Welch PSD across epochs and the same {len(common)} good EEG channels. This is diagnostic only.','Stimulation artifact QC');return {'common_good_eeg_channels':common,'n_common_good_eeg_channels':len(common),'qc_only_no_correction':True}
def main():
 a=parse_args();s=a.subject.removeprefix('sub-');root=resolve_project_root(a.platform,a.project_root);report=participant_report(root,s);figs=figure_dir(root,s);epochs={};audit={'subject':f'sub-{s}','conditions':{}}
 for c in CONDITIONS:epochs[c]=mne.read_epochs(stage_path(root,s,a.session,a.task,a.run,c,'ica','epo'),preload=True)
 eeg=epochs['stim'].copy().pick('eeg').ch_names;existing=sorted(set().union(*(set(epochs[c].info['bads']) for c in CONDITIONS)));before={}
 for c in CONDITIONS:
  before[c]=len(epochs[c]);good=[x for x in epochs[c].copy().pick('eeg').ch_names if x not in epochs[c].info['bads']];epochs[c].plot(picks=good,n_channels=min(a.n_channels,len(good)),block=True,title=f'sub-{s} {c}: manual bad-trial QC')
 while True:
  add=input('Additional EEG channels to mark BAD after epoch inspection (space/comma separated, Enter for none): ').replace(',',' ').split();unknown=[x for x in add if x not in eeg]
  if not unknown:break
  print('Unknown EEG channels:',unknown)
 new=sorted(set(add)-set(existing));reasons={}
 for ch in new:
  while not (r:=input(f'Reason for marking {ch} bad: ').strip()):print('Please enter a reason.')
  reasons[ch]=r
 final=sorted(set(existing)|set(add));clean={}
 for c in CONDITIONS:
  epochs[c].info['bads']=[x for x in final if x in epochs[c].ch_names];out=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo');epochs[c].save(out,overwrite=True);clean[c]=epochs[c].copy();spec,n=welch(epochs[c]);fig=spec.plot(show=False);report.add_figure(fig,str(figs/f'P03_{c}_final_epoch_PSD.png'),f'{c}: PSD of final cleaned epochs',f'Welch PSD; retained epochs {len(epochs[c])}; manually rejected {before[c]-len(epochs[c])}.','All-channel EEG preprocessing');plt.close(fig);audit['conditions'][c]={'n_epochs_before_manual_rejection':before[c],'n_epochs_after_manual_rejection':len(epochs[c]),'final_bad_channels':final,'final_psd':{'method':'welch','fmin_hz':0.1,'fmax_hz':min(100.,epochs[c].info['sfreq']/2.),'n_fft':n}}
 audit['additional_bad_channels_at_epoch_qc']=new;audit['additional_bad_channel_reasons']=reasons;audit['final_bad_channels']=final;audit['stimulation_artifact_qc']=stimulation_qc(clean,report,figs,s);(qc_dir(root,s)/'P03_manual_epoch_rejection.json').write_text(json.dumps(audit,indent=2)+'\n');report.add_text('P03 manual epoch and channel QC',f'Bad trials were rejected manually. Additional bad channels: {fmt_channels(new)}. Reasons: '+('; '.join(f'{k}: {v}' for k,v in reasons.items()) or 'None')+f'. Final bad channels: {fmt_channels(final)}.','All-channel EEG preprocessing');print(f'Updated PDF: {report.pdf_fname}')
if __name__=='__main__':main()
