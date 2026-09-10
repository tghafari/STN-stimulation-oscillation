"""Stage 3: manual bad-trial rejection using all retained EEG channels.

ICA-cleaned epochs are inspected in MNE using all good EEG electrodes. Marked bad
channels remain in info['bads'] and are excluded from the browser. Final trial
counts, PSDs and butterfly ERPs are appended to the participant PDF.
"""
from __future__ import annotations
import argparse,json,mne
from pipeline_config import CONDITIONS,qc_dir,resolve_project_root,stage_path
from all_channel_report import participant_report,figure_dir,fmt_channels

def parse_args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--subject',required=True); p.add_argument('--session',default='01'); p.add_argument('--task',default='SpAtt'); p.add_argument('--run',default='01'); p.add_argument('--platform',choices=['mac','bluebear'],default='mac'); p.add_argument('--project-root',default=None); p.add_argument('--n-channels',type=int,default=20); return p.parse_args()
def main():
 a=parse_args(); subject=a.subject.removeprefix('sub-'); root=resolve_project_root(a.platform,a.project_root); report=participant_report(root,subject); figs=figure_dir(root,subject); audit={'subject':f'sub-{subject}','conditions':{}}
 report.add_text('P03 method: manual bad-trial rejection','ICA-cleaned cue epochs are visually inspected using all good EEG channels. Previously marked bad electrodes are not displayed. The researcher scrolls through the MNE epoch browser and manually marks contaminated trials. No amplitude threshold or automatic epoch rejection is applied at this stage. Bad channels remain marked rather than interpolated or dropped.','All-channel EEG preprocessing')
 for c in CONDITIONS:
  infile=stage_path(root,subject,a.session,a.task,a.run,c,'ica','epo')
  if not infile.exists(): raise FileNotFoundError(f'Missing ICA-cleaned epochs: {infile}')
  ep=mne.read_epochs(infile,preload=True); before=len(ep); bad=list(ep.info['bads']); good=[ch for ch in ep.copy().pick('eeg').ch_names if ch not in bad]
  if not good: raise RuntimeError('No good EEG channels remain.')
  print('\n'+'='*72); print(f'sub-{subject} / {c}: MANUAL BAD-TRIAL REJECTION'); print(f'Epochs before: {before}'); print(f'Bad channels excluded from display: {bad or "None"}'); print('='*72)
  ep.plot(picks=good,n_channels=min(a.n_channels,len(good)),block=True,title=f'sub-{subject} {c}: manual rejection using all good EEG channels'); after=len(ep); out=stage_path(root,subject,a.session,a.task,a.run,c,'clean','epo'); ep.save(out,overwrite=True)
  psd=ep.compute_psd(fmin=.5,fmax=min(100.,ep.info['sfreq']/2)).plot(show=False); report.add_figure(psd,str(figs/f'P03_{c}_final_epoch_PSD.png'),f'{c}: PSD of final cleaned epochs',f'After manual trial rejection. Retained epochs: {after}; manually rejected: {before-after}.','All-channel EEG preprocessing')
  ev=ep.copy().pick(good).average(); butterfly=ev.plot(spatial_colors=True,show=False); report.add_figure(butterfly,str(figs/f'P03_{c}_final_butterfly_ERP.png'),f'{c}: final cleaned-epoch butterfly ERP','Average across retained trials for all good EEG channels. This is a QC visualization, not the final inferential ERP analysis.','All-channel EEG preprocessing')
  report.add_text(f'{c}: manual trial-rejection result',f'Epochs before manual inspection: {before}\nEpochs after manual inspection: {after}\nEpochs manually rejected: {before-after}\nBad channels excluded from visual trial QC: {fmt_channels(bad)}\nNumber of good EEG channels inspected: {len(good)}\nFinal cleaned epochs: {out}','All-channel EEG preprocessing')
  audit['conditions'][c]={'input_epochs':str(infile),'final_epochs':str(out),'n_epochs_before_manual_rejection':before,'n_epochs_after_manual_rejection':after,'n_epochs_manually_rejected':before-after,'bad_channels_retained_in_info':list(ep.info['bads']),'good_eeg_channels_used_for_visual_trial_qc':good}
 audit_file=qc_dir(root,subject)/'P03_manual_epoch_rejection.json'; audit_file.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8'); report.add_text('Preprocessing complete',f'All-channel preprocessing completed through manual bad-trial rejection.\nFinal audit: {audit_file}\nThe participant PDF now contains the inclusion decision, PyPREP/channel QC, rereferencing, ICA decisions, and final trial-rejection QC.','All-channel EEG preprocessing'); print(f'\nUpdated PDF: {report.pdf_fname}')
if __name__=='__main__': main()
