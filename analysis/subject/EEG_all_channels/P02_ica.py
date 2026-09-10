"""Stage 2: fit, inspect and apply ICA after rereferencing.

One FastICA decomposition is fit to concatenated stim/no-stim good EEG after
resampling a copy to 200 Hz and filtering 1-40 Hz. Component rejection is always
a human decision. Component maps, selected-component diagnostics and the complete
analysis settings are appended to the participant PDF.
"""
from __future__ import annotations
import argparse,json,mne
from mne.preprocessing import ICA
from pipeline_config import CONDITIONS,qc_dir,resolve_project_root,stage_path
from all_channel_report import participant_report,figure_dir,fmt_channels

def parse_args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--subject',required=True); p.add_argument('--session',default='01'); p.add_argument('--task',default='SpAtt'); p.add_argument('--run',default='01'); p.add_argument('--platform',choices=['mac','bluebear'],default='mac'); p.add_argument('--project-root',default=None); p.add_argument('--n-components',type=int,default=30); p.add_argument('--random-state',type=int,default=97); return p.parse_args()
def parse_component_list(text,n):
 cleaned=text.replace(',',' ').strip(); vals=[] if not cleaned else sorted(set(int(x) for x in cleaned.split())); bad=[x for x in vals if x<0 or x>=n]
 if bad: raise ValueError(f'Indices outside 0..{n-1}: {bad}')
 return vals
def main():
 a=parse_args(); subject=a.subject.removeprefix('sub-'); root=resolve_project_root(a.platform,a.project_root); report=participant_report(root,subject); figs=figure_dir(root,subject); raws={}; epochs={}
 for c in CONDITIONS:
  rp=stage_path(root,subject,a.session,a.task,a.run,c,'reref','raw'); ep=stage_path(root,subject,a.session,a.task,a.run,c,'reref','epo')
  if not rp.exists() or not ep.exists(): raise FileNotFoundError(f'Missing Stage-1 output for {c}')
  raws[c]=mne.io.read_raw_fif(rp,preload=True); epochs[c]=mne.read_epochs(ep,preload=True)
 fit=mne.concatenate_raws([raws[c].copy() for c in CONDITIONS],on_mismatch='warn'); fit.pick('eeg'); good=[ch for ch in fit.ch_names if ch not in fit.info['bads']]
 if len(good)<3: raise RuntimeError('Too few good EEG channels for ICA.')
 fit.resample(200); fit.filter(1.,40.); n=min(a.n_components,max(2,len(good)-1)); ica=ICA(method='fastica',random_state=a.random_state,n_components=n,max_iter='auto',verbose=True); ica.fit(fit,reject_by_annotation=True,picks='eeg')
 report.add_text('P02 method: independent component analysis',f'ICA method: FastICA\nFit data: concatenated stim + no-stim continuous EEG after average rereference\nBad EEG channels excluded from fit: {fmt_channels(fit.info["bads"])}\nICA-only resampling: 200 Hz\nICA-only filter: 1-40 Hz\nRandom state: {a.random_state}\nRequested components: {a.n_components}\nActual components: {ica.n_components_}\nThe ICA is fitted on the filtered/resampled copy but applied to the full-resolution rereferenced data. Component exclusion is selected manually by the researcher.','All-channel EEG preprocessing')
 compfig=ica.plot_components(show=False); report.add_figure(compfig,str(figs/'P02_ICA_all_components.png'),'ICA component maps','All fitted ICA component scalp maps before manual exclusion.','All-channel EEG preprocessing'); ica.plot_components(show=True); ica.plot_sources(fit,block=True,title=f'sub-{subject}: inspect ICA components')
 while True:
  try: excluded=parse_component_list(input('\nICA components to EXCLUDE (e.g. 0 1 5, Enter none): '),ica.n_components_); break
  except Exception as exc: print(f'Invalid component list: {exc}')
 if excluded:
  propfigs=ica.plot_properties(fit,picks=excluded,show=False)
  for idx,fig in zip(excluded,propfigs): report.add_figure(fig,str(figs/f'P02_ICA_component_{idx}_properties.png'),f'ICA component {idx}: diagnostic properties','Component selected for possible exclusion; inspect topography, spectrum, epoch image and time course.','All-channel EEG preprocessing')
  ica.plot_properties(fit,picks=excluded); ans=input('Apply this ICA exclusion list? [y/N]: ').strip().lower()
  if ans not in {'y','yes'}: raise RuntimeError('ICA stage stopped before applying components.')
  selected=ica.plot_components(picks=excluded,show=False); report.add_figure(selected,str(figs/'P02_ICA_removed_components.png'),'ICA components removed',f'Manually confirmed excluded components: {excluded}.','All-channel EEG preprocessing')
 report.add_text('P02 manual ICA decision',f'Excluded ICA components: {excluded if excluded else "None"}\nThis list was entered manually after visual inspection. No automatic ICA component rejection was used.','All-channel EEG preprocessing')
 ica.exclude=excluded; ica_file=qc_dir(root,subject)/f'sub-{subject}_ica.fif'; ica.save(ica_file,overwrite=True); audit={'subject':f'sub-{subject}','method':'fastica','random_state':a.random_state,'fit_filter_hz':[1.,40.],'fit_resample_hz':200,'requested_n_components':a.n_components,'actual_n_components':int(ica.n_components_),'excluded_components':excluded,'ica_file':str(ica_file),'conditions':{}}
 for c in CONDITIONS:
  rc=raws[c].copy(); ec=epochs[c].copy(); ica.apply(rc); ica.apply(ec); ro=stage_path(root,subject,a.session,a.task,a.run,c,'ica','raw'); eo=stage_path(root,subject,a.session,a.task,a.run,c,'ica','epo'); rc.save(ro,overwrite=True); ec.save(eo,overwrite=True); audit['conditions'][c]={'ica_raw':str(ro),'ica_epochs':str(eo),'bad_channels_retained_in_info':list(ec.info['bads'])}
 audit_file=qc_dir(root,subject)/'P02_ica_decisions.json'; audit_file.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8'); report.add_text('P02 saved outputs',f'ICA solution: {ica_file}\nICA decision audit: {audit_file}\nICA-cleaned raw and epoch FIF files were saved separately for stim and no-stim.','All-channel EEG preprocessing'); print(f'Updated PDF: {report.pdf_fname}')
if __name__=='__main__': main()
