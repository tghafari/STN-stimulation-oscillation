"""P02: ICA on concatenated rereferenced stim + no-stim epochs.

The rereferenced stim and no-stim Epochs are concatenated ONLY for fitting one
shared ICA decomposition. A filtered/resampled fitting copy is used (1-40 Hz,
200 Hz). The same fitted ICA solution is then applied separately to the original
full-resolution stim epochs and no-stim epochs. FastICA uses 19 components by
default and every fitted component topography is written to the participant PDF.
"""
from __future__ import annotations
import argparse,json
import mne
from mne.preprocessing import ICA
from pipeline_config import CONDITIONS,qc_dir,resolve_project_root,stage_path
from all_channel_report import participant_report,figure_dir,fmt_channels
def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--subject',required=True); p.add_argument('--session',default='01'); p.add_argument('--task',default='SpAtt'); p.add_argument('--run',default='01'); p.add_argument('--platform',choices=['mac','bluebear'],default='mac'); p.add_argument('--project-root',default=None); p.add_argument('--n-components',type=int,default=19); p.add_argument('--random-state',type=int,default=97); return p.parse_args()
def parse_components(text,n):
    vals=sorted(set(int(x) for x in text.replace(',',' ').split())) if text.strip() else []
    if any(x<0 or x>=n for x in vals):raise ValueError(f'components must be 0..{n-1}')
    return vals
def main():
    a=parse_args(); s=a.subject.removeprefix('sub-'); root=resolve_project_root(a.platform,a.project_root); report=participant_report(root,s); figs=figure_dir(root,s)
    epochs={}
    for c in CONDITIONS:
        p=stage_path(root,s,a.session,a.task,a.run,c,'reref','epo')
        if not p.exists():raise FileNotFoundError(f'Missing P01 epochs for {c}: {p}')
        epochs[c]=mne.read_epochs(p,preload=True)
    fit=mne.concatenate_epochs([epochs[c].copy() for c in CONDITIONS],add_offset=True,on_mismatch='warn'); fit.pick('eeg'); fit.resample(200); fit.filter(1,40)
    good=[ch for ch in fit.ch_names if ch not in fit.info['bads']]; n=min(a.n_components,max(2,len(good)-1))
    print(f'Fitting one ICA with {n} components on {len(fit)} concatenated rereferenced stim + no-stim epochs.')
    ica=ICA(method='fastica',random_state=a.random_state,n_components=n,max_iter='auto'); ica.fit(fit,picks='eeg',reject_by_annotation=True)
    component_figs=ica.plot_components(picks=range(ica.n_components_),show=False); component_figs=component_figs if isinstance(component_figs,(list,tuple)) else [component_figs]
    for i,fig in enumerate(component_figs):report.add_figure(fig,str(figs/f'P02_all_ICA_components_{i+1}.png'),f'All ICA components ({i+1}/{len(component_figs)})',f'All {ica.n_components_} fitted component topographies. No component is omitted.','ICA')
    ica.plot_sources(fit,block=True,title=f'sub-{s}: inspect ICA components')
    while True:
        try:excluded=parse_components(input('ICA components to EXCLUDE (space/comma separated, Enter for none): '),ica.n_components_);break
        except ValueError as exc:print(exc)
    if excluded:
        props=ica.plot_properties(fit,picks=excluded,show=False); props=props if isinstance(props,(list,tuple)) else [props]
        for i,fig in enumerate(props):report.add_figure(fig,str(figs/f'P02_ICA_exclusion_properties_{i+1}.png'),f'ICA exclusion diagnostic {i+1}',f'Manually selected component(s): {excluded}.','ICA')
        if input(f'Apply exclusions {excluded}? [y/N]: ').strip().lower() not in {'y','yes'}:raise RuntimeError('ICA stopped by user.')
    ica.exclude=excluded; ica.save(qc_dir(root,s)/f'sub-{s}_ica.fif',overwrite=True)
    audit={'subject':f'sub-{s}','fit_on':'concatenated rereferenced stim + no-stim epochs','fit_resample_hz':200,'fit_filter_hz':[1,40],'n_components':int(ica.n_components_),'excluded_components':excluded,'conditions':{}}
    for c in CONDITIONS:
        clean=epochs[c].copy(); ica.apply(clean); out=stage_path(root,s,a.session,a.task,a.run,c,'ica','epo'); clean.save(out,overwrite=True); audit['conditions'][c]={'n_epochs':len(clean),'bad_channels':list(clean.info['bads'])}
    report.add_text('ICA method and decision',f'One shared ICA decomposition was fitted on concatenated rereferenced stim + no-stim cue epochs.\nICA fitting copy: resampled to 200 Hz and filtered 1-40 Hz.\nMethod: FastICA; random state: {a.random_state}.\nComponents fitted: {ica.n_components_}.\nManually excluded components: {excluded or "None"}.\nThe fitted solution was then applied separately to the original stim epochs and no-stim epochs.\nBad channels retained in info: {fmt_channels(epochs[CONDITIONS[0]].info["bads"])}','ICA')
    (qc_dir(root,s)/'P02_ica_decisions.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
