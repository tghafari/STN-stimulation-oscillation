#!/usr/bin/env python
"""QC: scalp-layout alpha-power time courses for stimulation vs no stimulation.

For EACH EEG sensor independently, include every participant for whom that sensor
is present and marked good in BOTH stimulation conditions. Therefore N may differ
by sensor. No participant is removed because another sensor is bad.

For each included participant/condition:
- final cleaned cue epochs are loaded;
- attention-left/right trials are combined;
- multitaper TFR: 2-30 Hz, 1-Hz steps, n_cycles=f/2, time-bandwidth=2,
  FFT=True, ITC=False, trial-average=True, decim=2;
- percent baseline correction (-0.3 to -0.1 s) is applied separately to each condition;
- user-selected alpha frequencies are averaged;
- stimulation and no-stimulation alpha time courses are retained.

Each scalp panel shows the across-participant mean +/- SEM for that sensor and
prints its sensor-specific N. This script is descriptive QC only: no statistics.
"""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np

HERE=Path(__file__).resolve().parent;ANALYSIS_DIR=HERE.parents[1]
SUBJECT_DIR=ANALYSIS_DIR/"subject"/"EEG_all_channels"
if str(SUBJECT_DIR) not in sys.path:sys.path.insert(0,str(SUBJECT_DIR))
from pipeline_config import CONDITIONS,resolve_project_root,stage_path

FREQS=np.arange(2.,31.,1.);N_CYCLES=FREQS/2.;TIME_BANDWIDTH=2.;DECIM=2
PLOT_WINDOW=(-.3,1.4);BASELINE=(-.3,-.1);DEFAULT_ALPHA=(8.,12.)

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--subjects",nargs="+",required=True);p.add_argument("--session",default="01");p.add_argument("--task",default="SpAtt");p.add_argument("--run",default="01");p.add_argument("--platform",choices=["mac","bluebear"],default="mac");p.add_argument("--project-root",default=None);p.add_argument("--n-jobs",type=int,default=4);p.add_argument("--alpha",nargs=2,type=float,metavar=("FMIN","FMAX"),default=DEFAULT_ALPHA);a=p.parse_args()
 if a.alpha[0]>=a.alpha[1] or a.alpha[0]<FREQS.min() or a.alpha[1]>FREQS.max():p.error(f"--alpha must satisfy {FREQS.min():g} <= FMIN < FMAX <= {FREQS.max():g}")
 return a

def load(root,s,a):
 d={}
 for c in CONDITIONS:
  f=stage_path(root,s,a.session,a.task,a.run,c,"clean","epo")
  if not f.exists():raise FileNotFoundError(f)
  print(f"sub-{s} / {c}: {f}")
  e=mne.read_epochs(f,preload=True,verbose=False);keys=[k for k in ("cue_onset_right","cue_onset_left") if k in e.event_id];d[c]=e[keys] if keys else e
 return d

def good(pair):
 st,no=pair["stim"],pair["no-stim"];return[ch for ch in st.copy().pick("eeg").ch_names if ch in no.ch_names and ch not in st.info["bads"] and ch not in no.info["bads"]]

def alpha_curve(ep,ch,a):
 t=ep.copy().pick([ch]).compute_tfr(method="multitaper",freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,return_itc=False,average=True,decim=DECIM,n_jobs=a.n_jobs,verbose=False)
 fi=(t.freqs>=a.alpha[0])&(t.freqs<=a.alpha[1]);ti=(t.times>=PLOT_WINDOW[0])&(t.times<=PLOT_WINDOW[1])
 return t.data[0,fi][:,ti].mean(0),t.times[ti]

def template_info(epochs,channels):
 for pair in epochs.values():
  inf=pair["stim"].copy().pick("eeg").info
  if all(ch in inf.ch_names for ch in channels):return inf
 return next(iter(epochs.values()))["stim"].copy().pick("eeg").info

def scalp_cells(info,channels):
 pos=info.get_montage().get_positions()["ch_pos"];xy={c:np.asarray(pos[c])[:2] for c in channels if c in pos}
 channels=[c for c in channels if c in xy];xs=np.array([xy[c][0] for c in channels]);ys=np.array([xy[c][1] for c in channels])
 nr=11;nc=11;free={(r,c) for r in range(nr) for c in range(nc)};xmin,xmax=xs.min(),xs.max();ymin,ymax=ys.min(),ys.max();targets={c:((ymax-xy[c][1])/max(ymax-ymin,1e-12)*(nr-1),(xy[c][0]-xmin)/max(xmax-xmin,1e-12)*(nc-1)) for c in channels};cells={}
 for ch in sorted(channels,key=lambda c:targets[c]):
  tr,tc=targets[ch];cell=min(free,key=lambda q:(q[0]-tr)**2+(q[1]-tc)**2);cells[ch]=cell;free.remove(cell)
 return nr,nc,cells

def main():
 a=args();subs=[s.removeprefix("sub-") for s in a.subjects];root=resolve_project_root(a.platform,a.project_root);epochs={s:load(root,s,a) for s in subs};goods={s:good(epochs[s]) for s in subs}
 channels=[]
 for s in subs:
  for ch in goods[s]:
   if ch not in channels:channels.append(ch)
 by_ch={ch:[s for s in subs if ch in goods[s]] for ch in channels}
 info=template_info(epochs,channels);nr,nc,cells=scalp_cells(info,channels)
 out=root/"derivatives"/"reports"/"group"/"EEG_all_channels_alpha_QC";out.mkdir(parents=True,exist_ok=True)
 fig,axs=plt.subplots(nr,nc,figsize=(28,24));[ax.axis("off") for ax in axs.ravel()]
 rows=[]
 for ch in channels:
  if ch not in cells:continue
  st=[];no=[];times=None
  for s in by_ch[ch]:
   ys,t=alpha_curve(epochs[s]["stim"],ch,a);yn,t2=alpha_curve(epochs[s]["no-stim"],ch,a)
   if not np.allclose(t,t2):raise RuntimeError(f"Time mismatch sub-{s} {ch}")
   st.append(ys);no.append(yn);times=t
  st=np.asarray(st);no=np.asarray(no);n=len(st);ms=st.mean(0);mn=no.mean(0);ss=st.std(0,ddof=1)/np.sqrt(n) if n>1 else np.zeros_like(ms);sn=no.std(0,ddof=1)/np.sqrt(n) if n>1 else np.zeros_like(mn)
  ax=axs[cells[ch]];ax.axis("on");ax.plot(times,mn,label="No stim",linewidth=1);ax.fill_between(times,mn-sn,mn+sn,alpha=.18);ax.plot(times,ms,label="Stim",linewidth=1);ax.fill_between(times,ms-ss,ms+ss,alpha=.18);ax.axvline(0,color="k",ls="--",lw=.5);ax.set_title(f"{ch} (n={n})",fontsize=8);ax.tick_params(labelsize=5);ax.set_xlim(*PLOT_WINDOW)
  rows.append({"channel":ch,"n_subjects":n,"subjects":";".join("sub-"+s for s in by_ch[ch])})
 fig.suptitle(f"Alpha power QC: stimulation vs no stimulation, {a.alpha[0]:g}-{a.alpha[1]:g} Hz, mean +/- SEM\nPercent baseline -0.3 to -0.1 s applied separately within each condition; sensor-specific participant N",fontsize=17)
 handles=[plt.Line2D([],[],label="No stimulation"),plt.Line2D([],[],label="Stimulation")];fig.legend(handles=handles,loc="upper right");fig.subplots_adjust(left=.025,right=.97,bottom=.03,top=.93,wspace=.55,hspace=.65)
 png=out/f"alpha_{a.alpha[0]:g}-{a.alpha[1]:g}Hz_percent_baseline_all_sensors_scalp_mean_SEM.png";fig.savefig(png,dpi=200,bbox_inches="tight");plt.close(fig)
 with (out/f"alpha_{a.alpha[0]:g}-{a.alpha[1]:g}Hz_subjects_by_sensor.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=["channel","n_subjects","subjects"]);w.writeheader();w.writerows(rows)
 print("\nSaved QC figure:",png);print("Each panel uses all participants retaining that sensor in both conditions.")
 for r in rows:print(f"{r['channel']}: n={r['n_subjects']}")

if __name__=="__main__":main()
