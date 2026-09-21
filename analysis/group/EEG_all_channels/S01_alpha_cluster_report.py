#!/usr/bin/env python
"""Alpha cluster-permutation analysis with a self-contained PDF report.

Tests stimulation minus no-stimulation alpha power (8-12 Hz) from 0.2-1.2 s.
Default power is condition-wise percent baseline (-0.3 to -0.1 s), matching the
subject-level descriptive TFR analysis.

Report includes:
- analysis details and manuscript-style methods section;
- primary sensor x time spatio-temporal cluster results;
- individual sensors with FDR-significant time clusters;
- 8-channel posterior/occipital ROI average;
- 3-channel posterior ROI average (PO3, POz, PO4);
- figures shading significant temporal clusters and scalp maps of significant
  spatio-temporal clusters.

ROI inference is participant-level: channels are averaged WITHIN participant
before the paired one-sample permutation test.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats
from mne.stats import permutation_cluster_1samp_test,combine_adjacency,fdr_correction

HERE=Path(__file__).resolve().parent; ANALYSIS_DIR=HERE.parents[1]
for p in (ANALYSIS_DIR/"subject"/"EEG_all_channels",ANALYSIS_DIR/"utils"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from pipeline_config import CONDITIONS,resolve_project_root,stage_path
from pdf_report import ParticipantPDF

DEFAULT_ALPHA=(8.,12.);WINDOW=(.2,1.2)
FREQS=np.arange(2.,31.,1.);N_CYCLES=FREQS/2.;TIME_BANDWIDTH=2.;DECIM=2
ROI8=("PO3","POz","PO4","O1","Oz","O2","PO7","PO8");ROI3=("PO3","POz","PO4")
PTHRESH=.05;CLUSTER_FORMING_ALPHA=.10

def args():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--subjects",nargs="+",required=True)
 p.add_argument("--session",default="01")
 p.add_argument("--task",default="SpAtt")
 p.add_argument("--run",default="01")
 p.add_argument("--platform",choices=["mac","bluebear"],default="mac")
 p.add_argument("--project-root",default=None)
 p.add_argument("--n-jobs",type=int,default=4)
 p.add_argument("--alpha",nargs=2,type=float,metavar=("FMIN","FMAX"),default=DEFAULT_ALPHA,help="Frequency range in Hz to average before cluster testing, e.g. --alpha 8 12")
 p.add_argument("--seed",type=int,default=42)
 a=p.parse_args()
 if a.alpha[0]>=a.alpha[1]:
  p.error("--alpha requires FMIN < FMAX")
 if a.alpha[0]<FREQS.min() or a.alpha[1]>FREQS.max():
  p.error(f"--alpha must lie within {FREQS.min():g}-{FREQS.max():g} Hz")
 return a
def load(root,s,a):
 d={}
 for c in CONDITIONS:
  f=stage_path(root,s,a.session,a.task,a.run,c,"clean","epo")
  if not f.exists():raise FileNotFoundError(f)
  e=mne.read_epochs(f,preload=True,verbose=False);k=[x for x in ("cue_onset_right","cue_onset_left") if x in e.event_id];d[c]=e[k] if k else e
 return d
def good(x):
 st,no=x["stim"],x["no-stim"];return[ch for ch in st.copy().pick("eeg").ch_names if ch in no.ch_names and ch not in st.info["bads"] and ch not in no.info["bads"]]
def alpha(ep,chs,a):
 t=ep.copy().pick(chs).compute_tfr(method="multitaper",freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,return_itc=False,average=True,decim=DECIM,n_jobs=a.n_jobs,verbose=False)
 fi=(t.freqs>=a.alpha[0])&(t.freqs<=a.alpha[1])
 ti=(t.times>=WINDOW[0])&(t.times<=WINDOW[1])
 return t.data[:,fi][:,:,ti].mean(1),t.times[ti],t.info
def one_d(X,seed):
 return permutation_cluster_1samp_test(X,n_permutations="all",threshold=None,tail=0,adjacency=None,out_type="mask",seed=seed,verbose=False)
def sig_1d(T,cl,p,t):
 out=[]
 for i,(m,pv) in enumerate(zip(cl,p)):
  if pv<=PTHRESH:
   ix=np.where(np.asarray(m,bool))[0];out.append({"cluster":i,"p":float(pv),"t_start":float(t[ix].min()),"t_end":float(t[ix].max()),"peak_t":float(t[ix[np.argmax(np.abs(T[ix]))]]),"peak_stat":float(T[ix][np.argmax(np.abs(T[ix]))])})
 return out
def shade(ax,rows):
 for r in rows:ax.axvspan(r["t_start"],r["t_end"],alpha=.18)
def roi_plot(st,no,t,rows,title,ylabel):
 fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);ms=st.mean(0);mn=no.mean(0);sem_s=st.std(0,ddof=1)/np.sqrt(len(st));sem_n=no.std(0,ddof=1)/np.sqrt(len(no));ax.plot(t,mn,label="No stimulation");ax.fill_between(t,mn-sem_n,mn+sem_n,alpha=.15);ax.plot(t,ms,label="Stimulation");ax.fill_between(t,ms-sem_s,ms+sem_s,alpha=.15);shade(ax,rows);ax.axvline(0,color="k",ls="--",lw=.8);ax.set_xlabel("Time (s)");ax.set_ylabel(ylabel);ax.set_title(title);ax.legend();return fig
def sensor_plot(st,no,t,ch,rows):
 return roi_plot(st,no,t,rows,f"{ch}: alpha power, stimulation vs no stimulation","Alpha power")
def participant_difference_traces(diff,t,subjects,title):
 fig,ax=plt.subplots(figsize=(11,6),constrained_layout=True)
 for sub,y in zip(subjects,diff): ax.plot(t,y,linewidth=.9,alpha=.55,label=f"sub-{sub}")
 mean=diff.mean(0);sem=diff.std(0,ddof=1)/np.sqrt(len(diff))
 ax.plot(t,mean,color="k",linewidth=3,label="Group mean");ax.fill_between(t,mean-sem,mean+sem,color="k",alpha=.12)
 ax.axhline(0,color="k",linewidth=.8,linestyle=":");ax.set_xlabel("Time (s)");ax.set_ylabel("Stim - no-stim alpha power");ax.set_title(title);ax.legend(ncol=4,fontsize=7)
 return fig
def participant_window_effects(diff,subjects,title):
 values=diff.mean(1);fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);x=np.arange(len(subjects))
 ax.axhline(0,color="k",linewidth=.8,linestyle=":");ax.scatter(x,values,s=45,zorder=3);ax.plot(x,values,linewidth=.7,alpha=.45);ax.axhline(values.mean(),color="k",linewidth=2,label=f"Group mean = {values.mean():.4g}")
 ax.set_xticks(x);ax.set_xticklabels([f"sub-{z}" for z in subjects],rotation=60,ha="right");ax.set_ylabel(f"Mean stim - no-stim alpha, {WINDOW[0]:g}-{WINDOW[1]:g} s");ax.set_title(title);ax.legend()
 return fig,values
def cluster_threshold(n): return float(stats.t.ppf(1-CLUSTER_FORMING_ALPHA/2,n-1))
def summarize_clusters(T,clusters,pvals,t,chs=None):
 out=[]
 for i,(m,pv) in enumerate(zip(clusters,pvals)):
  mask=np.asarray(m,bool)
  if chs is None:
   ix=np.where(mask)[0]
   if not ix.size: continue
   vals=T[ix];out.append(dict(cluster=i,p=float(pv),cluster_stat=float(vals.sum()),n_samples=int(ix.size),n_sensors=1,t_start=float(t[ix].min()),t_end=float(t[ix].max()),sensors="ROI"))
  else:
   ti,ci=np.where(mask)
   if not ti.size: continue
   vals=T[mask];used=[chs[j] for j in sorted(set(ci))];out.append(dict(cluster=i,p=float(pv),cluster_stat=float(vals.sum()),n_samples=int(mask.sum()),n_sensors=len(used),t_start=float(t[ti].min()),t_end=float(t[ti].max()),sensors=", ".join(used)))
 return out
def spatial_rows(T,clusters,pvals,t,chs):
 out=[]
 for i,(m,p) in enumerate(zip(clusters,pvals)):
  if p>PTHRESH:continue
  ti,ci=np.where(np.asarray(m,bool));used=[chs[j] for j in sorted(set(ci))];out.append({"cluster":i,"p":float(p),"t_start":float(t[ti].min()),"t_end":float(t[ti].max()),"sensors":used,"mask":np.asarray(m,bool)})
 return out
def spatial_topomap(row,T,info,chs):
 mask=row["mask"];vals=np.zeros(len(chs));active=np.zeros(len(chs),bool)
 for ci in range(len(chs)):
  x=T[:,ci][mask[:,ci]]
  if x.size:vals[ci]=x.mean();active[ci]=True
 fig,ax=plt.subplots(figsize=(7,6));im,_=mne.viz.plot_topomap(vals,info,axes=ax,show=False,contours=0,mask=active,mask_params=dict(marker="o",markerfacecolor="none",markeredgecolor="k",linewidth=0,markersize=8));ax.set_title(f"Spatial cluster p={row['p']:.4f}, {row['t_start']:.3f}-{row['t_end']:.3f} s");fig.colorbar(im,ax=ax,label="Mean cluster t statistic");return fig

def main():
 a=args();subs=[s.removeprefix("sub-") for s in a.subjects];root=resolve_project_root(a.platform,a.project_root);ep={s:load(root,s,a) for s in subs};g={s:good(ep[s]) for s in subs};order=ep[subs[0]]["stim"].copy().pick("eeg").ch_names;common=[ch for ch in order if all(ch in g[s] for s in subs)]
 if not common:raise RuntimeError("No EEG sensor is good in both conditions for every participant.")
 ST=[];NO=[]
 for s in subs:
  st,t,info=alpha(ep[s]["stim"],common,a);no,t2,_=alpha(ep[s]["no-stim"],common,a)
  if not np.allclose(t,t2):raise RuntimeError("Time mismatch "+s)
  ST.append(st);NO.append(no)
 ST=np.asarray(ST);NO=np.asarray(NO);D=ST-NO
 out=root/"derivatives"/"reports"/"group"/"EEG_all_channels_alpha_cluster";figs=out/"figures";figs.mkdir(parents=True,exist_ok=True);report=ParticipantPDF(str(out),"alpha_cluster_"+"_".join(subs))
 section="Alpha cluster permutation"
 posterior_survived=[ch for ch in ROI8 if ch in common]
 posterior_removed=[ch for ch in ROI8 if ch not in common]
 survived_text=(
  f"Whole-sample cluster tests require a sensor to be good in BOTH stimulation conditions for EVERY participant.\n"
  f"All surviving EEG channels (n={len(common)}): {', '.join(common)}\n"
  f"Posterior/occipital ROI candidates surviving: {', '.join(posterior_survived) if posterior_survived else 'None'}\n"
  f"Posterior/occipital ROI candidates removed by the intersection: {', '.join(posterior_removed) if posterior_removed else 'None'}"
 )
 report.add_text("Channels surviving whole-sample intersection",survived_text,section)
 details=f"N={len(subs)} paired participants; contrast=stimulation minus no stimulation calculated from UNBASELINED power; selected alpha range={a.alpha[0]:g}-{a.alpha[1]:g} Hz averaged across frequency; inferential window={WINDOW[0]:g}-{WINDOW[1]:g} s. TFR: multitaper, 2-30 Hz in 1-Hz steps, n_cycles=f/2, time-bandwidth={TIME_BANDWIDTH:g}, FFT=True, ITC=False, trial average=True, decimation={DECIM}. No baseline correction is applied before the stimulation-minus-no-stimulation contrast. Two-sided one-sample cluster permutation tests are applied to within-participant differences with all possible sign flips supported by MNE; cluster-forming threshold uses two-sided pointwise alpha=0.10 (0.05 in each tail); family-wise cluster significance p<=0.05. Whole-scalp inference uses temporal adjacency plus EEG sensor adjacency and only sensors good in both conditions for every participant. Sensor-wise cluster p-values are additionally Benjamini-Hochberg FDR corrected across all sensor clusters. ROI channels are averaged within participant before permutation testing."
 report.add_text("Analysis details",details,section)
 methods=("Alpha-band stimulation effects were assessed using paired cluster-based permutation tests on participant-level stimulation-minus-no-stimulation power. Time-frequency power was estimated with multitaper convolution and alpha power was defined as the mean from the user-specified frequency range ("+f"{a.alpha[0]:g}-{a.alpha[1]:g} Hz). Statistical inference was restricted a priori to 0.2-1.2 s after cue onset. Stimulation-minus-no-stimulation contrasts were calculated from unbaselined power; no baseline correction was applied before contrast formation. The primary whole-scalp analysis clustered samples jointly across time and neighboring EEG sensors. Complementary analyses tested temporal clusters at individual sensors, with false-discovery-rate correction across sensor-level clusters, and a priori posterior averages comprising eight posterior/occipital sensors and PO3/POz/PO4. Each ROI used an ROI-specific complete-case sample: participants were included only if every required ROI channel was retained as good in both stimulation conditions. Sensor values were averaged within each included participant before group inference, ensuring participants rather than sensors were the unit of observation.")
 report.add_text("Manuscript-style statistical analysis",methods,section)
 # spatial
 sadj,names=mne.channels.find_ch_adjacency(info,ch_type="eeg")
 if list(names)!=list(common):
  ix=[names.index(ch) for ch in common];sadj=sadj[ix][:,ix]
 t_thresh=cluster_threshold(len(subs))
 X=D.transpose(0,2,1)
 T,cl,pv,_=permutation_cluster_1samp_test(X,n_permutations="all",threshold=t_thresh,tail=0,adjacency=combine_adjacency(len(t),sadj),out_type="mask",seed=a.seed,n_jobs=a.n_jobs,verbose=True)
 all_spatial=summarize_clusters(T,cl,pv,t,common);srows=spatial_rows(T,cl,pv,t,common);top_spatial=sorted(all_spatial,key=lambda r:r["p"])[:10]
 sigtxt="No significant spatio-temporal clusters." if not srows else "\n".join(f"Cluster {r['cluster']}: p={r['p']:.4f}; {r['t_start']:.3f}-{r['t_end']:.3f} s; sensors: {', '.join(r['sensors'])}" for r in srows)
 diagnostic="\n".join(f"Cluster {r['cluster']}: p={r['p']:.4f}; cluster_stat={r['cluster_stat']:.3f}; n_samples={r['n_samples']}; n_sensors={r['n_sensors']}; {r['t_start']:.3f}-{r['t_end']:.3f} s; sensors: {r['sensors']}" for r in top_spatial) if top_spatial else "None"
 report.add_text("Whole-scalp spatio-temporal results",f"Actual cluster-forming threshold: +/-{t_thresh:.4f} (two-sided pointwise alpha={CLUSTER_FORMING_ALPHA:.2f}; 0.05 in each tail; df={len(subs)-1}).\n{sigtxt}\n\nStrongest observed clusters, including non-significant clusters:\n{diagnostic}",section)
 print(f"Cluster-forming threshold: +/-{t_thresh:.4f}")
 for r in all_spatial: print(f"Spatial cluster {r['cluster']}: stat={r['cluster_stat']:.4f}, p={r['p']:.4f}, samples={r['n_samples']}, sensors={r['n_sensors']}, time={r['t_start']:.3f}-{r['t_end']:.3f}")
 for r in srows:report.add_figure(spatial_topomap(r,T,info,common),str(figs/f"spatial_cluster_{r['cluster']}.png"),f"Significant spatio-temporal cluster {r['cluster']}",f"Black circles mark sensors participating in the cluster at one or more samples. Cluster p={r['p']:.4f}; time extent {r['t_start']:.3f}-{r['t_end']:.3f} s.",section)
 # sensorwise
 rec=[];allp=[];raw_store={}
 for ci,ch in enumerate(common):
  Ts,cs,ps,_=one_d(D[:,ci,:],a.seed);raw_store[ch]=(Ts,cs,ps)
  for k,(m,p) in enumerate(zip(cs,ps)):
   ix=np.where(np.asarray(m,bool))[0];rec.append({"sensor":ch,"cluster":k,"p_uncorrected":float(p),"t_start":float(t[ix].min()),"t_end":float(t[ix].max())});allp.append(float(p))
 if allp:
  rej,q=fdr_correction(np.asarray(allp),alpha=PTHRESH,method="indep")
  for r,z,qq in zip(rec,rej,q):r["significant_fdr"]=bool(z);r["p_fdr"]=float(qq)
 sigsensor=sorted({r["sensor"] for r in rec if r.get("significant_fdr")})
 report.add_text("Sensors with significant temporal clusters",("None after FDR correction." if not sigsensor else ", ".join(sigsensor)),section)
 for ch in sigsensor:
  rows=[r for r in rec if r["sensor"]==ch and r.get("significant_fdr")]
  report.add_figure(sensor_plot(ST[:,common.index(ch),:],NO[:,common.index(ch),:],t,rows),str(figs/f"sensor_{ch}_significant_clusters.png"),f"{ch}: significant alpha temporal cluster(s)","Lines are group means; shading around lines is SEM; vertical shaded regions are FDR-significant temporal clusters.",section)
 # ROI analyses use ROI-specific complete-case subject samples.
 # A participant contributes only when EVERY channel in that ROI is good in BOTH conditions.
 roi_results={}
 for label,requested in (("8-channel posterior/occipital ROI",ROI8),("3-channel posterior ROI",ROI3)):
  roi_subs=[sub for sub in subs if all(ch in g[sub] for ch in requested)]
  excluded_subs=[sub for sub in subs if sub not in roi_subs]
  if len(roi_subs)<2:
   report.add_text(label+" results",f"Insufficient complete-case participants. Required channels: {', '.join(requested)}. Included: {', '.join('sub-'+x for x in roi_subs) if roi_subs else 'None'}. Excluded because at least one required ROI channel was bad/missing: {', '.join('sub-'+x for x in excluded_subs) if excluded_subs else 'None'}.",section)
   roi_results[label]={"required_channels":list(requested),"included_subjects":roi_subs,"excluded_subjects":excluded_subs,"rows":[]};continue
  roi_st=[];roi_no=[]
  for sub in roi_subs:
   st_sub,rt,rinfo=alpha(ep[sub]["stim"],list(requested),a);no_sub,rt2,_=alpha(ep[sub]["no-stim"],list(requested),a)
   if not np.allclose(rt,rt2):raise RuntimeError("ROI time mismatch "+sub)
   roi_st.append(st_sub.mean(0));roi_no.append(no_sub.mean(0))
  st=np.asarray(roi_st);no=np.asarray(roi_no);roi_diff=st-no
  roi_thresh=cluster_threshold(len(roi_subs))
  Tr,cr,pr,_=permutation_cluster_1samp_test(roi_diff,n_permutations="all",threshold=roi_thresh,tail=0,adjacency=None,out_type="mask",seed=a.seed,verbose=False)
  rows=sig_1d(Tr,cr,pr,rt);all_roi=summarize_clusters(Tr,cr,pr,rt)
  roi_results[label]={"required_channels":list(requested),"included_subjects":roi_subs,"excluded_subjects":excluded_subs,"n_subjects":len(roi_subs),"rows":rows,"all_clusters":all_roi}
  top_roi=sorted(all_roi,key=lambda r:r["p"])[:10]
  text=f"Required channels: {', '.join(requested)}. N={len(roi_subs)} complete-case participants. Included: {', '.join('sub-'+x for x in roi_subs)}. Excluded because at least one required ROI channel was bad/missing: {', '.join('sub-'+x for x in excluded_subs) if excluded_subs else 'None'}. "+("No significant temporal clusters." if not rows else " ".join(f"Cluster {r['cluster']}: p={r['p']:.4f}, {r['t_start']:.3f}-{r['t_end']:.3f} s." for r in rows))
  text+=f"\nActual ROI cluster-forming threshold: +/-{roi_thresh:.4f} (df={len(roi_subs)-1}; two-sided alpha={CLUSTER_FORMING_ALPHA:.2f}, 0.05 each tail).\nStrongest observed clusters, including non-significant:\n"+("\n".join(f"Cluster {r['cluster']}: p={r['p']:.4f}; stat={r['cluster_stat']:.3f}; n_samples={r['n_samples']}; {r['t_start']:.3f}-{r['t_end']:.3f} s" for r in top_roi) if top_roi else "None")
  report.add_text(label+" results",text,section)
  tag="ROI8" if requested==ROI8 else "ROI3"
  report.add_figure(roi_plot(st,no,rt,rows,label,"Alpha power"),str(figs/f"{tag}_alpha_clusters.png"),label+": stimulation vs no stimulation",f"Complete-case ROI sample N={len(roi_subs)}. All required ROI channels are retained in every included participant. Lines are group means; ribbons are SEM. Shaded regions are significant clusters.",section)
  report.add_figure(participant_difference_traces(roi_diff,rt,roi_subs,label+": individual participant differences"),str(figs/f"{tag}_individual_difference_traces.png"),label+": individual stim - no-stim traces",f"Only participants retaining every required ROI channel are shown (N={len(roi_subs)}).",section)
  effect_fig,effect_values=participant_window_effects(roi_diff,roi_subs,label+": participant mean effects");report.add_figure(effect_fig,str(figs/f"{tag}_participant_mean_effects.png"),label+": participant-level mean 0.2-1.2 s effects",f"Complete-case ROI N={len(roi_subs)}. Each point is one participant's mean stim - no-stim alpha effect.",section)
  roi_results[label]["participant_window_mean_effects"]={f"sub-{sub}":float(v) for sub,v in zip(roi_subs,effect_values)}
  print(f"\n{label}: N={len(roi_subs)} included; excluded={excluded_subs}")
  for r in all_roi:print(f"{label} cluster {r['cluster']}: stat={r['cluster_stat']:.4f}, p={r['p']:.4f}, samples={r['n_samples']}, time={r['t_start']:.3f}-{r['t_end']:.3f}")
 # audit
 audit={"subjects":subs,"n_subjects":len(subs),"common_good_channels":common,"posterior_roi_channels_surviving_intersection":posterior_survived,"posterior_roi_channels_removed_by_intersection":posterior_removed,"cluster_forming_threshold_t":t_thresh,"cluster_forming_alpha_two_sided":CLUSTER_FORMING_ALPHA,"cluster_forming_alpha_each_tail":CLUSTER_FORMING_ALPHA/2,"all_spatiotemporal_clusters":all_spatial,"alpha_hz":list(a.alpha),"window_s":WINDOW,"baseline_s":None,"contrast_power":"unbaselined","ROI8":roi_results.get("8-channel posterior/occipital ROI"),"ROI3":roi_results.get("3-channel posterior ROI"),"significant_sensorwise_fdr":sigsensor,"spatiotemporal":[{k:v for k,v in r.items() if k!="mask"} for r in srows]}
 (out/"alpha_cluster_report_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
 with (out/"sensorwise_clusters_all.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=["sensor","cluster","p_uncorrected","t_start","t_end","significant_fdr","p_fdr"]);w.writeheader();w.writerows(rec)
 print("\nPDF report:",report.pdf_fname)
 print("Significant spatial clusters:",len(srows));print("FDR-significant sensors:",sigsensor or "none")
 for k,v in roi_results.items():print(k,":",len(v["rows"]),"significant cluster(s)")

if __name__=="__main__":main()
