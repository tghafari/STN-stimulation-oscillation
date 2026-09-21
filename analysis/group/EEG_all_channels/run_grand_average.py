#!/usr/bin/env python
"""Run G04 with BrainVision-cap/MNE-layout report views. Numerical analysis unchanged."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne
import G04_complete_grand_average_report as pipeline

def baseline_percent_safe(x):
    data=np.asarray(x.data[0])
    mask=(x.times>=pipeline.BASELINE[0])&(x.times<=pipeline.BASELINE[1])
    base=data[:,mask].mean(axis=1,keepdims=True)
    return 100.*(data-base)/np.where(np.abs(base)<np.finfo(float).eps,1.,base)
pipeline.baseline_percent=baseline_percent_safe

_orig_evoked=pipeline.make_evoked
_erp_calls=[]
def make_evoked_collect(ep,ch):
    x=_orig_evoked(ep,ch)
    _erp_calls.append((ch,x))
    return x
pipeline.make_evoked=make_evoked_collect

_last_scalp=None

def _find_brainvision_montage(info):
    """Prefer the project's BrainVision 64-channel cap file when available."""
    candidates=[]
    for base in [Path.cwd(),Path(__file__).resolve()]:
        for p in [base,*base.parents]:
            candidates.append(p/"data"/"data-organised"/"new-64.bvef")
    for p in candidates:
        if p.exists():
            try:
                montage=mne.channels.read_custom_montage(str(p))
                return montage,p
            except Exception:
                pass
    return info.get_montage(),None

def _layout_info(info):
    """Return Info carrying the BrainVision cap coordinates, without changing data."""
    montage,path=_find_brainvision_montage(info)
    if montage is None:
        return info,path
    out=info.copy()
    try:
        out.set_montage(montage,on_missing="ignore")
    except Exception:
        pass
    return out,path

def _make_layout(info):
    """Use MNE's EEG layout derived from the cap/montage coordinates."""
    try:
        return mne.channels.make_eeg_layout(info)
    except Exception:
        return None

def _layout_axes(fig, info, channels):
    """Create compact, non-overlapping sensor axes from MNE layout positions."""
    li=_make_layout(info)
    if li is None:
        return None
    by_name={name:i for i,name in enumerate(li.names)}
    positions=[]
    for ch in channels:
        if ch in by_name:
            positions.append((ch,li.pos[by_name[ch]]))
    if not positions:
        return None
    # Keep MNE's sensor geometry but shrink boxes uniformly so neighboring sensors
    # never touch. Positions are normalized to the plotting area.
    box_scale=0.62
    left,bottom,width,height=0.04,0.04,0.84,0.88
    axes={}
    for ch,p in positions:
        x,y,w,h=p[:4]
        w*=box_scale; h*=box_scale
        x=left+x*width
        y=bottom+y*height
        axes[ch]=fig.add_axes([x,y,w*width,h*height])
    return axes

def tfr_scalp_topo(data,info,title,v):
    plot_info,_=_layout_info(info)
    channels=list(data)
    fig=plt.figure(figsize=(21,19))
    axes=_layout_axes(fig,plot_info,channels)
    if axes is None:
        raise RuntimeError("Could not construct an MNE EEG layout from the BrainVision cap or attached montage.")
    for ch,ax in axes.items():
        arr,t=data[ch]
        ax.imshow(arr,origin="lower",aspect="auto",
                  extent=[t[0],t[-1],pipeline.FREQS[0],pipeline.FREQS[-1]],
                  cmap="RdBu_r",vmin=v[0] if v else None,vmax=v[1] if v else None)
        ax.axvline(0,color="k",ls="--",lw=.4)
        ax.set_title(ch,fontsize=7,pad=1)
        ax.tick_params(labelsize=4,length=2,pad=1)
    fig.suptitle(title,fontsize=14,y=.96)
    # Colorbar lives entirely outside the sensor-layout region.
    if v:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        cax=fig.add_axes([.91,.20,.018,.60])
        sm=ScalarMappable(norm=Normalize(*v),cmap="RdBu_r")
        sm.set_array([])
        fig.colorbar(sm,cax=cax,label="Value")
    return fig

def scalp_capture(data,info,title,v):
    global _last_scalp
    _last_scalp=(data,info,title,v)
    return tfr_scalp_topo(data,info,title,v)
pipeline.scalp=scalp_capture

def posterior_tfr(data,title,v):
    fig,axs=plt.subplots(2,4,figsize=(20,9))
    im=None
    for ax,ch in zip(axs.ravel(),pipeline.ROI8):
        if ch not in data:
            ax.axis("off"); continue
        arr,t=data[ch]
        im=ax.imshow(arr,origin="lower",aspect="auto",
                     extent=[t[0],t[-1],pipeline.FREQS[0],pipeline.FREQS[-1]],
                     cmap="RdBu_r",vmin=v[0] if v else None,vmax=v[1] if v else None)
        ax.axvline(0,color="k",ls="--",lw=.5); ax.set_title(ch)
    fig.subplots_adjust(left=.06,right=.88,bottom=.08,top=.90,wspace=.32,hspace=.35)
    fig.suptitle(title+": eight posterior sensors")
    if im is not None:
        cax=fig.add_axes([.91,.18,.015,.62])
        fig.colorbar(im,cax=cax,label="Value")
    return fig

def build_erp_groups():
    if not _erp_calls:
        return None
    n=len(_erp_calls)//2
    out={pipeline.CONDITIONS[0]:{},pipeline.CONDITIONS[1]:{}}
    for cond,calls in zip(pipeline.CONDITIONS,(_erp_calls[:n],_erp_calls[n:])):
        bucket={}
        for ch,x in calls:
            bucket.setdefault(ch,[]).append(x)
        for ch,xs in bucket.items():
            out[cond][ch]=(np.mean([z.data[0] for z in xs],0),xs[0].times)
    return out

def erp_scalp_topo(erp,info):
    allchs=[c for c in erp["stim"] if c in erp["no-stim"]]
    plot_info,_=_layout_info(info)
    fig=plt.figure(figsize=(21,19))
    axes=_layout_axes(fig,plot_info,allchs)
    if axes is None:
        raise RuntimeError("Could not construct an MNE EEG layout from the BrainVision cap or attached montage.")
    for ch,ax in axes.items():
        st,t=erp["stim"][ch]; no,_=erp["no-stim"][ch]
        ax.plot(t,no*1e6,lw=.75,label="No stimulation")
        ax.plot(t,st*1e6,lw=.75,label="Stimulation")
        ax.axvline(0,color="k",ls="--",lw=.4)
        ax.set_xlim(pipeline.ERP_TMIN,pipeline.ERP_TMAX)
        ax.set_title(ch,fontsize=7,pad=1); ax.tick_params(labelsize=4,length=2,pad=1)
    fig.subplots_adjust(left=.04,right=.96,bottom=.03,top=.93,wspace=.0,hspace=.0)
    fig.suptitle("Grand-average ERP: BrainVision-cap/MNE sensor layout",fontsize=14,y=.96)
    return fig

def posterior_erp(erp):
    fig,axs=plt.subplots(2,4,figsize=(20,9),constrained_layout=True)
    for ax,ch in zip(axs.ravel(),pipeline.ROI8):
        if ch not in erp["stim"] or ch not in erp["no-stim"]:
            ax.axis("off"); continue
        st,t=erp["stim"][ch]; no,_=erp["no-stim"][ch]
        ax.plot(t,no*1e6,label="No stimulation"); ax.plot(t,st*1e6,label="Stimulation")
        ax.axvline(0,color="k",ls="--",lw=.5); ax.set_xlim(pipeline.ERP_TMIN,pipeline.ERP_TMAX); ax.set_title(ch)
    axs[0,0].legend(); fig.suptitle("Grand-average ERP: eight posterior sensors")
    return fig

_BasePDF=pipeline.ParticipantPDF
class DetailPDF(_BasePDF):
    def __init__(self,*a,**k):
        super().__init__(*a,**k); self._erp_inserted=False
    def add_figure(self,fig,png,title,caption,section):
        global _last_scalp
        if section=="ERP" and title.startswith("ERP mean") and not self._erp_inserted:
            e=build_erp_groups()
            if e:
                info=_erp_calls[0][1].info
                super().add_figure(erp_scalp_topo(e,info),
                                   str(png).replace("ERP_mean","ERP_all_sensors_scalp"),
                                   "ERP: all sensors BrainVision-cap/MNE layout",
                                   "Sensor positions follow the BrainVision 64-channel cap when new-64.bvef is available, using MNE make_eeg_layout; individual axes are uniformly reduced to prevent overlap. Numerical values are unchanged.","ERP")
                super().add_figure(posterior_erp(e),
                                   str(png).replace("ERP_mean","ERP_8posterior"),
                                   "ERP: eight posterior sensors",
                                   "PO3, POz, PO4, O1, Oz, O2, PO7, PO8 in a 2 x 4 layout.","ERP")
            self._erp_inserted=True
        super().add_figure(fig,png,title,caption,section)
        if section.startswith("TFR - ") and title.endswith(" scalp") and _last_scalp is not None:
            data,info,stitle,v=_last_scalp
            super().add_figure(posterior_tfr(data,stitle,v),
                               str(png).replace("_scalp.png","_8posterior.png"),
                               stitle+": eight posterior sensors",
                               "Eight posterior sensors shown in 2 x 4 using the same already-computed arrays and scale.",section)
pipeline.ParticipantPDF=DetailPDF

if __name__=="__main__":
    pipeline.main()
