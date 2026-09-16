"""Scalp-layout plotting helpers for G04.

Uses actual EEG montage x/y positions rather than channel list order. Axes are
placed directly in normalized figure coordinates, preserving left/right and
anterior/posterior relations. A dedicated right-side colorbar axis is reserved so
it can never overlap sensor panels.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def _sensor_axes(fig,channels,info,left=.045,right=.86,bottom=.07,top=.91,w=.105,h=.105):
    montage=info.get_montage()
    if montage is None:raise RuntimeError('EEG montage is required for anatomical scalp layout')
    pos=montage.get_positions().get('ch_pos',{})
    missing=[ch for ch in channels if ch not in pos]
    if missing:raise RuntimeError('Missing montage positions for: '+', '.join(missing))
    xy=np.asarray([np.asarray(pos[ch],float)[:2] for ch in channels]);x=xy[:,0];y=xy[:,1]
    xn=(x-x.min())/max(np.ptp(x),1e-12);yn=(y-y.min())/max(np.ptp(y),1e-12)
    # centers follow true montage coordinates; enough margin prevents clipping.
    cx=left+w/2+xn*((right-left)-w);cy=bottom+h/2+yn*((top-bottom)-h)
    return {ch:fig.add_axes([cx[i]-w/2,cy[i]-h/2,w,h]) for i,ch in enumerate(channels)}

def anatomical_tfr_scalp(data,channels,info,title,vlim,freqs):
    fig=plt.figure(figsize=(22,17));axes=_sensor_axes(fig,channels,info,w=.092,h=.095)
    for ch,ax in axes.items():
        if ch not in data:ax.axis('off');continue
        arr,t=data[ch];ax.imshow(arr,origin='lower',aspect='auto',extent=[t[0],t[-1],freqs[0],freqs[-1]],cmap='RdBu_r',vmin=vlim[0] if vlim else None,vmax=vlim[1] if vlim else None);ax.axvline(0,color='k',ls='--',lw=.45);ax.set_title(ch,fontsize=8,pad=1);ax.tick_params(labelsize=5,length=2)
    fig.suptitle(title,fontsize=16,y=.975)
    if vlim:
        # Dedicated colorbar outside the entire sensor-layout region.
        cax=fig.add_axes([.91,.20,.018,.60]);sm=ScalarMappable(norm=Normalize(vlim[0],vlim[1]),cmap='RdBu_r');sm.set_array([]);fig.colorbar(sm,cax=cax,label='Value')
    return fig

def anatomical_erp_scalp(group_erp,channels,info,erp_tmin,erp_tmax):
    fig=plt.figure(figsize=(22,17));axes=_sensor_axes(fig,channels,info,w=.092,h=.095)
    first=True
    for ch,ax in axes.items():
        if ch not in group_erp['stim'] or ch not in group_erp['no-stim']:ax.axis('off');continue
        times=group_erp['stim'][ch][1];ax.plot(times,group_erp['no-stim'][ch][0]*1e6,label='No stimulation',lw=.8);ax.plot(times,group_erp['stim'][ch][0]*1e6,label='Stimulation',lw=.8);ax.axvline(0,color='k',ls='--',lw=.45);ax.set_xlim(erp_tmin,erp_tmax);ax.set_title(ch,fontsize=8,pad=1);ax.tick_params(labelsize=5,length=2)
        if first:handles,labels=ax.get_legend_handles_labels();first=False
    if not first:fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.985,.965))
    fig.suptitle('Grand-average cue-locked ERP: all channels',fontsize=16,y=.975);return fig
