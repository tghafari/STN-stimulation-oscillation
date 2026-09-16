def roi_erp(no,st,times,title):
    """Plot ROI ERP waveforms, accepting either (n_times,) or (1, n_times)."""
    no=np.asarray(no).squeeze();st=np.asarray(st).squeeze();times=np.asarray(times).squeeze()
    if no.ndim != 1 or st.ndim != 1 or times.ndim != 1:
        raise ValueError(f'ROI ERP arrays must be 1-D after squeeze: times={times.shape}, no={no.shape}, stim={st.shape}')
    if len(times) != len(no) or len(times) != len(st):
        raise ValueError(f'ROI ERP time/data length mismatch: times={len(times)}, no={len(no)}, stim={len(st)}')
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);ax.plot(times,no*1e6,label='No stimulation');ax.plot(times,st*1e6,label='Stimulation');ax.axvline(0,color='k',ls='--',lw=.7);ax.set_xlim(ERP_TMIN,ERP_TMAX);ax.set_xlabel('Time (s)');ax.set_ylabel('Amplitude (uV)');ax.set_title(title);ax.legend();return fig
