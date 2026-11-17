"""Generate force, motion, actuation, and ablation result figures."""
from functools import lru_cache
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from .labels import LABELS, AB, COL
DATA=OUT=None

STYLE=['-','--','-.',':',(0,(5,1,1,1)),(0,(4,1,1,1,1,1)),(0,(1,1)),(0,(7,2))]
AUDIT=[]
@lru_cache(maxsize=12)
def load(path):
    with np.load(DATA/path) as z: return z['trace'],z['reference']
def select(rows,method,tau):
    found=[r for r in rows if r['case']['method']==method and r['case']['contact_tau']==tau]
    assert len(found)==1,(method,tau,len(found));return found[0]
def new_axes(height=3.4):
    fig,axs=plt.subplots(2,1,figsize=(3.5,height),sharex=True,layout='constrained')
    fig.get_layout_engine().set(h_pad=.03,w_pad=.025,hspace=.025)
    for ax,name in zip(axs,['(a) More compliant contact','(b) Stiffer contact']):
        ax.set_title(name,loc='left',pad=3);ax.tick_params(labelsize=7.5,pad=2)
    return fig,axs
def legend(ax,ablation=False,ncol=None):
    obj=ax.legend(loc='upper center',ncol=ncol or (2 if ablation else 3),fontsize=6.6,
        frameon=False,columnspacing=.65,handlelength=1.9,handletextpad=.35,
        borderpad=.25,labelspacing=.28)
    for t in obj.get_texts():
        if t.get_text()=='MSFC':t.set_color(COL[0]);t.set_fontweight('bold')
    return obj
def note(ax,text,y=.70):
    ax.text(.025,y,text,transform=ax.transAxes,fontsize=6.7,color=COL[0],
        va='top',ha='left',bbox=dict(boxstyle='round,pad=.22',fc='white',ec=COL[0],lw=.6,alpha=.95))
def arrays(row,signal):
    a,ref=load(row['trace_path'])
    y={'feedback':a[:,14],'raw':a[:,1],'tangent':1000*(a[:,3]-a[:,7]),'torque':a[:,12]}[signal]
    mask=(a[:,0]>=1.2)&(np.arange(len(a))%20==0)
    if signal=='raw':mask=(a[:,0]>=4)&(a[:,0]<=4.01)
    assert np.isfinite(y[mask]).all() and np.isfinite(ref[mask]).all()
    return a[mask,0],y[mask],ref[mask]
def lines(ax,rows,tau,signal,ablation=False):
    names=AB if ablation else LABELS;expected={};src=[];previous=None
    for j,(method,label) in enumerate(names.items()):
        row=select(rows,method,tau);x,y,ref=arrays(row,signal)
        if previous is not None:
            assert np.array_equal(x,previous[0]) and np.array_equal(ref,previous[1])
        previous=(x,ref)
        ax.plot(x,y,label=label,color=COL[j],ls=STYLE[j],lw=1.15 if j==0 else .8,zorder=8 if j==0 else 3)
        expected[label]=(x,y);src.append(row['trace_path'])
    if signal in ['feedback','raw']:
        ax.plot(x,ref,color='black',ls=(0,(5,2)),lw=.9,label=r'$F_d$',zorder=7)
        expected[r'$F_d$']=(x,ref)
    ax.set_xlim((4,4.01) if signal=='raw' else (1.2,8.4))
    return expected,src

def check(ax,expected,src,name,tau):
    data={l.get_label():l for l in ax.lines if not l.get_label().startswith('_')}
    assert data.keys()==expected.keys(),(name,data.keys(),expected.keys())
    for label,(x,y) in expected.items():
        assert np.array_equal(data[label].get_xdata(),x) and np.array_equal(data[label].get_ydata(),y)
        assert np.all(np.diff(x)>0)
        lo,hi=ax.get_ylim();assert y.min()>=lo-1e-10 and y.max()<=hi+1e-10,(name,label,y.min(),y.max(),lo,hi)
    if ax.get_legend():
        assert {t.get_text() for t in ax.get_legend().get_texts()}==set(expected)
    # Quantify natural overlap rather than shifting curves to separate them.
    close={};methods=[k for k in expected if k!=r'$F_d$']
    tol={'comparison_tangent':.03,'comparison_torque':.03}.get(name,.05)
    for i,l in enumerate(methods):
        for m in methods[i+1:]:
            frac=float(np.mean(abs(expected[l][1]-expected[m][1])<tol))
            if frac>.5:close[l+' / '+m]=round(frac,3)
    AUDIT.append({'figure':name,'contact_tau':tau,'labels':list(expected),'samples_per_curve':len(next(iter(expected.values()))[0]),'source_paths':src,'legend_matches':True,'source_data_matches':True,'nonfinite_values':0,'clipped_main_curves':0,'overlap_threshold':tol,'pairs_overlapping_more_than_half':close})
def save(fig,name):
    fig.canvas.draw()
    fig.set_layout_engine('none')
    renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        for text in ax.texts:
            if text.get_bbox_patch() is None:continue
            box=text.get_bbox_patch().get_window_extent(renderer)
            if ax.get_legend() is not None:
                assert not box.overlaps(ax.get_legend().get_window_extent(renderer)),(name,'annotation overlaps legend')
            for line in ax.lines:
                points=ax.transData.transform(np.c_[line.get_xdata(),line.get_ydata()])
                covered=(points[:,0]>box.x0)&(points[:,0]<box.x1)&(points[:,1]>box.y0)&(points[:,1]<box.y1)
                assert not covered.any(),(name,'annotation covers data',text.get_text(),line.get_label())
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.025);plt.close(fig)
def inset(ax,rows,tau,signal,limits,rect):
    ia=ax.inset_axes(rect)
    for j,(m,label) in enumerate(LABELS.items()):
        x,y,_=arrays(select(rows,m,tau),signal);mask=(x>=limits[0])&(x<=limits[1])
        ia.plot(x[mask],y[mask],color=COL[j],ls=STYLE[j],lw=.8 if j==0 else .6,zorder=8 if j==0 else 3)
    ia.set_xlim(limits[:2]);ia.set_ylim(limits[2:]);ia.tick_params(labelsize=5.5,pad=1,length=2)
    ia.set_xticks([limits[0],limits[1]]);ia.set_yticks([limits[2],0,limits[3]])
    ia.set_title('Near-zero detail',fontsize=6,pad=1);ia.grid(alpha=.15)
    ax.add_patch(Rectangle((limits[0],limits[2]),limits[1]-limits[0],limits[3]-limits[2],fill=False,ec=COL[0],lw=.7,zorder=9))
    return ia

def render_response(signal,name,rows=None):
    if rows is None: rows=TRACES
    fig,axs=new_axes(3.5 if signal=='tangent' else 3.4)
    for ax,tau in zip(axs,[.01,.002]):
        expected,src=lines(ax,rows,tau,signal)
        yy=np.concatenate([a[1] for a in expected.values()]);lo,hi=yy.min(),yy.max();span=max(hi-lo,1)
        ax.set_ylim(lo-.06*span,hi+.90*span)
        ax.set_ylabel({'feedback':'Feedback force (N)','raw':'Contact force (N)','tangent':'Tangential error (mm)','torque':'Joint 1 torque (N m)'}[signal],fontsize=8)
        ours=select(rows,'reference_inverse',tau)
        if signal=='feedback':
            # This is a trajectory metric, not the 16-task average.
            note(ax,f"MSFC raw-force nMAE: {100*ours['nmae']:.2f}%",.66)
        elif signal=='raw':
            x,y=expected['MSFC'];indices=np.flatnonzero((x>=4.002)&(x<=4.0035));i=int(indices[np.argmax(y[indices])])
            ax.annotate('MSFC',xy=(x[i],y[i]),xytext=(.06,.66),textcoords='axes fraction',
                fontsize=7,color=COL[0],arrowprops=dict(arrowstyle='->',color=COL[0],lw=.7),
                bbox=dict(fc='white',ec='none',pad=1))
        elif signal=='tangent':
            ax.set_ylim(-11,26)
            note(ax,f"MSFC RMSE\n{1000*ours['tangent_rmse']:.3f} mm",.66)
            inset(ax,rows,tau,signal,(4.1,4.4,-.5,.5),[.53,.59,.43,.16])
        else:
            x,y=expected['MSFC'];note(ax,f"MSFC peak |torque|: {abs(y).max():.2f} N m",.66)
        legend(ax,ncol=4 if signal in ['tangent','torque'] else 3) if ax is axs[0] else None
        check(ax,expected,src,name,tau)
    axs[-1].set_xlabel('Time (s)',fontsize=8);save(fig,name)

def render_ramps():render_response('feedback','comparison_ramps',RAMPS)
def render_configs():
    fig,axs=new_axes()
    for ax,tau in zip(axs,[.01,.002]):
        expected={}
        for j,(m,label) in enumerate(LABELS.items()):
            rr=sorted([r for r in ROWS if r['case']['method']==m and r['case']['contact_tau']==tau],key=lambda r:r['case']['configuration'])
            assert len(rr)==16
            x=np.arange(1,17);y=np.array([100*r['nmae'] for r in rr]);expected[label]=(x,y)
            ax.plot(x,y,color=COL[j],ls=STYLE[j],lw=1.15 if j==0 else .8,label=label,zorder=8 if j==0 else 3)
        ax.set_xlim(1,16);ax.set_ylim(0,155);ax.set_ylabel('Normalized MAE (%)');ax.set_xticks([1,4,8,12,16])
        avg=expected['MSFC'][1].mean();best=min((k for k in expected if k!='MSFC'),key=lambda k:expected[k][1].mean());others=expected[best][1].mean()
        assert avg<others
        note(ax,f'Mean: MSFC {avg:.2f}% vs {best} {others:.2f}%',.72)
        legend(ax,ncol=4) if ax is axs[0] else None
        check(ax,expected,[str(DATA/'results.json')],'configuration_comparison',tau)
    axs[-1].set_xlabel('Physical configuration');save(fig,'configuration_comparison')
def render_ablation():
    fig,axs=new_axes()
    for ax,tau in zip(axs,[.01,.002]):
        expected,src=lines(ax,ABLATIONS,tau,'feedback',True)
        ys=np.concatenate([v[1] for v in expected.values()]);ax.set_ylim(min(0,ys.min()-1),ys.max()+18)
        ax.set_ylabel('Feedback force (N)')
        means={m:100*np.mean([r['nmae'] for r in ROWS if r['case']['method']==m and r['case']['contact_tau']==tau]) for m in AB}
        assert all(means[m]>means['reference_inverse'] for m in AB if m!='reference_inverse')
        note(ax,f"Mean: MSFC {means['reference_inverse']:.2f}%; no motion {means['no_motion']:.2f}%",.66)
        legend(ax,True) if ax is axs[0] else None
        check(ax,expected,src,'ablation_response',tau)
    axs[-1].set_xlabel('Time (s)');save(fig,'ablation_response')
FIGURES={'comparison_feedback':lambda:render_response('feedback','comparison_feedback'),
         'comparison_raw':lambda:render_response('raw','comparison_raw'),
         'comparison_tangent':lambda:render_response('tangent','comparison_tangent'),
         'comparison_torque':lambda:render_response('torque','comparison_torque'),
         'configuration_comparison':render_configs,'comparison_ramps':render_ramps,
         'ablation_response':render_ablation}
def render(output):
    global DATA, OUT, ROWS, TRACES, RAMPS, ABLATIONS
    DATA=Path(output);OUT=DATA/'figures';OUT.mkdir(parents=True,exist_ok=True)
    ROWS=json.loads((DATA/'results.json').read_text())
    TRACES=[r for r in ROWS if r['case']['configuration']==0 and r['case']['method'] in LABELS]
    RAMPS=[r for r in ROWS if r['case']['configuration']==1 and r['case']['method'] in LABELS]
    ABLATIONS=[r for r in ROWS if r['case']['configuration']==0 and r['case']['method'] in AB]
    plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':8,'axes.labelsize':8,'axes.titlesize':8,
        'legend.fontsize':6.4,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.grid':True,'grid.alpha':.17,'axes.linewidth':.6,'pdf.fonttype':42})
    for name,plot in FIGURES.items():
        plot();print('Saved',OUT/(name+'.pdf'),flush=True)
