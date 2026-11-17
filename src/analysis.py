"""Aggregate physical-force, motion, actuation, and paired comparison metrics."""
import csv
import json
from pathlib import Path
import numpy as np
from .labels import LABELS, EXTRA, AB


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output):
    output = Path(output)
    rows = json.loads((output/'results.json').read_text())
    keys = {(r['case']['method'], r['case']['contact_tau'], r['case']['configuration']) for r in rows}
    expected = {(m, tau, i) for m in {*LABELS, *EXTRA, *AB} for tau in [.01, .002] for i in range(16)}
    if keys != expected or len(rows) != 512:
        raise ValueError('Complete all 512 evaluation runs before aggregating results.')
    dest = output/'tables'
    dest.mkdir(exist_ok=True)
    summaries = []
    waveforms = []
    paired = []
    bootstrap_rng = np.random.default_rng(10472)
    # Same resampling order as the reported comparisons.
    for method in list(LABELS)[1:]:
        for tau in [.01, .002]:
            a = sorted([r for r in rows if r['case']['method'] == 'reference_inverse' and r['case']['contact_tau'] == tau], key=lambda r:r['case']['configuration'])
            b = sorted([r for r in rows if r['case']['method'] == method and r['case']['contact_tau'] == tau], key=lambda r:r['case']['configuration'])
            delta = np.array([y['nmae']-x['nmae'] for x,y in zip(a,b)])
            # New torque-interface comparisons used their own reproducible draws.
            rng = np.random.default_rng(65021) if method in ['ph_force_tcst','oracle_tcst'] else bootstrap_rng
            draws = delta[rng.integers(0,16,(10000,16))].mean(1)
            lo,hi = 100*np.quantile(draws,[.025,.975])
            paired.append(dict(method=LABELS[method],contact_tau_s=tau,msfc_wins=int(sum(delta>0)),tasks=16,mean_difference_pp=float(100*delta.mean()),ci_lower_pp=float(lo),ci_upper_pp=float(hi)))
    for tau in [.01, .002]:
        for method,label in {**LABELS, **EXTRA, **AB}.items():
            rr = [r for r in rows if r['case']['method'] == method and r['case']['contact_tau'] == tau]
            mean = lambda key:float(np.mean([r[key] for r in rr]))
            summaries.append(dict(method=method,label=label,contact_tau_s=tau,tasks=len(rr),nmae_percent=100*mean('nmae'),median_nmae_percent=float(100*np.median([r['nmae'] for r in rr])),force_rmse_N=mean('force_rmse'),peak_force_ratio=max(r['peak_ratio'] for r in rr),tangent_rmse_mm=1000*mean('tangent_rmse'),contact_loss_s=mean('loss_seconds'),torque_rms_Nm=mean('torque_rms'),torque_saturation_fraction=mean('torque_saturation_fraction')))
            for profile in ['sinusoidal','ramps']:
                values=[r['nmae'] for r in rr if r['case']['force_profile']==profile]
                waveforms.append(dict(method=label,contact_tau_s=tau,force_profile=profile,tasks=len(values),nmae_percent=float(100*np.mean(values))))
    save_csv(dest/'all_metrics.csv',summaries)
    save_csv(dest/'comparison.csv',[r for r in summaries if r['method'] in {*LABELS,*EXTRA}])
    save_csv(dest/'ablation.csv',[r for r in summaries if r['method'] in AB])
    save_csv(dest/'waveforms.csv',waveforms)
    save_csv(dest/'paired_comparison.csv',paired)
    save_csv(dest/'individual_runs.csv',[dict(configuration=r['case']['configuration'],contact_tau_s=r['case']['contact_tau'],method=r['case']['method'],**{k:v for k,v in r.items() if isinstance(v,(int,float))}) for r in rows])
    root=Path(__file__).resolve().parents[1]
    save_csv(dest/'task_parameters.csv',json.loads((root/'config/cases.json').read_text()))
    controllers=json.loads((root/'config/controllers.json').read_text())
    save_csv(dest/'controller_settings.csv',[dict(method=m,**c['controller_parameters']) for m,c in controllers.items()])
    print('Saved result tables to',dest,flush=True)


def compare_reference(output):
    output=Path(output)
    ref=Path(__file__).resolve().parents[1]/'config/reference_metrics.csv'
    with ref.open() as f:
        reference={(int(r['configuration']),float(r['contact_tau']),r['method']):r for r in csv.DictReader(f)}
    rows=json.loads((output/'results.json').read_text())
    errors=[]
    for r in rows:
        c=r['case'];old=reference[c['configuration'],c['contact_tau'],c['method']]
        for metric in ['nmae','force_rmse','peak_ratio','loss_seconds','tangent_rmse','torque_rms','accepted_updates']:
            errors.append(dict(configuration=c['configuration'],contact_tau_s=c['contact_tau'],method=c['method'],metric=metric,reference=float(old[metric]),computed=r[metric],absolute_difference=abs(r[metric]-float(old[metric]))))
    save_csv(output/'reference_comparison.csv',errors)
    maximum=max(r['absolute_difference'] for r in errors)
    print(f'Reference comparison: {len(rows)} runs; maximum absolute metric difference {maximum:.3g}',flush=True)
    return maximum


def validation_summary(output):
    output=Path(output)
    constants=json.loads((output/'constant_force.json').read_text())
    save_csv(output/'constant_force_cases.csv',[dict(method=r['case']['method'],contact_tau_s=r['case']['contact_tau'],stiffness_N_per_m=r['case']['stiffness'],late_nmae_percent=100*r['late_nmae'],nmae_percent=100*r['nmae']) for r in constants])
    summary=[]
    for method in sorted({r['case']['method'] for r in constants}):
        for tau in [.01,.002]:
            rr=[r for r in constants if r['case']['method']==method and r['case']['contact_tau']==tau]
            summary.append(dict(method=method,contact_tau_s=tau,tasks=len(rr),nmae_percent=float(100*np.mean([r['nmae'] for r in rr])),late_nmae_percent=float(100*np.mean([r['late_nmae'] for r in rr])),peak_force_ratio=max(r['peak_ratio'] for r in rr)))
    save_csv(output/'constant_force_summary.csv',summary)
    records=[]
    for filename in ['refinement','msfc_refinement']:
        rows=json.loads((output/(filename+'.json')).read_text())
        for method in sorted({r['case']['method'] for r in rows}):
            for tau in [.01,.002]:
                a=[r['nmae'] for r in rows if r['case']['method']==method and r['case']['contact_tau']==tau and r['case']['dt']==5e-5]
                b=[r['nmae'] for r in rows if r['case']['method']==method and r['case']['contact_tau']==tau and r['case']['dt']==2.5e-5]
                assert len(a)==len(b) and a
                records.append(dict(method=method,contact_tau_s=tau,tasks=len(a),nmae_50us=float(np.mean(a)),nmae_25us=float(np.mean(b)),relative_change_percent=float(100*(np.mean(b)/np.mean(a)-1))))
    save_csv(output/'refinement_summary.csv',records)
