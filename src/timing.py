"""Measured Python controller cost on exactly replayed arm measurements."""
from pathlib import Path
import collections,json,os,platform,time
import numpy as np
from .plant import Comparator,force_reference
ROOT=Path(__file__).resolve().parents[1]
def measure(output,rows):
    output=Path(output)
    records=[]
    for r in rows:
        c=r['case'];z=np.load(output/r['trace_path']);a=z['trace'];stride=round(.001/c['dt'])
        data=[];rng=np.random.default_rng(c['seed']);x0=.65-.012-.025-.005
        for n in range(0,len(a),stride):
            if n==0:f,x,v,t=0.,x0,0.,0.
            else:f,x,v,t=a[n-1,1],a[n-1,2],a[n-1,4],a[n-1,0]
            data.append((f+rng.normal(0,c.get('force_noise',.1)),x+rng.normal(0,1e-6),v+rng.normal(0,.0002),t))
        costs=[];updates=[];errors=[]
        for rep in range(3):
            ctl=Comparator(c['method'],c['target'],x0,c.get('controller_parameters'));q=collections.deque([0.]*round(c['delay']/.001))
            if c['method']=='force_cbf':ctl.controller.limit=1.5*(c['target']+5.)
            for j,(f,x,v,t) in enumerate(data):
                ctl.target=force_reference(t,c['target'],c['force_profile'])
                if ctl.adaptive:ctl.controller.target=ctl.target
                old=ctl.controller.accepted if ctl.adaptive else 0
                t0=time.perf_counter_ns();u=ctl.step(f,x,v,t);ns=time.perf_counter_ns()-t0
                q.append(u);errors.append(abs(q.popleft()-a[j*stride,6]))
                # First full pass warms imports and LAPACK; subsequent full
                # passes restart identical states and are measured.
                if rep>0:
                    costs.append(ns/1000.)
                    if ctl.adaptive and ctl.controller.accepted>old:updates.append(ns/1000.)
        assert max(errors)<1e-8,(c['method'],max(errors))
        record=dict(method=c['method'],force_profile=c['force_profile'],calls=len(costs),median_us=float(np.median(costs)),p95_us=float(np.quantile(costs,.95)),p99_us=float(np.quantile(costs,.99)),maximum_us=float(max(costs)),over_one_ms=int(np.sum(np.array(costs)>1000)),accepted_update_calls=len(updates),update_median_us=float(np.median(updates)) if updates else None,replay_max_command_difference=float(max(errors)))
        records.append(record);print(record,flush=True)
    out=output/'controller_timing.json'
    cpuinfo=Path('/proc/cpuinfo')
    cpu_model=next((s.split(':',1)[1].strip() for s in cpuinfo.read_text().splitlines() if s.startswith('model name')),platform.processor()) if cpuinfo.exists() else platform.processor()
    out.write_text(json.dumps(dict(environment=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,cpu=platform.processor(),cpu_model=cpu_model,blas_threads=os.environ.get('OPENBLAS_NUM_THREADS')),records=records,interpretation='Wall-clock call costs in a local Python process after one full-trajectory warmup. Includes operating-system scheduling, not a worst-case real-time execution guarantee.'),indent=2))
