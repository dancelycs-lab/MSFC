"""Run the MuJoCo comparisons, analyses, and mathematical checks."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[variable]='1'
os.environ['MPLBACKEND']='Agg'
ROOT=Path(__file__).resolve().parent


def read_config(name):
    return json.loads((ROOT/'config'/f'{name}.json').read_text())


def case_key(case):
    return json.dumps(case,sort_keys=True)


def worker(item):
    from src.plant import run
    case,save,output=item
    return run(case,save=save,output_dir=output)


def batch(cases,name,args,save=False):
    path=args.output/(name+'.json')
    expected={case_key(c) for c in cases}
    if len(expected)!=len(cases):
        raise ValueError('Duplicate simulation cases.')
    rows=json.loads(path.read_text()) if path.exists() else []
    if any(case_key(r['case']) not in expected for r in rows):
        raise ValueError(f'{path} contains a different experiment; use a separate --output directory.')
    done={case_key(r['case']) for r in rows if not (save(r['case']) if callable(save) else save) or (r.get('trace_path') and (args.output/r['trace_path']).exists())}
    rows=[r for r in rows if case_key(r['case']) in done]
    pending=[c for c in cases if case_key(c) not in done]
    print(f'{name}: {len(done)}/{len(cases)} complete; {len(pending)} remaining',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,(c,save(c) if callable(save) else save,str(args.output))) for c in pending]
        for future in concurrent.futures.as_completed(futures):
            row=future.result();rows.append(row)
            # Restore design order for paired statistics and training/validation splits.
            by_key={case_key(r['case']):r for r in rows}
            ordered=[by_key[case_key(c)] for c in cases if case_key(c) in by_key]
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(ordered,indent=2));temp.replace(path)
            if len(rows)%16==0 or len(rows)==len(cases):
                print(f'{name}: {len(rows)}/{len(cases)} complete',flush=True)
    return json.loads(path.read_text())


def evaluation_cases():
    return [dict(c,**method) for c in read_config('cases') for method in read_config('controllers').values()]


def evaluation(args,smoke=False):
    from src.labels import LABELS,AB
    cases=evaluation_cases()
    if smoke:
        cases=[c for c in cases if c['configuration']==0 and c['contact_tau']==.01]
    save=lambda c:(c['configuration']==0 and c['method'] in {*LABELS,*AB}) or (c['configuration']==1 and c['method'] in LABELS)
    batch(cases,'results',args,save=save)
    from src.analysis import compare_reference
    difference=compare_reference(args.output)
    if smoke and difference>1e-6:
        raise RuntimeError('Smoke-test metrics differ from the reference by more than 1e-6. See reference_comparison.csv.')


def theory(args):
    import itertools
    import numpy as np
    from src.identities import run_checks
    from src.certificate import verify_exact,matrix,symbolic_check
    from src.dynamics_check import run_checks as dynamics_check
    certificate_rows=[]
    report={'identities':run_checks(),'material_rate_identity':symbolic_check(),'certificates':[]}
    for cert in read_config('certificates')['results']:
        P=np.asarray(cert['P']);bounds=cert['bounds']
        result=verify_exact(P,bounds)
        assert result['verified']
        mats=np.array([matrix(*v) for v in itertools.product(*bounds)])
        result['minimum_margin_per_s']=float(np.linalg.eigvalsh(-(mats.transpose(0,2,1)@P+P@mats)).min()*100)
        result['bounds']=bounds
        result['parameter_order']=cert['order']
        report['certificates'].append(result)
        certificate_rows.append(dict(domain=len(certificate_rows)+1,**{name+'_min':bound[0] for name,bound in zip(cert['order'],bounds)},**{name+'_max':bound[1] for name,bound in zip(cert['order'],bounds)},minimum_margin_per_s=result['minimum_margin_per_s'],exact_vertices=result['exact_vertices']))
    report['dynamics']=dynamics_check()
    (args.output/'theory.json').write_text(json.dumps(report,indent=2))
    from src.analysis import save_csv
    (args.output/'tables').mkdir(exist_ok=True)
    save_csv(args.output/'tables/certificates.csv',certificate_rows)
    print('Symbolic identities, 128 exact certificate vertices, and physical/reduced dynamics verified.',flush=True)


def validate(args):
    from src.analysis import validation_summary
    for name in ['constant_force','refinement','msfc_refinement']:
        batch(read_config(name),name,args)
    validation_summary(args.output)


def analyze(args):
    from src.analysis import summarize
    from src.plotting import render
    summarize(args.output)
    render(args.output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',default='all',choices=['all','smoke','run','analyze','theory','validate','timing','train'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    args.output=(args.output or ROOT/('smoke_output' if args.stage=='smoke' else 'output')).resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage in ['all','run','smoke']:evaluation(args,args.stage=='smoke')
    if args.stage in ['all','analyze']:analyze(args)
    if args.stage in ['all','theory']:theory(args)
    if args.stage in ['all','validate']:validate(args)
    if args.stage in ['all','timing']:
        rows=batch(read_config('timing'),'timing_trajectories',args,save=True)
        from src.timing import measure
        measure(args.output,rows)
    if args.stage=='train':
        batch(read_config('training'),'training',args,save=True)
        from src.training import fit
        fit(args.output)


if __name__=='__main__':
    main()
