"""Compare physical Kelvin states with the derived coupled error dynamics."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from .certificate import matrix

ROOT=Path(__file__).resolve().parents[1]

def parameters(t):
    k=2000+180*np.sin(t);c=15+2*np.sin(t)
    return k,c,550+450*np.sin(12*t),.11+.09*np.sin(17*t+.3),180*np.cos(t)/k,2*np.cos(t)

def support(t):
    return .001*np.sin(2*t),.002*np.cos(2*t),-.004*np.sin(2*t)

def physical(t,s):
    x,z,nu=s;k,c,a,b,_,_=parameters(t);w,wd,_=support(t)
    force=(k*(x-w)+c*(nu-wd)+c*.004*10)/(1+c*.004)
    velocity=nu-.004*(force-10)
    return [velocity,a*(force-z),100*(velocity-b*(force-z)-nu)]

def reduced(t,xi):
    k,c,a,b,alpha,beta=parameters(t);wa=support(t)[2]
    B=np.array([-c/(1+.004*c),0.,-1.])/np.array([10.,10.,.01])
    forcing=np.array([alpha/(1+.004*c),0.,0.])
    return matrix(k,c,a,b,alpha,beta)@xi*100+B*wa+forcing

def run_checks():
    t=np.linspace(0,10,10001);x0=.005;z0=10.;nu0=.002
    kwargs=dict(t_eval=t,rtol=1e-9,atol=1e-11,max_step=.001)
    physical_solution=solve_ivp(physical,(0,10),[x0,z0,nu0],**kwargs)
    reduced_solution=solve_ivp(reduced,(0,10),[0.,0.,0.],**kwargs)
    assert physical_solution.success and reduced_solution.success
    x,z,nu=physical_solution.y;k,c,a,b,alpha,beta=parameters(t);w,wd,_=support(t)
    force=(k*(x-w)+c*(nu-wd)+c*.004*10)/(1+c*.004)
    reconstructed=np.array([(force-10)/10,(z-10)/10,(nu-wd)/.01])
    difference=float(np.max(np.abs(reconstructed-reduced_solution.y)))
    assert difference<2e-6
    cert=json.loads((ROOT/'config/certificates.json').read_text())['results'][1]
    assert cert['exact']['verified']
    P=np.array(cert['P']);eig=np.linalg.eigvalsh(P);q=cert['minimum_margin']*100
    d=np.array([reduced(tt,xx)-matrix(*parameters(tt))@xx*100 for tt,xx in zip(t,reconstructed.T)])
    # The Lyapunov comparison uses the complete exogenous material/motion input.
    bound=2*np.sqrt(eig.max()/eig.min())*np.max(np.linalg.norm(d@P,axis=1))/q
    actual=float(np.max(np.linalg.norm(reconstructed,axis=0)))
    assert actual<=bound
    result=dict(physical_and_reduced_state_max_difference=difference,
        max_scaled_state_norm=actual,conservative_iss_bound=float(bound),
        k_range=[float(k.min()),float(k.max())],c_range=[float(c.min()),float(c.max())],
        max_abs_log_stiffness_rate=float(np.max(np.abs(alpha))),max_abs_damping_rate=float(np.max(np.abs(beta))),
        interpretation='Numerical cross-check of two independently formulated ideal-contact ODEs; not a MuJoCo result or a full-system certificate.')
    return result
