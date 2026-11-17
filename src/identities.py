def run_checks():
    """Check the projection and reference-mismatch identities used in the manuscript."""
    import json
    from pathlib import Path
    import numpy as np
    import sympy as sp
    s,k,c,kh,ch,g,ell=sp.symbols('s k c kh ch g ell',positive=True)
    G=(c*s+k)/s; H=s/(ch*s+kh); Ce=g*(1+ell/s)+ell/(ch*s+kh)
    p=sp.cancel(s*s*(ch*s+kh)*(1+G*Ce))
    assert sp.cancel((G*H-1)/(1+G*Ce)-s*s*((c-ch)*s+k-kh)/p)==0
    coeffs=[ch*(1+g*c),(1+g*c)*kh+g*ch*k+ell*c*(1+g*ch),g*ell*ch*k+g*kh*k+g*ell*kh*c+ell*k,g*ell*kh*k]
    assert sp.expand(p-sum(a*s**(3-i) for i,a in enumerate(coeffs)))==0
    R2=g*ch*k+ell*c*(1+g*ch);R1=g*kh*k+g*ell*kh*c+ell*k
    assert sp.expand(coeffs[1]*coeffs[2]-coeffs[0]*coeffs[3]-(1+g*c)*kh*R1-R2*(g*ell*ch*k+R1))==0
    rng=np.random.default_rng(1729); errors=[]
    for degenerate in [False,True]:
        for _ in range(100):
            t=np.linspace(-1,0,120);B=np.c_[np.ones_like(t),t,t*t]
            x=rng.normal(size=120);v=B@rng.normal(size=3) if degenerate else rng.normal(size=120)
            Z=np.c_[B,v/.04];P=np.eye(120)-Z@np.linalg.pinv(Z);PB=np.eye(120)-B@np.linalg.pinv(B)
            xb=PB@x;vb=PB@v;q=x@P@x
            gram=np.array([[xb@xb,xb@vb],[vb@xb,vb@vb]])
            error=abs(np.linalg.det(gram)-(vb@vb)*q)/max(1,abs(np.linalg.det(gram)))
            assert error<1e-10;errors.append(error)
            r=rng.normal(size=120)*.01;f=2000*x+15*v+B@rng.normal(size=3)+r
            fit=x@P@f/q;delta=(P@x)@(P@r)/q
            assert abs(fit-2000-delta)<1e-8
            assert abs(fit-2000)<=np.linalg.norm(P@r)/np.sqrt(q)+1e-8
            assert np.linalg.norm(P@r)/np.sqrt(q)<=np.max(abs(r))/np.sqrt(q/120)+1e-8
    out={'reference_mismatch_identity':'exact symbolic pass','characteristic_polynomial':'exact symbolic pass','routh_identity':'exact symbolic pass','projection_trials':200,'includes_polynomial_velocity':True,'max_relative_gram_error':max(errors)}
    return out
