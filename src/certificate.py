"""Conditional persistent-contact certificate with moving, varying material.

Actual actuator, sensor filters, saturation and contact switching are excluded.
Parameter boxes are theorem examples, not constraints of the deployed controller.
"""
from pathlib import Path
import itertools
import json
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
SCALE = np.array([10., 10., .01])

def matrix(k, c, a, b, alpha, beta):
    g, ell = .004, 100.
    p, q, h = k/(1+g*c), c/(1+g*c), 1/(1+g*c)
    A = np.array([[-g*p-ell*q*(g+b)+alpha-g*beta*h, ell*q*b,
                   p+beta*h-q*alpha], [a, -a, 0.],
                  [-ell*(g+b), ell*b, 0.]])
    return A*SCALE[None, :]/SCALE[:, None]/100.

def symbolic_check():
    k,c,g,ell,a,b,alpha,beta,fd,e,z,r,wa = sp.symbols(
        'k c g ell a b alpha beta fd e z r wa')
    nu_dot = -ell*g*e-ell*b*(e-z)
    deformation = (fd+(1+c*g)*e-c*r)/k
    direct = (-k*g*e+k*r+c*nu_dot-c*wa+k*alpha*deformation+beta*(r-g*e))/(1+c*g)
    p,q,h = k/(1+c*g),c/(1+c*g),1/(1+c*g)
    closed = (-g*p-ell*q*(g+b)+alpha-g*beta*h)*e+ell*q*b*z+(p+beta*h-q*alpha)*r-q*wa+fd*alpha*h
    assert sp.simplify(direct-closed) == 0
    return str(sp.factor(direct))

def verify_exact(P, bounds):
    P = sp.Matrix([[sp.Rational(f'{v:.12f}') for v in row] for row in P])
    if not all(P[:i,:i].det()>0 for i in [1,2,3]):
        return dict(verified=False, reason='rounded P not positive definite')
    D=sp.diag(10,10,sp.Rational(1,100)); g=sp.Rational(1,250); ell=100
    checked=0
    for vertex in itertools.product(*bounds):
        k,c,a,b,alpha,beta=[sp.Rational(str(x)) for x in vertex]
        p,q,h=k/(1+g*c),c/(1+g*c),1/(1+g*c)
        A=sp.Matrix([[-g*p-ell*q*(g+b)+alpha-g*beta*h,ell*q*b,p+beta*h-q*alpha],
                     [a,-a,0],[-ell*(g+b),ell*b,0]])
        A=D.inv()*A*D/100
        Q=-(A.T*P+P*A)
        if not all(Q[:i,:i].det()>0 for i in [1,2,3]):
            return dict(verified=False, failed_vertex=[str(v) for v in vertex])
        checked+=1
    return dict(verified=True, exact_vertices=checked,
                P=[[str(x) for x in row] for row in P.tolist()])

