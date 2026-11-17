"""Scalar no-extra-payload adaptation of Samuel et al., RA-L 2024.
DOI 10.1109/LRA.2024.3406055, equations 10,14,16,19,33.
Velocity form of Eq.33; Tn excludes transport delay, as in the paper.
Cn=kp+ki/s, Rn=1/(m*s+b). N2=0 (no separate payload sensor).
Bilinear discretization at the common outer-loop period. Sensor smoothing,
command clipping, actuator lag and uncertain plant belong to the common testbed.
Not a reproduction of the authors' hardware results.
"""
import numpy as np
from scipy.signal import cont2discrete
class Filter:
 def __init__(self,num,den,h):
  b,a,_=cont2discrete((num,den),h,method='bilinear');b=np.ravel(b);a=np.ravel(a)
  self.b=b/a[0];self.a=a/a[0];self.z=np.zeros(len(a)-1)
 def step(self,u):
  y=self.b[0]*u+self.z[0]
  self.z[:-1]=self.z[1:]+self.b[1:-1]*u-self.a[1:-1]*y
  self.z[-1]=self.b[-1]*u-self.a[-1]*y
  return y
class SaturatedPROB:
 """Equation 10 interconnection with actual clipped command fed back to Q.
 Resolves bilinear Q direct feedthrough analytically before applying clipping.
 Equivalent to velocity Eq.33 in the unsaturated case; avoids feeding an
 unrealizable command to the observer when the benchmark imposes limits.
 """
 def __init__(self,kind,gain=.004,tau=.02,omega=30.,h=.001,kp=180.,ki=3000.,mass=1.,damping=1.,limit=.06):
  self.limit=limit;self.admittance=Filter([gain],[tau,1.],h)
  self.q=Filter([omega],[1.,omega],h)
  self.qt=Filter(omega*np.array([mass,damping+kp,ki]),np.convolve([1.,omega],[kp,ki]),h)
  if kind=='oidob':num,den=[gain,0.],np.convolve([tau,1.],[1.,omega])
  elif kind=='mob':num,den=[omega,0.],np.convolve([1.,omega],[kp,ki])
  elif kind=='cdyob':num,den=omega*gain*np.array([mass,damping,0.]),np.convolve(np.convolve([1.,omega],[kp,ki]),[tau,1.])
  else:raise ValueError(kind)
  self.n1=Filter(num,den,h)
 def step(self,target,force,velocity):
  nominal=self.admittance.step(target-force)-self.qt.step(velocity)-self.n1.step(force)
  raw=(nominal+self.q.z[0])/(1.-self.q.b[0])
  command=float(np.clip(raw,-self.limit,self.limit));self.q.step(command)
  return command
