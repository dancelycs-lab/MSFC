"""Two-candidate force estimator for a measured-velocity interface.

Uses COBA's two-observer interpolation (CEP 2014, Eq. 13-14), but
replaces its computed-torque plant by Fdot=k*v. This is an estimator
adaptation, not a reproduction of the paper's complete force controller.
"""
import numpy as np
class TwoForceObservers:
    def __init__(self, correction=.05, stiffness=(100.,30000.), step=.001):
        self.correction=correction;self.ks=np.array(stiffness,dtype=float)
        self.h=step;self.f=np.zeros(2)
    def step(self,force,velocity):
        predicted=self.f+self.h*self.ks*velocity
        self.f=(1-self.correction)*predicted+self.correction*force
        gamma=(self.f[0]-self.f[1])/(self.ks[1]-self.ks[0])
        if abs(gamma)<=2e-6:return None
        fitted=self.ks[1]-(force-self.f[1])/gamma
        return float(fitted) if 300<fitted<30000 else None
