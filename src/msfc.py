"""Force regulation from force and tool encoder measurements.

Inputs: force [N], axial position [m], velocity [m/s], timestamp [s].
Output: axial velocity command [m/s]. Positive direction enters the workpiece.
No workpiece state, true material parameter, or simulator API is accessed.
"""
from collections import deque
import math
import numpy as np


class SeparatedForceController:
    def __init__(self, target_force=10., command_delay=.0, step=.001,
                 stiffness_prior=5000., damping_prior=10., feedback_gain=.004,
                 initial_position=.025, use_delay_gain_cap=False,
                 identify_until=float('inf'), project_time=True,
                 motion_compensation=True, fixed_contact_model=False,
                 parameter_estimator=None, reference_inverse=False):
        if step != .001:
            raise ValueError('This validated configuration uses a 1 ms control period.')
        self.target=float(target_force)
        self.delay=float(command_delay)
        self.h=float(step)
        self.k=float(stiffness_prior)
        self.c=float(damping_prior)
        if self.k<=0 or self.c<=0 or self.delay<0 or self.target<=0:
            raise ValueError('Positive material priors and target, nonnegative delay required.')
        self.g=float(feedback_gain)
        self.use_delay_gain_cap=bool(use_delay_gain_cap)
        self.identify_until=float(identify_until)
        self.project_time=bool(project_time)
        self.motion_compensation=bool(motion_compensation)
        self.fixed_contact_model=bool(fixed_contact_model)
        self.parameter_estimator=parameter_estimator
        self.reference_inverse=bool(reference_inverse)
        self.reference_elastic=self.target
        self.update_jump_squared=0.;self.update_jump_max=0.;self.reset_count=0
        self.f=0.;self.x=float(initial_position);self.v=0.
        self.z=0.;self.nu=0.;self.accepted=0;self.index=0
        self.history=deque(maxlen=120)

    def step(self, force, position, velocity, timestamp):
        self.f+=.25*(float(force)-self.f)
        self.x+=.25*(float(position)-self.x)
        self.v+=.25*(float(velocity)-self.v)
        self.history.append((float(timestamp),self.x,self.f,self.v))
        old_k,old_c=self.k,self.c
        if self.parameter_estimator is not None:
            fitted=self.parameter_estimator.step(self.f,self.v)
            if self.f>1. and fitted is not None and self.index%10==0:
                self.k=.8*self.k+.2*fitted
                self.accepted+=1
        elif len(self.history)==120 and self.index%10==0 and timestamp<self.identify_until:
            hist=np.asarray(self.history)
            if hist[:,2].min()>1.:
                t=(hist[:,0]-hist[-1,0])/.12
                B=np.column_stack([np.ones(120),t,t*t,hist[:,3]/.04])
                if not self.project_time:B[:,1:3]=0.
                values=hist[:,1:3]
                res=values-B@np.linalg.lstsq(B,values,rcond=None)[0]
                q=float(res[:,0]@res[:,0])
                if q/120>(2e-6)**2:
                    fitted=float(res[:,0]@res[:,1]/q)
                    error=float(np.sqrt(np.mean((res[:,1]-fitted*res[:,0])**2)))
                    if 300<fitted<30000 and error<.4:
                        if not self.fixed_contact_model:self.k=.8*self.k+.2*fitted
                        self.accepted+=1
                        coeff=np.linalg.lstsq(B,hist[:,2]-fitted*hist[:,1],rcond=None)[0]
                        damping=float(coeff[3]/.04)
                        if 0<damping<200 and not self.fixed_contact_model:self.c=.8*self.c+.2*damping
        if self.k!=old_k or self.c!=old_c:
            old_beta=-math.expm1(-self.h/max(.001,old_c/old_k))/(self.h*old_k)
            new_beta=-math.expm1(-self.h/max(.001,self.c/self.k))/(self.h*self.k)
            jump=(new_beta-old_beta)*(self.f-self.z)
            self.update_jump_squared+=jump*jump
            self.update_jump_max=max(self.update_jump_max,abs(jump))
            self.reset_count+=1
        control_force=self.f
        control_velocity=self.v
        previous=self.z
        tau=max(.001,self.c/self.k)
        self.z+=(1-math.exp(-self.h/tau))*(control_force-self.z)
        rate=(self.z-previous)/self.h
        if control_force>.5:
            alpha=.1
            self.nu+=alpha*(float(np.clip(control_velocity-rate/self.k,-.06,.06))-self.nu)
        else:
            self.nu*=.98
        gain=min(self.g,.7/(5000.*(self.delay+.02))) if self.use_delay_gain_cap else self.g
        trust=min(1.,self.accepted/10.)
        if not self.motion_compensation:trust=0.
        reference_velocity=0.
        if self.reference_inverse:
            old_reference=self.reference_elastic
            self.reference_elastic+=(1-math.exp(-self.h/tau))*(self.target-self.reference_elastic)
            reference_velocity=(self.reference_elastic-old_reference)/(self.h*self.k)
        command=trust*self.nu+gain*(self.target-control_force)+reference_velocity
        if control_force<=.5:command=self.g*(self.target-control_force)
        self.index+=1
        return float(np.clip(command,-.06,.06))

    def diagnostics(self):
        return dict(stiffness=self.k,damping=self.c,force=self.f,
                    equilibrium_velocity=self.nu,accepted_updates=self.accepted)
