"""Scalar adaptation of Wang et al., RA-L, DOI 10.1109/LRA.2025.3636028.
Observer: Eq. (10); velocity projection: Eq. (19).
This is not the authors' complete UR3e implementation. The nominal controller,
sampling and servo are those of our test plant. No safety claim is transferred
from the velocity-kinematic theorem to the delayed dynamic plant.
"""
class ForceCBF:
    def __init__(self, stiffness=5000., observer_gain=30., barrier_gain=20., margin=10.,limit=15.):
        self.k=stiffness;self.L=observer_gain;self.alpha=barrier_gain
        self.margin=margin;self.limit=limit;self.z=0.;self.active=False
    def step(self, force, velocity, nominal, dt):
        if force<=.5:
            self.active=False;return nominal,False
        if not self.active:self.z=-self.L*force;self.active=True
        disturbance_rate=self.z+self.L*force
        predicted_force=force
        upper=(self.alpha*(self.limit-predicted_force)-disturbance_rate-self.margin)/self.k
        projected=min(nominal,upper)
        self.z+=dt*(-self.L*(self.k*velocity+self.z+self.L*force))
        # Caller applies actuator/velocity bounds; if upper<-0.06 the QP is infeasible.
        return projected,upper<-.06
