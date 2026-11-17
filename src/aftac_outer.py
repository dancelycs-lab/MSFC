"""Scalar outer-loop adaptation of Liu et al., TIE 2024, Sec. III-B.
DOI: 10.1109/TIE.2024.3383029. Equations (32)-(33), Fig. 2.
Positive velocity pushes into the fixture, reversing the paper's normal axis.
Measured robot velocity is used as indicated by Fig. 2 (x dot); common PI
servo replaces the paper's NN position controller. This is not full AFTAC.
"""
import math
class AFTACOuter:
    def __init__(self,mass=1.,damping=350.,bandwidth=10.,step=.001):
        self.m=mass;self.b=damping;self.L=bandwidth;self.h=step
        self.z=0.;self.u=0.
    def step(self,target,force,velocity):
        error=target-force
        disturbance=self.z+self.L*velocity
        rho=math.exp(-self.b*self.h/self.m)
        self.u=rho*self.u+(1-rho)*(error+self.m*disturbance)/self.b
        # Saturated outer reference is the implemented integrator state.
        self.u=max(-.06,min(.06,self.u))
        obsrho=math.exp(-self.L*self.h)
        self.z=obsrho*self.z-(1-obsrho)*(self.L*velocity+(error-self.b*velocity)/self.m)
        return self.u
