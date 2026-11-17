"""Native torque-interface adaptations of two TCST 2026 controllers.

Ma et al., doi:10.1109/TCST.2025.3606519, equations (51)--(54).
Petrone et al., doi:10.1109/TCST.2025.3635172, equations (4),(7)--(12).
Force corrections and position requests use the configured outer-loop delay.
"""
from pathlib import Path
import collections
import time
import numpy as np
import mujoco


class Ensemble:
    def __init__(self, path):
        with np.load(path) as archive:
            self.p = {name:archive[name] for name in archive.files}

    def predict(self, inputs):
        p = self.p
        z = ((inputs-p['x_mean'])/p['x_scale']).astype(np.float32)
        outputs = []
        for j in range(3):
            a = z
            for i in range(4):
                a = a@p[f'w{j}_{i}'] + p[f'b{j}_{i}']
                if i != 3:
                    a = np.maximum(a, 0.)
            outputs.append(a)
        return np.mean(outputs, axis=0)*p['y_scale']+p['y_mean']


class TCSTTorqueController:
    def __init__(self, method, nominal_model, kinematics, parameters=None):
        self.method = method
        self.m = nominal_model
        self.d = mujoco.MjData(self.m)
        self.kinematics = kinematics
        self.p = parameters or {}
        self.mass = np.zeros((self.m.nv, self.m.nv))
        self.h = .001
        self.f = 0.
        self.df = 0.
        self.z = np.zeros(2)
        self.integral = 0.
        self.active = False
        self.previous_action = None
        self.rows = []
        self.surface = .613
        self.last_action = self.surface+.002
        self.ensemble = None
        self.residual_samples = []
        self.inference_seconds = []
        self.outer_queue = None
        if 'outer_delay' in self.p:
            initial = np.zeros(2) if method=='ph_force_tcst' else self.surface+.002
            self.outer_queue = collections.deque([initial for _ in range(round(self.p['outer_delay']/self.h))])
        if method == 'oracle_tcst':
            self.ensemble = Ensemble(Path(__file__).resolve().parents[1] / self.p['model_path'])

    def mechanics(self, q, qdot):
        """Nominal robot model only; never read the plant's contact/material state."""
        self.d.qpos[:2] = q
        self.d.qvel[:2] = qdot
        mujoco.mj_forward(self.m, self.d)
        mujoco.mj_fullM(self.m, self.d, self.mass)
        M = self.mass[:2, :2].copy()
        bias = self.d.qfrc_bias[:2].copy() + .2*qdot
        x, J = self.kinematics(q)
        Ji = np.linalg.inv(J)
        B = Ji.T@M@Ji
        # Directional derivative along qdot, for p_bar = B(x) v.
        eps = 1e-5
        Bs, Js = [], []
        for sign in [-1, 1]:
            qq = q + sign*eps*qdot
            self.d.qpos[:2] = qq
            mujoco.mj_forward(self.m, self.d)
            mujoco.mj_fullM(self.m, self.d, self.mass)
            _, jj = self.kinematics(qq)
            jji = np.linalg.inv(jj)
            Bs.append(jji.T@self.mass[:2, :2]@jji)
            Js.append(jj)
        return x, J, Ji, M, B, (Bs[1]-Bs[0])/(2*eps), (Js[1]-Js[0])/(2*eps), bias

    def step(self, force, q, qdot, reference, yref, t):
        old_f = self.f
        self.f += .25*(force-self.f)
        self.df += .25*((self.f-old_f)/self.h-self.df)
        x, J, Ji, M, B, Bdot, Jdot, bias = self.mechanics(q, qdot)
        v = J@qdot
        if self.method == 'ph_force_tcst':
            km = self.p.get('stiffness', 100.)
            dm = self.p.get('damping', 40.)
            kf = self.p.get('force_gain', .05)
            md = self.p.get('desired_mass', 1.)
            K = np.diag([km, km])
            ef = np.array([self.f-reference, 0.])
            fe = np.array([-self.f, 0.])
            # The original law uses constant references. For trajectory tests
            # setpoints are supplied directly; no new tracking theorem is assumed.
            e = x-np.array([self.surface, yref])
            zdot = B@(kf*ef)
            phat = B@v + 2*K@(e+self.z) + kf*ef
            base = -md*np.linalg.solve(B, K@e)-2*K@v-dm/md*(B@v+2*K@e)
            outer = -md*np.linalg.solve(B, K@self.z+kf*ef)-2*K@zdot
            outer += kf*np.array([-self.df, 0.])-dm/md*(2*K@self.z+kf*ef)-fe
            if self.outer_queue is not None:
                self.outer_queue.append(outer.copy())
                outer = self.outer_queue.popleft()
            accel = np.linalg.solve(B, base+outer-Bdot@v)
            tau = M@Ji@(accel-Jdot@qdot) + bias
            self.z += self.h*zdot
            self.last_action = x[0]
        else:
            kp = self.p.get('kp', .0001)
            ki = self.p.get('ki', .005)
            error = reference-self.f
            if not self.active and self.f > .5:
                self.active = True
                self.integral = self.last_action-self.surface-kp*error
            if self.active:
                self.integral += self.h*ki*error
                base = self.surface+kp*error+self.integral
            else:
                base = self.surface+.002
            action = base
            if self.ensemble is not None and self.active:
                clock=time.perf_counter()
                rho = self.p.get('radius', .002)
                candidates = base + np.linspace(-rho, rho, 21)
                inputs = np.column_stack([np.full(21, x[0]-self.surface),
                    np.full(21, v[0]), np.full(21, self.f), candidates-self.surface])
                pred = self.f + self.ensemble.predict(inputs)[:, 2]
                # Position coordinates are translated to the nominal surface.
                loss = abs(reference-pred)+self.p.get('alpha', 50.)*(candidates-self.surface)**2
                if self.previous_action is not None:
                    loss += self.p.get('beta', 50.)*abs(candidates-self.previous_action)
                action = float(candidates[np.argmin(loss)])
                self.residual_samples.append(action-base)
                self.inference_seconds.append(time.perf_counter()-clock)
            self.previous_action = action
            self.last_action = action
            if self.outer_queue is not None:
                self.outer_queue.append(action)
                action = self.outer_queue.popleft()
            K = self.p.get('impedance_stiffness', 1000.)
            D = 2*np.sqrt(K)
            tau = J.T@(K*(np.array([action, yref])-x)-D*v)+bias
            self.rows.append([t, x[0]-self.surface, v[0], self.f, action-self.surface])
        return tau
