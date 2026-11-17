"""Two-link MuJoCo arm: tangential motion and normal force regulation.

Controller input is tool force and robot encoders only. The moving support and
material state are used by the physical plant and offline evaluation.
"""
from pathlib import Path
import collections
import hashlib
import json
import math
import time
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
from .msfc import SeparatedForceController
from .support_motion import anchor
from .force_cbf import ForceCBF
from .prob_admittance import SaturatedPROB
from .aftac_outer import AFTACOuter
from .two_force_observers import TwoForceObservers
from .torque_control import TCSTTorqueController

def model(case):
    dt=case.get('dt',.0002);mass=case.get('mass_scale',1.)
    tool_joints=''
    if case.get('tool_stiffness') is not None:
        kt=case['tool_stiffness'];ct=case.get('tool_damping',20.)
        tool_joints=f'<joint name="tool_x" type="slide" axis="1 0 0" stiffness="{kt}" damping="{ct}" armature="0"/><joint name="tool_y" type="slide" axis="0 1 0" stiffness="{kt}" damping="{ct}" armature="0"/>'
    return mujoco.MjModel.from_xml_string(f'''<mujoco model="planar_contact_arm">
    <option timestep="{dt}" gravity="0 0 0" integrator="implicitfast" cone="{case.get('cone','pyramidal')}"/>
    <default><geom solref="{case.get('contact_tau',.002)} 1" solimp=".99 .999 .001" friction="{case.get('friction',.15)} .002 .001" condim="{case.get('condim',3)}"/>
    <joint damping=".2" armature=".01"/></default>
    <worldbody><light pos="0 0 2"/><geom type="plane" pos="0 0 -.1" size="1 1 .1" rgba=".85 .85 .85 1"/>
    <body name="upper"><joint name="q1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .4 0 0" size=".018" mass="{1.2*mass}" contype="0" conaffinity="0" rgba=".25 .32 .4 1"/>
      <body name="forearm" pos=".4 0 0"><joint name="q2" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 .35 0 0" size=".015" mass="{.8*mass}" contype="0" conaffinity="0" rgba=".4 .46 .52 1"/>
        <body name="tool" pos=".35 0 0">{tool_joints}<geom name="tip" type="sphere" size=".025" mass=".15" rgba=".18 .2 .22 1"/>
          <site name="tool_site" size=".003"/></body></body></body>
    <body name="fixture" pos=".65 0 0"><joint name="fixture_slide" type="slide" axis="1 0 0" damping="{case.get('damping',12.)}" armature="0" stiffness="{case['stiffness']}"/>
      <geom name="surface" type="box" size=".012 .25 .065" mass="{case.get('fixture_mass',.15)}" rgba=".65 .65 .65 1"/>
    </body></worldbody><actuator><motor joint="q1" ctrllimited="true" ctrlrange="-35 35"/>
    <motor joint="q2" ctrllimited="true" ctrlrange="-25 25"/></actuator></mujoco>''')

def kinematics(q):
    a,b=q;ab=a+b
    x=np.array([.4*math.cos(a)+.35*math.cos(ab),.4*math.sin(a)+.35*math.sin(ab)])
    J=np.array([[-.4*math.sin(a)-.35*math.sin(ab),-.35*math.sin(ab)],
                [.4*math.cos(a)+.35*math.cos(ab),.35*math.cos(ab)]])
    return x,J

def inverse(x,y):
    b=math.acos((x*x+y*y-.4**2-.35**2)/(2*.4*.35))
    return [math.atan2(y,x)-math.atan2(.35*math.sin(b),.4+.35*math.cos(b)),b]

class Comparator:
    def __init__(self,label,target,x0,parameters=None):
        parameters=parameters or {}
        self.label=label;self.target=target;self.force=0.;self.velocity=0.;self.integral=0.
        self.g=.004;self.ti=.02
        self.adaptive=label in ['retained','reference_inverse','calibrate_only','no_time','no_motion','fixed_observer','two_observer_estimator']
        if self.adaptive:
            options={}
            if label=='reference_inverse':options['reference_inverse']=True
            if label=='calibrate_only':options['identify_until']=1.2
            if label=='no_time':options['project_time']=False
            if label=='no_motion':options['motion_compensation']=False
            if label=='fixed_observer':
                s=[0,0,20000.,50.]
                options.update(stiffness_prior=s[2],damping_prior=s[3],fixed_contact_model=True)
                options.update(parameters)
            if label=='two_observer_estimator':
                s=[0,0,.25,50.]
                options.update(parameter_estimator=TwoForceObservers(correction=parameters.get('correction',s[2])),damping_prior=parameters.get('damping_prior',s[3]))
            if parameters.get("reference_inverse",False):options["reference_inverse"]=True
            for key in ['stiffness_prior','damping_prior']:
                if key in parameters:options[key]=parameters[key]
            controller_type=SeparatedForceController
            self.controller=controller_type(target_force=target,initial_position=x0,**options)
        elif label=='pi':
            s=[0,0,.004,.02];self.g=s[2];self.ti=s[3]
            self.g=parameters.get('gain',self.g);self.ti=parameters.get('integral_time',self.ti)
        elif label=='force_cbf':
            self.g=parameters.get('gain',self.g)
            self.controller=ForceCBF(observer_gain=parameters.get('observer_gain',30.),barrier_gain=parameters.get('barrier_gain',5.),limit=1.5*target)
        elif label in ['oidob','mob','cdyob']:
            s={'oidob': {'gain': 0.008, 'prob_omega': 15.0}, 'mob': {'gain': 0.008, 'prob_omega': 15.0}, 'cdyob': {'gain': 0.008, 'prob_omega': 15.0}}[label]
            self.controller=SaturatedPROB(label,parameters.get('gain',s['gain']),.02,parameters.get('omega',s['prob_omega']))
        elif label=='aftac_outer':
            s=[0,0,125.,5.];self.controller=AFTACOuter(damping=parameters.get('damping',s[2]),bandwidth=parameters.get('bandwidth',s[3]))
        else:raise ValueError(label)
    def step(self,f,x,v,t):
        if self.adaptive:return self.controller.step(f,x,v,t)
        self.force+=.25*(f-self.force);self.velocity+=.25*(v-self.velocity)
        e=self.target-self.force
        if self.label=='pi':
            proposed=self.integral+.001*self.g/self.ti*e;raw=self.g*e+proposed
            if abs(raw)<.06 or raw*e<0:self.integral=proposed
            u=self.g*e+self.integral
        elif self.label=='force_cbf':u,_=self.controller.step(self.force,v,self.g*e,.001)
        else:u=self.controller.step(self.target,self.force,self.velocity)
        return float(np.clip(u,-.06,.06))

def force_reference(t,base,profile,frequency=.6,amplitude=4.):
    if profile=='constant':return base
    phase=max(0.,t-1.5)
    if profile=='sinusoidal':return base+amplitude*math.sin(2*math.pi*frequency*phase)**3
    if profile=='ramps':
        smooth=lambda z:.5-.5*math.cos(math.pi*np.clip(z,0.,1.))
        return base+5*smooth((t-1.5)/.5)-10*smooth((t-3.5)/.5)+5*smooth((t-6)/.5)
    raise ValueError(profile)

def run(case,save=False,output_dir=None):
    start=time.perf_counter();m=model(case);d=mujoco.MjData(m)
    x0=.65-.012-.025-.005;y0=case.get('y0',-.04)
    fixture_joint=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'fixture_slide')
    fixture_q=m.jnt_qposadr[fixture_joint];fixture_dof=m.jnt_dofadr[fixture_joint]
    d.qpos[:2]=inverse(x0,y0);mujoco.mj_forward(m,d)
    dt=m.opt.timestep;stride=round(.001/dt);target=case.get('target',10.);duration=case.get('duration',8.4)
    torque_method=case['method'] in ['ph_force_tcst','oracle_dfc','oracle_tcst']
    if torque_method:
        nominal=model(dict(stiffness=5000.,mass_scale=1.,damping=15.))
        torque_parameters=dict(case.get('controller_parameters',{}))
        outer_delay=case.get('torque_delay_location')=='outer'
        if outer_delay:torque_parameters['outer_delay']=case.get('delay',.005)
        control=TCSTTorqueController(case['method'],nominal,kinematics,torque_parameters)
        control.adaptive=False
        torque_queue=collections.deque([np.zeros(2) for _ in range(0 if outer_delay else round(case.get('delay',.005)/.001))])
        torque_requested=np.zeros(2)
    else:
        control=Comparator(case['method'],target,x0,case.get('controller_parameters'))
    queue=collections.deque([0.]*round(case.get('delay',.005)/.001))
    actuator=np.zeros(2);integral=np.zeros(2);command=force=0.;noise=np.random.default_rng(case.get('seed',23000))
    acquisition=case.get('force_acquisition','instantaneous')
    if acquisition not in ['instantaneous','interval_mean']:raise ValueError(acquisition)
    force_sum=0.;force_count=0
    tip=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'tip');surface=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'surface')
    variable_reference=case.get('force_profile','constant')!='constant'
    reference_trace=np.empty(round(duration/dt))
    if variable_reference and case['method']=='force_cbf':control.controller.limit=1.5*(target+5.)
    steps=round(duration/dt);trace=np.empty((steps,15));jointlimit=np.array([35.,25.]);saturated=0
    for n in range(steps):
        t=d.time;x,J=kinematics(d.qpos[:2]);v=J@d.qvel[:2]
        reference=force_reference(t,target,case.get('force_profile','constant'),case.get('reference_frequency',.6),case.get('reference_amplitude',4.))
        reference_trace[n]=reference
        if n%stride==0:
            if variable_reference:
                control.target=reference
                if control.adaptive:control.controller.target=reference
            sensed=force_sum/force_count if acquisition=='interval_mean' and force_count else force
            f=sensed+noise.normal(0,case.get('force_noise',.1));xm=x[0]+noise.normal(0,case.get('position_noise',1e-6));vm=v[0]+noise.normal(0,case.get('velocity_noise',.0002))
            force_sum=0.;force_count=0
            if torque_method:
                phase=max(0.,t-1.2)
                yr=y0+case.get('tangent_amplitude',.04)*(1-math.cos(2*math.pi*case.get('tangent_rate',.3)*phase))
                qm=np.array(inverse(xm,x[1]));_,Jm=kinematics(qm)
                qvm=np.linalg.solve(Jm,np.array([vm,v[1]]))
                torque_queue.append(control.step(f,qm,qvm,reference,yr,t))
                torque_requested=torque_queue.popleft()
            else:
                u=control.step(f,xm,vm,t);queue.append(u);command=queue.popleft()
        phase=max(0.,t-1.2);rate=case.get('tangent_rate',.3);amplitude=case.get('tangent_amplitude',.04)
        yr=y0+amplitude*(1-math.cos(2*math.pi*rate*phase));yd=amplitude*2*math.pi*rate*math.sin(2*math.pi*rate*phase)
        vref=np.array([command,np.clip(yd+8*(yr-x[1]),-.12,.12)])
        error=vref-v;ii=integral+3000*error*dt
        wrench=180*error+ii;raw=J.T@wrench
        # Shared Cartesian PI, torque-limited motors and actuator dynamics.
        if torque_method:
            raw=torque_requested
            requested=np.clip(raw,-jointlimit,jointlimit)
        else:
            if np.all(abs(raw)<jointlimit):integral=ii
            requested=np.clip(J.T@(180*error+integral),-jointlimit,jointlimit)
        saturated+=int(np.any(abs(raw)>=jointlimit))
        actuator+=(1-math.exp(-dt/case.get('actuator_lag',.004)))*(requested-actuator)
        d.ctrl[:]=actuator
        support,speed=anchor(t,case.get('profile','multisine'));scale=case.get('motion_scale',1.)
        support*=scale;speed*=scale
        ratio=1.
        if case.get('material_change',True):
            smooth=lambda z:.5-.5*math.cos(math.pi*np.clip(z,0,1))
            ratio=1+(case.get('stiffness_ratio',.5)-1)*(smooth((t-3)/.5)-smooth((t-6)/.5))
        k=case['stiffness']*ratio;c=case.get('damping',12.)
        m.jnt_stiffness[fixture_joint]=k;m.dof_damping[fixture_dof]=c
        d.qfrc_applied[fixture_dof]=k*support+c*speed
        mujoco.mj_step(m,d)
        force=0.
        for j in range(d.ncon):
            contact=d.contact[j]
            if {int(contact.geom1),int(contact.geom2)}=={tip,surface}:
                cf=np.zeros(6);mujoco.mj_contactForce(m,d,j,cf)
                # The box face normal is x; include its actual contact-frame projection.
                force+=abs((contact.frame.reshape(3,3).T@cf[:3])[0])
        force_sum+=force;force_count+=1
        pos,JJ=kinematics(d.qpos[:2]);vel=JJ@d.qvel[:2]
        diag=control.controller.diagnostics() if control.adaptive else {}
        filtered_force=control.f if torque_method else (control.controller.f if control.adaptive else control.force)
        trace[n]=[d.time,force,pos[0],pos[1],vel[0],vel[1],command,yr,d.qpos[fixture_q],k,diag.get('stiffness',0.),diag.get('accepted_updates',0.),*d.ctrl,filtered_force]
    a=trace;dynamic=a[:,0]>1.2
    result=dict(case=case,nmae=float(np.mean(abs(a[dynamic,1]-target))/target),peak_ratio=float(a[:,1].max()/target),
        loss_seconds=float(np.sum(a[dynamic,1]<.5)*dt),tangent_rmse=float(np.sqrt(np.mean((a[dynamic,3]-a[dynamic,7])**2))),
        torque_saturation_fraction=saturated/len(a),accepted_updates=int(a[-1,11]),runtime=time.perf_counter()-start)
    if variable_reference:
        result.update(nmae=float(np.mean(abs(a[dynamic,1]-reference_trace[dynamic]))/np.mean(reference_trace[dynamic])),
            peak_ratio=float(np.max(a[:,1]/reference_trace)),absolute_peak=float(a[:,1].max()),
            reference_rms_error=float(np.sqrt(np.mean((a[dynamic,1]-reference_trace[dynamic])**2))))
    error=a[dynamic,1]-reference_trace[dynamic]
    relative_error=abs(error)/reference_trace[dynamic]
    loss=a[dynamic,1]<.5
    edges=np.diff(np.r_[False,loss,False].astype(int))
    lengths=(np.flatnonzero(edges==-1)-np.flatnonzero(edges==1))*dt
    command_samples=a[::stride,6]
    result.update(force_rmse=float(np.sqrt(np.mean(error**2))),
        force_error_p95=float(np.quantile(abs(error),.95)),
        force_error_max=float(np.max(abs(error))),absolute_peak=float(np.max(a[:,1])),
        relative_error_p95=float(np.quantile(relative_error,.95)),
        within_ten_percent_fraction=float(np.mean(relative_error<=.1)),
        longest_contact_loss=float(np.max(lengths,initial=0.)),contact_loss_events=int(len(lengths)),
        loss_fraction=float(np.mean(loss)),
        command_rms=float(np.sqrt(np.mean(command_samples**2))),
        command_saturation_fraction=float(np.mean(abs(command_samples)>=.06-1e-12)),
        command_total_variation=float(np.sum(abs(np.diff(command_samples)))),
        torque_rms=float(np.sqrt(np.mean(np.sum(a[dynamic,12:14]**2,axis=1)))),
        late_nmae=float(np.mean(abs(a[a[:,0]>duration/2,1]-reference_trace[a[:,0]>duration/2]))/np.mean(reference_trace[a[:,0]>duration/2])))
    assert np.isfinite(a).all()
    if torque_method:
        # Velocity commands are not defined for these native torque interfaces.
        for field in ['command_rms','command_saturation_fraction','command_total_variation']:
            result[field]=None
        result['actuation_interface']='torque'
        if control.residual_samples:
            residual=np.asarray(control.residual_samples)
            result['residual_abs_mean']=float(np.mean(abs(residual)))
            result['residual_nonzero_fraction']=float(np.mean(abs(residual)>1e-12))
            result['residual_optimizer_median_seconds']=float(np.median(control.inference_seconds))
            result['residual_optimizer_p99_seconds']=float(np.quantile(control.inference_seconds,.99))
    if save:
        out=Path(output_dir or ROOT/'output')/'traces';out.mkdir(parents=True,exist_ok=True)
        key=hashlib.sha256(('geometry_v2'+json.dumps(case,sort_keys=True)).encode()).hexdigest()[:16];path=out/(key+'.npz')
        np.savez_compressed(path,trace=a,reference=reference_trace,columns=np.array(['t','force','x','y','vx','vy','command','yref','fixture_x','true_k','estimated_k','accepted','tau1','tau2','feedback_force']))
        result['trace_path']='traces/'+path.name
        if torque_method and control.rows and case.get('collect_model',False):
            training_path=path.with_name(path.stem+'_transitions.npz')
            np.savez_compressed(training_path,measurements=np.asarray(control.rows))
            result['transition_path']='traces/'+training_path.name
    return result
