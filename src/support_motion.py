import math

def anchor(t,profile):
    # Moving spring support: a smooth disturbance, not a teleport of a dynamic body.
    if profile=='stationary': return 0.,0.
    if profile=='repeated_withdraw':
        if t<1.2:return 0.,0.
        z=(t-1.2)%1.8;a=.012;T=.25
        if z<T:return a*.5*(1-math.cos(math.pi*z/T)),a*.5*math.pi/T*math.sin(math.pi*z/T)
        if z<.5:return a,0.
        if z<.75:
            q=(z-.5)/T
            return a*.5*(1+math.cos(math.pi*q)),-a*.5*math.pi/T*math.sin(math.pi*q)
        return 0.,0.
    if profile=='chirp':
        if t<1.2:return 0.,0.
        z=t-1.2;phase=2*math.pi*(.45*z+.04*z*z);rate=2*math.pi*(.45+.08*z)
        return .003*(1-math.cos(phase)),.003*rate*math.sin(phase)
    if profile=='multisine':
        if t<1.2:return 0.,0.
        z=t-1.2;a1=.003;a2=.001;w1=2*math.pi*.9;w2=2*math.pi*2.3
        return a1*(1-math.cos(w1*z))+a2*(1-math.cos(w2*z)),a1*w1*math.sin(w1*z)+a2*w2*math.sin(w2*z)
    if profile=='oscillating':
        if t<1.2:return 0.,0.
        z=t-1.2; a=.004; w=2*math.pi*1.5
        return a*(1-math.cos(w*z)),a*w*math.sin(w*z)
    a=.008
    if 1.4<t<1.7:
        z=(t-1.4)/.3;return a*(.5-.5*math.cos(math.pi*z)),a*.5*math.pi/.3*math.sin(math.pi*z)
    if 1.7<=t<=2.2:return a,0.
    if 2.2<t<2.5:
        z=(t-2.2)/.3;return a*(.5+.5*math.cos(math.pi*z)),-a*.5*math.pi/.3*math.sin(math.pi*z)
    return 0.,0.
