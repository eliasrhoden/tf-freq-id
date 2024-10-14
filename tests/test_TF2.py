import control as ctrl
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tf_freq_id

s = ctrl.tf('s')

def pt2(w,d):
    return ctrl.tf(w**2,[1,2*w*d,w**2])

G = pt2(40,0.1)*1/pt2(20,0.3) * pt2(600,0.05)*1/pt2(500,0.1) * ctrl.tf([1],[1,0])

G

mag,phase,w = ctrl.bode(G)

indx = np.arange(0,len(w),1)
indx = np.mod(indx,2)==0

ws = w[indx]
mags = np.array(mag[indx])
mags = np.copy(mags)
phases = np.array(phase[indx])
phases = np.copy(phases)


np.random.seed(1337)

N = len(mags)

n0 = 0.002*np.random.randn(N)
n1 = 0.006*np.random.randn(N)

if False:
    plt.figure()
    plt.plot(n0,label='N0')
    plt.plot(n1,label='N1')
    plt.legend()
    plt.show()

mags = 10**(np.log10(mags) + n0)
phases =  phases  + n1

#mag *= 1000

plt.figure()
plt.semilogx(w,ctrl.mag2db(mag))
plt.semilogx(ws, ctrl.mag2db(mags),'o-')
plt.ylabel('mag')

plt.figure()
plt.semilogx(w, phase)
plt.semilogx(ws, phases,'o-')
plt.ylabel('phase')

#plt.show()
##raise Exception("eee")

Gi,Gd,error = tf_freq_id.tf_id.fit(ws,mags,phases,model_order=6)

print(Gi)


G.name = 'True'
Gi.name = "Identified"

plt.figure()
ctrl.bode_plot(G)
ctrl.bode_plot(Gd)
ctrl.bode_plot(Gi)

plt.legend(['True','Identified'])

plt.show()


