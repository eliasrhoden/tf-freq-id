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

wn = 10
dn = 0.2


G = pt2(10,0.2) *1/pt2(5,0.5)
#G = pt2(5,0.5)
G

mag,phase,w = ctrl.bode(G,plot=False)

indx = np.arange(0,len(w),1)
indx = np.mod(indx,5)==0

ws = w[indx]
mags = mag[indx]
phases = phase[indx]

np.random.seed(1337)

mags = 10**(np.log10(mags) + 0.1*np.random.randn(len(mags)))
phases += np.random.randn(len(phases))*np.deg2rad(5)

plt.figure()
plt.semilogx(w,mag)
plt.plot(ws, mags,'o-')
plt.figure()
plt.semilogx(w, phase)
plt.semilogx(ws, phases,'o-')

m = tf_freq_id.TfRefine()

m0 = tf_freq_id.PT2(10,0.2)/tf_freq_id.PT2(5,0.2)

G0,G1 = m.identify_tf(m0,mags,phases,ws)

print(G0)

plt.figure()
ctrl.bode_plot(G)
ctrl.bode_plot(G0)
ctrl.bode_plot(G1,linestyle='--')

plt.show()


