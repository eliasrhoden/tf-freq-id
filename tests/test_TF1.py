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

G = pt2(10,0.2) * 1/pt2(5,0.5)*pt2(500,0.1)

mag,phase,w = ctrl.bode(G,plot=False)

indx = np.arange(0,len(w),1)
indx = np.mod(indx,5)==0

ws = w[indx]
mags = mag[indx]
phases = phase[indx]

np.random.seed(1337)

mags = 10**(np.log10(mags) + 0.05*np.random.randn(len(mags)))
phases += np.random.randn(len(phases))*np.deg2rad(3)


if False:
    m = tf_freq_id.TfRefine()

    m0 = tf_freq_id.PT2(10,0.2)/tf_freq_id.PT2(5,0.2)

    G0,G1 = m.identify_tf(m0,mags,phases,ws)

    print(G0)

    plt.figure()
    ctrl.bode_plot(G)
    ctrl.bode_plot(G0)
    ctrl.bode_plot(G1,linestyle='--')

    plt.show()
   
Gm = (tf_freq_id.PT2(50,0.5)*tf_freq_id.PT2(500,0.5))/tf_freq_id.PT2(1000,0.5)

tfrefine = tf_freq_id.TfRefine()
G0,G1 = tfrefine.identify_tf(Gm,mags,phases,ws)

mag0,ph0,w0 = ctrl.bode(G0,plot=False,wrap_phase=True)
mag1,ph1,w1 = ctrl.bode(G1,plot=False,wrap_phase=True)
#magg = ctrl.mag2db(magg)
#phasg = np.rad2deg(phasg)

plt.figure()
plt.subplot(2,1,1)
plt.semilogx(w,ctrl.mag2db(mag))
plt.plot(ws, ctrl.mag2db(mags),'.-')
plt.plot(w0, ctrl.mag2db(mag0))
plt.plot(w1, ctrl.mag2db(mag1),'--')

plt.subplot(2,1,2)
plt.semilogx(w, phase)
plt.semilogx(ws, phases,'.-')
plt.semilogx(w0, ph0)
plt.semilogx(w1, ph1,'--')

plt.show()


