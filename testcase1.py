import control as ctrl
import matplotlib.pyplot as plt
import numpy as np

import tf_freq_id

s = ctrl.tf('s')

def pt2(w,d):
    return ctrl.tf(w**2,[1,2*w*d,w**2])


G = pt2(40,0.1)*1/pt2(20,0.3) * pt2(600,0.05)*1/pt2(500,0.1) * ctrl.tf([1],[1,0])
#G =  pt2(wd2,dd2)

G

mag,phase,w = ctrl.bode(G)
mag = np.log10(mag)


indx = np.arange(0,len(w),1)
indx = np.mod(indx,5)==0

ws = w[indx]
mags = mag[indx]
phases = phase[indx]

np.random.seed(1337)

mags += 0.1*np.random.randn(len(mags))

plt.figure()
plt.semilogx(w,mag)
plt.plot(ws, mags,'o-')
plt.figure()
plt.semilogx(w, phase)
plt.semilogx(ws, phases,'o-')

m = tf_freq_id.ModelOrder(2,2,0,0,True)

id = tf_freq_id.TfIdent(m)

G0,Gi = id.identify_tf(mags,phases,ws)
print(Gi)


G.name = 'True'
Gi.name = "Identified"

plt.figure()
ctrl.bode_plot(G)
ctrl.bode_plot(G0)
ctrl.bode_plot(Gi)

plt.legend(['True','First opt','Second opt'])

plt.show()


