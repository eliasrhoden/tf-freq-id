

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tf_freq_id
import control as ctrl 
import matplotlib.pyplot as plt
import numpy as np

def pt2(w,d):
    return ctrl.tf(w**2,[1,2*w*d,w**2])


G = pt2(10,0.2) *1/pt2(5,0.5)

Ts = 0.01 
Gd = G.sample(Ts,method='tustin')

print("CT Poles")
print(G.pole())

print("DT Poles")
dt_poles = (1+Gd.pole()*Ts/2)/(1-Gd.pole()*Ts/2)
dt_poles = 2/Ts*(Gd.pole() - 1)/(Gd.pole() + 1)
print(dt_poles)

Ge = tf_freq_id.d2c(Gd,Ts)

ctrl.bode_plot(G)
ctrl.bode_plot(Gd)
ctrl.bode_plot(Ge,linestyle='--')
plt.show()

