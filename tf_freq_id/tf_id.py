import control as ctrl
import numpy as np
import casadi
import dataclasses
from .tfElements import PT2, PT1, TfElement



def d2c(Gd, Ts):
    """
    Converts a discrete time transfer function to continuous time
    https://github.com/tomasmckelvey/fsid/blob/master/python/fsid.py
    """        
    Gd_ss = ctrl.tf2ss(Gd)
    a, b, c, d = Gd_ss.A, Gd_ss.B, Gd_ss.C, Gd_ss.D
    n =  np.shape(a)[0]
    ainv = np.linalg.inv(np.eye(n)+a)
    ac = np.dot(ainv, a-np.eye(n))*2/Ts
    bc = np.dot(ainv, b)*2/np.sqrt(Ts)
    cc = np.dot(c, ainv)*2/np.sqrt(Ts)
    dc = d - np.linalg.multi_dot([c, ainv, b]) 
    return ctrl.ss(ac, bc, cc, dc)


def fit(w, Y, dt, model_order=None, order_tol=0.1):
    """
    Fit a transfer function model to frequency response data
    """

    if model_order is None:
        for i in range(10):
            Ge, error = self._fit_tf(w,Y, dt, i, order_tol)
            if error < order_tol:
                break
    else:
        Ge, error = self._fit_tf(w,Y, dt, model_order, order_tol)

    return Ge, error



def _fit_tf(w, Y, dt, model_order, order_tol):

    n = model_order

    M = []
    Y = []

    for i,yi in enumerate(GY):

        Ri = mag[i]
        phi = phase[i]
        wk = w[i]   

        Oi = wk*Ts

        row1 = []

        # Real part

        #R1 = Ri*(np.cos(phi) + np.cos(phi - Oi)*a[1] + np.cos(phi - 2*Oi)*a[2])
        #R2 = b[0] + np.cos(-Oi)*b[1] + np.cos(-2*Oi)*b[2]

        for i in range(1,n):
            row1.append(-Ri*np.cos(phi - Oi*i))

        for i in range(0,n):
            if i == 0:
                row1.append(1.0)
            else:
                row1.append(np.cos(-Oi*i))

        row2 = []
        # Imag part
        #I1 = Ri*(np.sin(phi) + np.sin(phi - Oi)*a[1] + np.sin(phi - 2*Oi)*a[2])
        #I2 = np.sin(-Oi)*b[1] + np.sin(-2*Oi)*b[2] # NO b0!

        for i in range(1,n):
            row2.append(-Ri*np.sin(phi - Oi*i))

        for i in range(0,n):
            if i == 0:
                row2.append(0.0)
            else:
                row2.append(np.sin(-Oi*i))

        M.append(row1)
        M.append(row2)

        Y.append(Ri*np.cos(phi))
        Y.append(Ri*np.sin(phi))

    sol = np.linalg.lstsq(M,Y,rcond=None)
    theta = sol[0]
    theta

    a = theta[0:n-1]
    b = theta[n-1:]

    a = np.r_[1,a]

    Ge = ctrl.tf(b,a,Ts)
    Ge = ctrl.minreal(Ge,tol=0.1)

    Y_mdl = ctrl.freqresp(Ge,w)

    error = np.mean(np.abs(Y - Y_mdl))

    return d2c(Ge,Ts), error
