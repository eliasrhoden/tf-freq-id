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


def fit(w, mag, phase, model_order=None, order_tol=0.5):
    """
    Fit a transfer function model to frequency response data
    """
    w_max = np.max(w) * 5
    f_max = w_max/(2*np.pi)
    Ts = 1/(2*f_max)

    if model_order is None:
        for i in range(2,10):
            Ge,Gd, error = _fit_tf(w,mag, phase, Ts, i)
            print(error)
            if error < order_tol:
                break
    else:
        Ge,Gd, error = _fit_tf(w,mag, phase, Ts, model_order)

    return Ge,Gd, error



def _fit_tf(w, mag, phase, Ts, n):

    M = []
    Y = []
    alpha = 1e7

    for i,wi in enumerate(w):

        Ri = mag[i]
        #Ri = np.abs(yi)
        phi = phase[i]
        #phi = np.angle(yi)
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

        row1 = np.array(row1)*alpha
        row2 = np.array(row2)*alpha

        M.append(row1)
        M.append(row2)

        Y.append(Ri*np.cos(phi)*alpha)
        Y.append(Ri*np.sin(phi)*alpha)

    w_arr = []
    for wi in w:
        #wscale = 1/wi * 1/np.abs(Ri)
        #wscale = 1/np.abs(Ri)
        wscale = 1/wi 
        w_arr.append(wscale)
        w_arr.append(wscale)

    W = np.diag(w_arr)
    W *= 1/w_arr[-1]
    W *= 1e3

    #sol = np.linalg.lstsq(M,Y,rcond=None)

    #W = np.diag(np.ones(len(w_arr)))*1e7

    M = np.array(M)
    Y= np.array(Y)

    Ms = (M.T@W@M)
    Ys = (M.T@W@Y)
    sol = np.linalg.lstsq(Ms,Ys,rcond=None)

    theta = sol[0]
    theta

    a = theta[0:n-1]
    b = theta[n-1:]

    a = np.r_[1,a]

    Ge = ctrl.tf(b,a,Ts)
    Ge = ctrl.minreal(Ge,tol=0.001,verbose=False)

    Y_mdl = ctrl.freqresp(Ge,w).fresp[0][0]

    Ymeas = mag*np.exp(1j*phase)

    error = np.mean(np.abs(Ymeas - Y_mdl))

    return d2c(Ge,Ts),Ge, error
