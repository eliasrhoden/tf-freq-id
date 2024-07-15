

import control as ctrl
import numpy as np
import casadi

import dataclasses
from .tfElements import PT2, PT1, TfElement


@dataclasses.dataclass
class ModelOrder:
    NoPT2_numer:int
    NoPT2_denom:int
    NoPT1_numer:int
    NoPT1_denom:int
    integrator_sys:int


class TfIdent:

    def __init__(self, model_order:ModelOrder):
        self.model_order = model_order
        self.opti = casadi.Opti()

        self.num_PT2s:list[TfElement] = []
        self.den_PT2s:list[TfElement] = []

        self.num_PT1s:list[TfElement] = []
        self.den_PT1s:list[TfElement] = []

    def _create_tf_elements(self, sol=None, wc_limits = None):
        # a bit of copy paste here...
        num_PT2s_new = []
        for i in range(self.model_order.NoPT2_numer):
            if sol:
                oldPt2 = self.num_PT2s[i]
                w0 = sol.value(oldPt2.wd)
                d0 =  sol.value(oldPt2.dd)
                pt2 = self._create_PT2_var(w0=w0,d0=d0,delta_lim=0.05,wc_limits=wc_limits)
            else:
                pt2 = self._create_PT2_var(wc_limits=wc_limits)
            num_PT2s_new.append(pt2)
        self.num_PT2s = num_PT2s_new


        den_PT2s_new = []
        for i in range(self.model_order.NoPT2_denom):

            if sol:
                oldPt2 = self.den_PT2s[i]
                w0 = sol.value(oldPt2.wd)
                d0 =  sol.value(oldPt2.dd)
                pt2 = self._create_PT2_var(w0=w0,d0=d0,delta_lim=0.05,wc_limits=wc_limits)
            else:
                pt2 = self._create_PT2_var(wc_limits=wc_limits)

            den_PT2s_new.append(pt2)

        self.den_PT2s = den_PT2s_new


        self.num_PT1s = [self._create_PT1_var() for _ in range(self.model_order.NoPT1_numer)]
        self.den_PT1s = [self._create_PT1_var() for _ in range(self.model_order.NoPT1_denom)]


    def _create_list_of_tf_elem(self,sys_create, nr_elems)->list[TfElement]:
        elems:list[TfElement] = []
        for _ in range(nr_elems):
            elems.append(sys_create())
        return elems

    def _create_PT1_var(self, tau0 = 10):

        opti = self.opti

        tau = opti.variable()
        opti.subject_to(tau>=0.0001)
        opti.subject_to(tau <= 100)
        opti.set_initial(tau,tau0)

        return PT1(tau)

    def _create_PT2_var(self,delta_lim=0.35,w0=100,d0=0.7,wc_limits=None):
            opti = self.opti

            wd = opti.variable()
            dd = opti.variable()
        

            if wc_limits:
                if wc_limits[0] < 1.0:
                    raise Exception("Very low wc limit in PT2 element!")
                opti.subject_to(wd >= wc_limits[0])
                opti.subject_to(wd <= wc_limits[1]*0.9)
            else:
                opti.subject_to(wd>=10)
                opti.subject_to(wd <= 1e5)
            opti.set_initial(wd,w0)

            opti.subject_to(dd>=delta_lim)
            # A lower damping introduces non-convexity, so might be a trick to keep up our sleves
            # to raise this lower limit if we run into numerical issues
            opti.subject_to(dd <= 1.01)
            opti.set_initial(dd,d0)

            return PT2(wd,dd)

    def _formulate_cost(self,mag,phase,omega):
        # Formulate cost
        J = 0


        for i in range(len(omega)):

            wi = omega[i]

            magi = casadi.log10(self.k)
            phi = 0

            for pt2 in self.num_PT2s:
                magi += pt2.get_log_mag(wi)
                phi += pt2.get_phase(wi)

            for pt2 in self.den_PT2s:
                magi -= pt2.get_log_mag(wi)
                phi -= pt2.get_phase(wi)

            for pt1 in self.num_PT1s:
                magi += pt1.get_log_mag(wi)
                phi += pt1.get_phase(wi)

            for pt1 in self.den_PT1s:
                magi -= pt1.get_log_mag(wi)
                phi -= pt1.get_phase(wi)

            if self.model_order.integrator_sys:
                magi -= casadi.log10(wi)
                phi -=  np.pi/2

            mag_true = mag[i]
            m_err = (mag_true - magi)**2
            ph_err = 180/np.pi*(phase[i] - phi)**2

            J_scale = 1/wi

            J += J_scale*(1e3*m_err + ph_err)
        return J


    def identify_tf(self,mag, phase, omega):
        """
        Identifies the transfer function of the frequency data
        mag: Magnitude
        phase: phase, in radians
        omega: frequency in rad/s
        """
        mag = np.log10(mag)

        opti = self.opti

        # We don't want to estimate a model with a too small gain (k),
        # maybe want to add a pre-check and scale up the freq. data.
        k = opti.variable()
        opti.subject_to(k>=0.7)
        opti.subject_to(k <= 1e9)
        opti.set_initial(k,100)
        self.k = k

        # Create the model elements
        max_omega = np.max(omega)
        min_omega = np.min(omega)
        wc_limits = [min_omega,max_omega]
        self._create_tf_elements(wc_limits=wc_limits)
        J = self._formulate_cost(mag,phase,omega)

        opti.minimize(J)

        opti.solver('ipopt')
        sol = opti.solve()

        self._check_sol(sol)

        G0 = self._create_tf_from_sol(sol)

        # Prepare second solve
        opti.set_initial(self.k,sol.value(self.k))
        self._create_tf_elements(sol,wc_limits=wc_limits)
        J = self._formulate_cost(mag,phase,omega)

        opti.minimize(J)

        opti.solver('ipopt')
        sol = opti.solve()

        self._check_sol(sol)

        G1 = self._create_tf_from_sol(sol)

        return G0,G1




    def _check_sol(self,sol):
        if np.abs(sol.value(self.k)) < 0.9:
            print(f"k is {sol.value(self.k)}, might want to scale up your input data for better numerical accuracy")

        pt2s = self.num_PT2s.copy()
        pt2s.extend(self.den_PT2s.copy())

        for p in pt2s:
            if sol.value(p.dd) < 0.1:
                print("You have one PT2 element with delta below 0.1, might want to do two iterations with a higher lower bound on d for better accuracy")

    def _pt2_tf(self,sol,p:PT2):
        w = sol.value(p.wd)
        d = sol.value(p.dd)
        return ctrl.tf([1,2*d*w,w**2],[w**2])

    def _pt1_tf(self,sol,p:PT1):
        tau = sol.value(p.tau)
        return ctrl.tf([tau,1],[1])

    def _create_tf_from_sol(self,sol):

        G = sol.value(self.k)

        for pt2 in self.num_PT2s:
            G *= self._pt2_tf(sol,pt2)

        for pt2 in self.den_PT2s:
            G *= 1/self._pt2_tf(sol,pt2)

        for pt1 in self.num_PT1s:
            G *= self._pt1_tf(sol,pt1)

        for pt1 in self.den_PT1s:
            G *= 1/self._pt1_tf(sol,pt1)

        if self.model_order.integrator_sys:
            G *= ctrl.tf([1],[1,0])

        return ctrl.minreal(G,verbose=False)






