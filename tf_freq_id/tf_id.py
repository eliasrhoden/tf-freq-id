

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

    def _create_tf_elements(self):
        self.num_PT2s = [self._create_PT2_var() for _ in range(self.model_order.NoPT2_numer)]
        self.den_PT2s = [self._create_PT2_var() for _ in range(self.model_order.NoPT2_denom)]
        self.num_PT1s = [self._create_PT1_var() for _ in range(self.model_order.NoPT1_numer)]
        self.den_PT1s = [self._create_PT1_var() for _ in range(self.model_order.NoPT1_denom)]


    def _create_list_of_tf_elem(self,sys_create, nr_elems)->list[TfElement]:
        elems:list[TfElement] = []
        for _ in range(nr_elems):
            elems.append(sys_create())
        return elems

    def _create_PT1_var(self):

        opti = self.opti

        tau = opti.variable()
        opti.subject_to(tau>=0.001)
        opti.subject_to(tau <= 1e9)
        opti.set_initial(tau,10)

        return PT1(tau)

    def _create_PT2_var(self):
            opti = self.opti

            wd = opti.variable()
            dd = opti.variable()

            opti.subject_to(wd>=1)
            opti.subject_to(wd <= 1e9)
            opti.set_initial(wd,100)

            opti.subject_to(dd>=0.07) 
            # A lower damping introduces non-convexity, so might be a trick to keep up our sleves 
            # to raise this lower limit if we run into numerical issues
            opti.subject_to(dd <= 1.01)
            opti.set_initial(dd,0.9)

            return PT2(wd,dd)


    def identify_tf(self,mag, phase, omega):

        opti = self.opti

        # We don't want to estimate a model with a too small gain (k),
        # maybe want to add a pre-check and scale up the freq. data.
        k = opti.variable()
        self.k = k
        opti.subject_to(k>=0.7)
        opti.subject_to(k <= 1e9)
        opti.set_initial(k,100)


        # Create the model elements
        self._create_tf_elements()

        # Formulate cost
        J = 0

        for i in range(len(omega)):

            wi = omega[i]

            magi = casadi.log10(k)
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

            J += 1e3*m_err + ph_err

        opti.minimize(J);

        opti.solver('ipopt');
        sol = opti.solve();

        if np.abs(sol.value(k)) < 0.9:
            print("Low value on k, might want to rescale data!")

        return self._create_tf_from_sol(sol)


    def _check_sol(self,sol):
        if np.abs(sol.value(k)) < 0.9:
            print(f"k is {sol.value(k)}, might want to scale up your input data for better numerical accuracy")

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






