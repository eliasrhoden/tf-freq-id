import control as ctrl
import numpy as np
import casadi
import dataclasses
from .tfElements import PT2, PT1, TfElement



class TfRefine:

    def __init__(self):
        self.opti = casadi.Opti()

        self.num_PT2s: list[TfElement] = []
        self.den_PT2s: list[TfElement] = []

        self.num_PT1s: list[TfElement] = []
        self.den_PT1s: list[TfElement] = []

        self.integrator_sys = False

    def _create_tf_elements(self, mdls:list[TfElement], wc_limits=None):

        # extract elements of inital mdl
        pt1_den0, pt1_num0, pt2_den0, pt2_num0 = self._split_init_mdl(mdls)

        # a bit of copy paste here...
        num_PT2s_new = []
        for p in pt2_num0:

            # numerator can have unstable PT2
            delt_lim = 0.05
            if p.dd < 0:
                delt_lim *= -1

            pt2 = self._create_PT2_var(
                w0=p.wd, d0=p.dd, delta_lim=delt_lim, wc_limits=wc_limits)

            num_PT2s_new.append(pt2)
        self.num_PT2s = num_PT2s_new

        den_PT2s_new = []
        for p in pt2_den0:

            pt2 = self._create_PT2_var(
                w0=p.wd, d0=p.dd, delta_lim=0.05, wc_limits=wc_limits)

            den_PT2s_new.append(pt2)

        self.den_PT2s = den_PT2s_new

        # maybe add support for unstable PT1 in numerator?
        self.num_PT1s = [self._create_PT1_var(p.tau,wc_limits=wc_limits)
                         for p in range(len(pt1_num0))]
        self.den_PT1s = [self._create_PT1_var(p.tau,wc_limits=wc_limits)
                         for p in range(len(pt1_den0))]

    def _split_init_mdl(self, mdls:list[TfElement]):

        pt1_den = []
        pt1_num = []

        pt2_den = []
        pt2_num = []

        for inv,mdl in mdls:

            if isinstance(mdl,PT1):
                if not inv:
                    pt1_den.append(mdl)
                else:
                    pt1_num.append(mdl)

            if isinstance(mdl,PT2):
                if not inv:
                    pt2_den.append(mdl)
                else:
                    pt2_num.append(mdl)

        return pt1_den, pt1_num, pt2_den, pt2_num


    def _create_list_of_tf_elem(self, sys_create, nr_elems) -> list[TfElement]:
        elems: list[TfElement] = []
        for _ in range(nr_elems):
            elems.append(sys_create())
        return elems

    def _create_PT1_var(self, tau0=10,wc_limits=None):

        opti = self.opti

        tau = opti.variable()

        if False:
            if wc_limits:
                opti.subject_to(tau >= 1/wc_limits[0])
                opti.subject_to(tau <= 1/wc_limits[1])
            else:
                opti.subject_to(tau >= 0.0001)
                opti.subject_to(tau <= 100)

        opti.set_initial(tau, tau0)

        return PT1(tau)

    def _create_PT2_var(self, delta_lim=0.35, w0=100, d0=0.7, wc_limits=None):
        opti = self.opti

        wd = opti.variable()
        dd = opti.variable()

        if wc_limits:
            if wc_limits[0] < 1.0:
                raise Exception("Very low wc limit in PT2 element!")
            opti.subject_to(wd >= wc_limits[0])
            opti.subject_to(wd <= wc_limits[1]*0.9)

        opti.set_initial(wd, w0)

        if delta_lim >= 0:
            # stable PT2
            opti.subject_to(dd >= delta_lim)
            # A lower damping introduces non-convexity, so might be a trick to keep up our sleves
            # to raise this lower limit if we run into numerical issues
            opti.subject_to(dd <= 1.01)

        else:
            # unstalbe PT2
            opti.subject_to(dd <= delta_lim)
            opti.subject_to(dd >= -1.01)

        opti.set_initial(dd, d0)

        return PT2(wd, dd)

    def _formulate_cost(self, mag, phase, omega):
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

            mag_true = mag[i]
            m_err = (mag_true - magi)**2
            ph_err = 180/np.pi*(phase[i] - phi)**2

            J_scale = 1/wi

            #J += J_scale*(1e3*m_err + ph_err)
            J += J_scale*(1e3*m_err)
        return J
    def _formulate_cost2(self, mag, phase, omega):
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

            mag_true = mag[i]
            m_err = (mag_true - magi)**2
            ph_err = 180/np.pi*(phase[i] - phi)**2

            rel_true = mag_true*np.cos(phase[i])
            img_true = mag_true*np.sin(phase[i])

            rel_mdl = magi*casadi.cos(phi)
            img_mdl = magi*casadi.sin(phi)

            err = (rel_true - rel_mdl)**2 + (img_true - img_mdl)**2

            J_scale = 1/wi

            #J += J_scale*(1e3*m_err + ph_err)
            J += J_scale*(err)
        return J

    def identify_tf(self, mdl, mag, phase, omega):
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
        opti.subject_to(k >= 0.7)
        opti.subject_to(k <= 1e9)
        opti.set_initial(k, np.mean(mag))
        self.k = k

        # Create the model elements
        max_omega = np.max(omega)
        min_omega = np.min(omega)
        wc_limits = [min_omega, max_omega]

        if isinstance(mdl,list):
            pass 
        else:
            mdl = mdl.mdl_chain

        self._create_tf_elements(mdl,wc_limits=wc_limits)
        J = self._formulate_cost2(mag, phase, omega)

        opti.minimize(J)

        opti.solver('ipopt')
        sol = opti.solve()

        self._check_sol(sol)

        G0 = self._create_tf_from_sol(initial=True)
        G1 = self._create_tf_from_sol(sol)

        return G0,G1

    def _check_sol(self, sol):
        return # todo
        if np.abs(sol.value(self.k)) < 0.9:
            print(f"k is {sol.value(self.k)}, might want to scale up your input data for better numerical accuracy")

        pt2s = self.num_PT2s.copy()
        pt2s.extend(self.den_PT2s.copy())

        for p in pt2s:
            if sol.value(p.dd) < 0.1:
                print("You have one PT2 element with delta below 0.1, might want to do two iterations with a higher lower bound on d for better accuracy")

    def _pt2_tf(self, p: PT2, sol=None, initial=False):
        if initial:
            opti = self.opti
            w = opti.value(p.wd,opti.initial())
            d = opti.value(p.dd,opti.initial())
        else:
            w = sol.value(p.wd)
            d = sol.value(p.dd)

        return ctrl.tf([1, 2*d*w, w**2], [w**2])

    def _pt1_tf(self, p: PT1, sol=None, initial=False):
        if initial:
            opti = self.opti
            tau = opti.value(p.tau,opti.initial())
        else:
            tau = sol.value(p.tau)
        return ctrl.tf([tau, 1], [1])

    def _create_tf_from_sol(self, sol=None, initial=False):

        if initial:
            opti = self.opti    
            #G  = self.opti.value(self.k,opti.initial)
            G = opti.debug.value(self.k, opti.initial())
        else:
            G = sol.value(self.k)
    

        for pt2 in self.num_PT2s:
            G *= self._pt2_tf(pt2,sol, initial=initial)

        for pt2 in self.den_PT2s:
            G *= 1/self._pt2_tf(pt2,sol, initial=initial)

        for pt1 in self.num_PT1s:
            G *= self._pt1_tf( pt1,sol, initial=initial)

        for pt1 in self.den_PT1s:
            G *= 1/self._pt1_tf(pt1,sol, initial=initial)


        return ctrl.minreal(G, verbose=False)
