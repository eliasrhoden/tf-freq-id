
import casadi
from typing import Protocol

class TfElement(Protocol):

    def get_log_mag(self,w):
        pass 

    def get_phase(self,w):
        pass

class Integrator(TfElement):
    def get_log_mag(self,w):
        return  -1.0*casadi.log10(w)

    def get_phase(self,w):
        return -1.0*casadi.atan2(w,1)

class PT1(TfElement):
    # Represents a First order polynominal
    # G(s) = (s*tau + 1)
    def __init__(self,tau):
        self.tau = tau

    def get_log_mag(self,w):
        return casadi.log10(casadi.sqrt(1 + (w*self.tau)**2))

    def get_phase(self,w):
        return casadi.atan2(w,1)

class PT2(TfElement):
    # Represents a second order polynominal
    # G(s) = 1/wd^2 * (s^2 + 2*s*dd*wd + wd^2)
    def __init__(self,wd,dd):
        self.wd = wd
        self.dd = dd

    def _parts(self,wi):
        t2Re = (self.wd**2 - wi**2)
        t2Im = (2*self.wd*wi*self.dd)
        return t2Re,t2Im

    def get_log_mag(self,w):
        t2Re,t2Im = self._parts(w)
        t2 = t2Re**2 + t2Im**2
        return casadi.log10(casadi.sqrt(t2)) - 2*casadi.log10(self.wd)

    def get_phase(self,w):
        t2Re,t2Im = self._parts(w)
        return casadi.atan2(t2Im,t2Re) 

