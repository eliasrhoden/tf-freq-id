
import casadi
from typing import Protocol





class TfElement:

    def __init__(self,element=None):
        # int in touple = inverse
        self.mdl_chain = []

        if element is not None:
            self.mdl_chain.append((False,element))

    def get_log_mag(self, w):
        pass

    def get_phase(self, w):
        pass

    def __mul__(self,other):
        mdl = TfElement()
        mdl.mdl_chain.extend(self.mdl_chain)
        mdl.mdl_chain.extend(other.mdl_chain)
        return mdl


    def __rtruediv__(self,other):
        mdl = TfElement()

        for e in self.mdl_chain:
            inv = not e[0]
            mdl.mdl_chain.append((inv,e[1])) 
        return mdl

    def __truediv__(self,other):
        mdl = TfElement()
        mdl.mdl_chain.extend(self.mdl_chain)

        for e in other.mdl_chain:
            inv = not e[0]
            mdl.mdl_chain.append((inv,e[1])) 
        return mdl

    def __str__(self):
        out = ""
        for e in self.mdl_chain:
            if e[0]:
                out += "1/"
            out += str(e[1]) + " "
        return out



class Integrator(TfElement):

    def __init__(self):
        super().__init__(self)

    def get_log_mag(self, w):
        return -1.0*casadi.log10(w)

    def get_phase(self, w):
        return -1.0*casadi.atan2(w, 1)


class PT1(TfElement):
    # Represents a First order polynominal
    # G(s) = (s*tau + 1)
    def __init__(self, tau):
        self.tau = tau
        super().__init__(self)

    def get_log_mag(self, w):
        return casadi.log10(casadi.sqrt(1 + (w*self.tau)**2))

    def get_phase(self, w):
        return casadi.atan2(w*self.tau, 1)

    def __str__(self):
        return f"PT1({self.tau})"


class PT2(TfElement):
    # Represents a second order polynominal
    # G(s) = 1/wd^2 * (s^2 + 2*s*dd*wd + wd^2)

    def __init__(self, wd, dd):
        self.wd = wd
        self.dd = dd
        super().__init__(self)

    def _parts(self, wi):
        t2Re = (self.wd**2 - wi**2)
        t2Im = (2*self.wd*wi*self.dd)
        return t2Re, t2Im

    def get_log_mag(self, w):
        t2Re, t2Im = self._parts(w)
        t2 = t2Re**2 + t2Im**2
        return casadi.log10(casadi.sqrt(t2)) - 2*casadi.log10(self.wd)

    def get_phase(self, w):
        t2Re, t2Im = self._parts(w)
        return casadi.atan2(t2Im, t2Re)

    def __str__(self):
        return f"PT2({self.wd},{self.dd})"
