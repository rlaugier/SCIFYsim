# This module is designed 
import sympy as sp
import numpy as np
from kernuller import fprint

import logging

logit = logging.getLogger(__name__)

class integrator():
    def __init__(self, keepall=True,
                 ron=0., ENF=1., mgain=1.,
                 well=np.inf):
        self.keepall = True
        self.ron = ron
        self.ENF = ENF
        self.mgain = mgain
        self.well = well
        self.reset()
        
    def accumulate(self,value):
        self.acc += value
        self.runs += 1
        if self.keepall:
            self.vals.append(value)
    def compute_stats(self):
        if self.keepall:
            arr = np.array(self.vals)
            self.mean =  arr.sum(axis=0) / self.runs
            self.std = arr.std(axis=0)
            return self.mean, self.std
        else:
            pass
    def compute_noised(self):
        logit.warning("Noises not implemented yet")
        raise NotImplementedError("Noises not implemented yet")
    def get_total(self):
        """
        Made a little bit complicated bye the ability to simulate CRED1 camera
        """
        electrons = self.acc * self.mgain
        electrons = np.random.poisson(lam=electrons*self.ENF)/self.ENF
        electrons = np.clip(electrons, 0, self.well)
        read = electrons + np.random.normal(size=electrons.shape, scale=self.ron)
        return read
    def reset(self,):
        self.vals = []
        self.acc = 0
        self.runs = 0

class spectrograph(object):
    def __init__(self):
        """
        This class describres the behaviour of a simple spectrograph.
        
        
        """
        self.a, self.sigmax, self.sigmay = sp.symbols("a sigma_x sigma_y", real=True)
        self.x, self.x0, self.y,self.y0= sp.symbols("x x_0 y y_0", real=True)
        self.d = sp.symbols("d", real=True)
        
        self.lamb = sp.symbols("lambda")
        
        # Defining the main function controlling the PSF: au 2D Gaussian
        expterm = - ((self.self.x - self.x0)**2/(2 * self.sigmax**2) +(self.y - self.y0)**2/(2 * self.sigmay**2))
        self.gs = self.a*1/(self.sigmax*self.sigmay)*sp.exp(expterm)
        
        

        
class basic_integrator():
    def __init__(self, keepall=True):
        self.keepall = True
        self.reset()
    def accumulate(self,value):
        self.acc += value
        self.runs += 1
        if self.keepall:
            self.vals.append(value)
    def compute_stats(self):
        arr = np.array(self.acc)
        self.mean =  arr.sum(axis=0) / self.runs
        self.std = arr.std(axis=0)
        return self.mean, self.std
    def compute_noised(self):
        logit.warning("Noises not implemented yet")
        raise NotImplementedError("Noises not implemented yet")
    def get_total(self):
        return self.acc
    def reset(self,):
        self.vals = []
        self.acc = 0
        self.runs = 0