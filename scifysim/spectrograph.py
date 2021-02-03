# This module is designed 
import sympy as sp
import numpy as np
from kernuller import fprint
from .utilities import ee

import logging

logit = logging.getLogger(__name__)

class integrator():
    def __init__(self, config=None, keepall=True, n_sources=4):
        """
        Contains the pixel model of the detector.
        Currently it does not implement a system gain (1ADU = 1photoelectron)
        
        keepall    : [boolean] Whether to keep each of the steps accumulated
        eta        :            The quantum efficiency
        ron        : [photoelectrons] The read-out noise 
        ENF        :            The Excess Noise Factor 1 if not applicable
        mgain      :            The amplification gain (for EMCCDs or APD/eAPD sensors)
        well       : [photelectrons] The well depth
        n_sources  :            The number of different sources propagated through the instrument
        
        
        """
        
        self.keepall = keepall
        self.n_sources = n_sources
        if config is None:
            self.eta=0.7
            self.ron=0.
            self.dark=0.05
            self.ENF=1.
            self.mgain=1.
            self.well=np.inf
        else:
            self.eta = config.getfloat("detector", "eta")
            self.ron = config.getfloat("detector", "ron")
            self.dark = config.getfloat("detector", "dark")
            self.ENF = config.getfloat("detector", "ENF")
            self.mgain = config.getfloat("detector", "mgain")
            well = config.getfloat("detector", "well")
            if np.isclose(well, 0.):
                self.well = np.inf
            else:
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
            # The number self.runs is actually incremented for eanch sources
            # at each of the computation steps
            self.mean =  arr.sum(axis=0) / (self.runs / self.n_sources)
            self.std = arr.std(axis=0)
            return self.mean, self.std
        else:
            pass
    def compute_noised(self):
        logit.warning("Noises not implemented yet")
        raise NotImplementedError("Noises not implemented yet")
    def get_total(self, spectrograph=None, t_exp=None):
        """
        Made a little bit complicated by the ability to simulate CRED1 camera
        spectrograph  : A spectrograph object to map the spectra on
                    a 2D detector
        t_exp         : [s]  Integration time to take into account dark current
        """
        obtained_dark = self.dark * t_exp * self.mgain
        if spectrograph is not None:
            acc = spectrograph.get_spectrum_image(self.acc)
        else:
            acc = self.acc
        electrons = acc * self.eta * self.mgain
        electrons = electrons + obtained_dark
        expectancy = electrons.copy()
        electrons = np.random.poisson(lam=electrons*self.ENF)/self.ENF
        electrons = np.clip(electrons, 0, self.well)
        read = electrons + np.random.normal(size=electrons.shape, scale=self.ron)
        self.forensics = {"Expectancy": expectancy,
                         "Read noise": self.ron,
                         "Dark signal": obtained_dark}
        return read
    def reset(self,):
        self.vals = []
        self.acc = 0.
        self.runs = 0
        self.forensics = {}
        self.mean = None
        self.std=None

class spectrograph(object):
    def __init__(self, aconfig, lamb_range, n_chan=8):
        """
        This class describres the behaviour of a simple spectrograph.
        
        
        """
        self.lamb_range = lamb_range
        
        val_lamb0 = aconfig.getfloat("spectrograph", "lamb0") # The wl at which psf is parameterized
        val_sigmax0 = aconfig.getfloat("spectrograph", "sigmax0")
        val_sigmay0 = aconfig.getfloat("spectrograph", "sigmay0")
        val_x0 = aconfig.getfloat("spectrograph", "x0")
        val_y0 = aconfig.getfloat("spectrograph", "y0")
        val_delta_x = aconfig.getfloat("spectrograph", "delta_x")
        val_dispersion = aconfig.getfloat("spectrograph", "dispersion")
        
        
        ssx = 2*val_x0 + (n_chan-1)*val_delta_x
        ssy = 2*val_y0 + val_dispersion * (np.max(lamb_range)-np.min(lamb_range))
        
        self.a, self.sigmax, self.sigmay = sp.symbols("a sigma_x sigma_y", real=True)
        self.x, self.x0, self.y,self.y0= sp.symbols("x x_0 y y_0", real=True)
        self.d = sp.symbols("d", real=True)
        
        self.lamb = sp.symbols("lambda")
        
        # Defining the main function controlling the PSF: au 2D Gaussian
        expterm = - ((self.x - self.x0)**2/(2 * self.sigmax**2) +(self.y - self.y0)**2/(2 * self.sigmay**2))
        self.gs = self.a*1/(self.sigmax*self.sigmay)*sp.exp(expterm)
        
        self.delta_x = sp.symbols("delta_x", real=True)
        self.k_out = sp.symbols("k_{out}", real=True) # output index
        self.thesubs = [(self.y0, val_y0 + val_dispersion * (self.lamb-val_lamb0)), #dispersion function
                  (self.x0, val_x0 + self.k_out * val_delta_x), # parameterized on output index
                  (self.sigmax, val_sigmax0*self.lamb/val_lamb0),
                  (self.sigmay, val_sigmay0*self.lamb/val_lamb0)]# 2.5+1.
        self.gsub = self.gs.subs(self.thesubs)
        self.ee = ee(self.gsub)
        self.ee.lambdify((self.x, self.y, self.a, self.lamb, self.k_out),
                                    modules="numpy")
        
        self.xx , self.yy = np.meshgrid(np.arange(ssx), np.arange(ssy))
        self.blank_sensor = np.zeros_like(self.xx)
        
    def get_spectrum_image(self, signal):
        image = self.blank_sensor.copy()
        for i, anoutput in enumerate(signal.T):
            cube = self.ee(self.xx[None,:,:], self.yy[None,:,:],
                anoutput[:,None,None],
                self.lamb_range[:,None,None],
                i)
            
            image += np.sum(cube, axis=0)
        return image
            
        
        
        

        
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