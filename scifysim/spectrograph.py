# This module is designed 
import sympy as sp
import numpy as np
import numexpr as ne
from kernuller import fprint
from .utilities import ee

import logging

logit = logging.getLogger(__name__)

class integrator():
    def __init__(self, config=None,
                 keepall=True,
                 n_sources=4,
                 infinite_well=False):
        """
        Contains the pixel model of the detector.
        Currently it does not implement a system gain (1ADU = 1photoelectron)
        
        **Parameters:**
        
        * keepall    : [boolean] Whether to keep each of the steps accumulated
        * eta        :            The quantum efficiency
        * ron        : [photoelectrons] The read-out noise 
        * ENF        :            The Excess Noise Factor 1 if not applicable
        * mgain      :            The amplification gain (for EMCCDs or APD/eAPD sensors)
        * well       : [photelectrons] The well depth
        * n_sources  :            The number of different sources propagated through the instrument
        
        
        """
        
        self.keepall = keepall
        self.n_sources = n_sources
        self.exposure = 0.
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
            if np.isclose(well, 0.) or infinite_well:
                self.well = np.float32(1.e20) #np.inf
            else:
                self.well = well
        self.reset()
        
    def accumulate(self,value):
        """
        Accumulates some signal.
        
        **Parameters:**
        
        * value   : The signals to accumulate
        """
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
    def get_total(self, spectrograph=None, t_exp=None,
                 n_pixsplit=None):
        """
        Made a little bit complicated by the ability to simulate CRED1 camera
        
        **Parameters:**
        
        * spectrograph  : A spectrograph object to map the spectra on
          a 2D detector
        * t_exp         : [s]  Integration time to take into account dark current
          This is only used if self.exposure is close to 0. (if 
          the )
        * n_pixsplit    : The number of pixels 
        """
        if n_pixsplit is not None: # Splitting the signal over a number pixels
            thepixels = self.acc.copy()
            thepixels = ((thepixels/n_pixsplit)[None,:,:]*np.ones(n_pixsplit)[:,None,None])
        else:
            thepixels = self.acc.copy()
        if np.isclose(self.exposure, 0., atol=1.e-8):
            self.exposure = t_exp
        obtained_dark = self.dark * self.exposure * self.mgain
        if spectrograph is not None:
            acc = spectrograph.get_spectrum_image(thepixels)
        else:
            acc = thepixels
        electrons = acc * self.eta * self.mgain
        electrons = electrons + obtained_dark
        expectancy = electrons.copy()
        electrons = np.random.poisson(lam=electrons*self.ENF)/self.ENF
        electrons = np.clip(electrons, 0, self.well)
        read = electrons + np.random.normal(size=electrons.shape, scale=self.ron)
        if n_pixsplit is not None: # Binning the pixels again
            read = np.sum(read, axis=0)
        self.forensics = {"Expectancy": expectancy,
                         "Read noise": self.ron,
                         "Dark signal": obtained_dark}
        return read
    def get_static(self):
        """
        Returns the total value of integrated light from static sources.
        Not affected by noises
        """
        static = np.array(self.static)
        total_static = self.n_subexps * static.sum(axis=0)
        return total_static
    
    def get_starlight(self):
        return self.starlight.sum(axis=0)    
    def get_planetlight(self):
        return self.planetlight.sum(axis=0)
    
    def reset(self,):
        """
        Reset the values of the accumulation of signal in the detector.
        """
        self.vals = []
        self.acc = 0.
        self.runs = 0
        self.forensics = {}
        self.mean = None
        self.std = None
        self.exposure = 0.
        self.starlight = None
        self.planetlight = None
        
    def prepare_t_exp_base(self):
        """
        Computes some basic laws regarding exposure times
        for the current configuration:
        
        **Parameters:**
        
        - self.expr_snr_t       : expression of SNR as function of time
          which also gets printed
        - self.expr_t_for_snr   : time to get a given SNR
        - self.expr_well_fraction : Fraction of well filled by the number of Ph considered
        - self.expr_t_max       : Maximum exposure time to fill the well
        - self.t_exp_base       : lambdified version of expr_t_for_snr
        - self.snr_t            : lambdified version of expr_sr_t
        - self.well_fraction    : lambdified version of expr_well_faction 
        - self.t_exp_max        : lambdified version of expr_t_max
        
        .. code-block::
        
            eta, f_planet, n_pix, f_tot, ron, =  sp.symbols("eta, f_{planet}, n_p, f_{tot}, ron", real=True)
            n_planet, n_tot, t_exp0, t_exp = sp.symbols("n_{planet}, n_{tot}, t_{esxp0}, t_{exp}", real=True)
        
        """
        eta, f_planet, n_pix, f_tot, ron, =  sp.symbols("eta, f_{planet}, n_p, f_{tot}, ron", real=True)
        n_planet, n_tot, t_exp0, t_exp = sp.symbols("n_{planet}, n_{tot}, t_{esxp0}, t_{exp}", real=True)
        SNR = sp.symbols("SNR", real=True)
        mysubs = [(ron, self.ron),
                 (eta, self.eta),
                 ]
        
        self.expr_snr_t = eta*f_planet*t_exp/(sp.sqrt(eta*f_tot*t_exp + n_pix*ron**2))
        self.expr_t_for_snr = sp.solve(self.expr_snr_t - 1, t_exp)[0]
        self.expr_well_fraction = eta*n_tot/n_pix/self.well
        self.expr_t_max = self.well/(eta*f_tot/n_pix)
        self.t_exp_base = sp.lambdify((f_planet, f_tot, n_pix),
                         self.expr_t_for_snr.subs(mysubs), modules="numpy")
        fprint(self.expr_snr_t)
        self.snr_t = sp.lambdify((f_planet, f_tot, t_exp, n_pix),
                                 self.expr_snr_t.subs(mysubs), modules="numpy")
        self.well_fraction = sp.lambdify((n_tot, n_pix),
                                         self.expr_well_fraction.subs(mysubs),
                                         modules="numpy")
        self.t_exp_max = sp.lambdify((f_tot, n_pix),
                                    self.expr_t_max.subs(mysubs),
                                    modules="numpy")
    def consolidate_metrologic(self, ):
        """
        Compiles the results of metrologic exposure. 
        
        **Paramters compiled:**
        
        * ``self.nsamples``    : The number of integration steps
        * ``self.summed_signal`` : The total integration of 
          all sources throughout the time steps
        * ``self.total_summed_signal`` : The total integration over
          both sources and time steps
        * ``self.sums``        : The list of signals from each of the sources
          summed over the time steps
        * ``self.source_labels`` : The labels of the sources
          corresponding to sums.
        
        """
        self.nsamples = self.n_subexps
        sumstatic = np.sum(self.static, axis=0)
        self.summed_signal = sumstatic[None,:,:]+self.starlight+\
                            self.planetlight
        self.total_summed_signal = self.summed_signal.sum(axis=0)

        self.sums = []
        self.source_labels = []
        for i, astatic in enumerate(self.static):
            self.source_labels.append(self.static_list[i])
            self.sums.append(astatic * self.nsamples)
        self.sums.append(self.starlight.sum(axis=0))
        self.source_labels.append("Starlight")
        self.sums.append(self.planetlight.sum(axis=0))
        self.source_labels.append("Planet")
        

class spectrograph(object):
    def __init__(self, aconfig, lamb_range, n_chan=8):
        """
        This class describres the behaviour of a simple spectrograph.
        
        
        """
        self.lamb_range = lamb_range
        
        over = aconfig.getint("spectrograph", "oversampling")
        val_lamb0 = aconfig.getfloat("spectrograph", "lamb0") # The wl at which psf is parameterized
        val_sigmax0 = over*aconfig.getfloat("spectrograph", "sigmax0")
        val_sigmay0 = over*aconfig.getfloat("spectrograph", "sigmay0")
        val_x0 = over*aconfig.getfloat("spectrograph", "x0")
        val_y0 = over*aconfig.getfloat("spectrograph", "y0")
        val_delta_x = over*aconfig.getfloat("spectrograph", "delta_x")
        val_dispersion = aconfig.getfloat("spectrograph", "dispersion")
        self.over = over
        
        
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
        ishape = self.blank_sensor.shape
        for i, anoutput in enumerate(signal.T):
            cube = self.ee(self.xx[None,:,:], self.yy[None,:,:],
                anoutput[:,None,None],
                self.lamb_range[:,None,None],
                i)
            image += np.sum(cube, axis=0)
        
        if self.over != 1:
            image = image.reshape(self.over, ishape[0]//self.over, ishape[1]//self.over, self.over).sum(axis=(0,3))
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