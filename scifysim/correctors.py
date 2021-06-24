import numpy as np
import sympy as sp
from sympy.functions.elementary.piecewise import Piecewise
from kernuller import mas2rad, rad2mas
from . import utilities
from . import n_air
from astropy import constants
from astropy import units
import scipy.interpolate as interp
from pathlib import Path



parent = Path(__file__).parent.absolute()
znse_file = parent/"data/znse_index.csv"

# Utility functions for optimization
##################################################

from lmfit import Parameters, minimize
def extract_corrector_params(corrector, params):
    """
    Utility function to reconstruct the *b* and *c*
    vectors from the lmfit Parameters object.
    
    Parameters:
    -----------
    corrector : The corrector object
    params   :  at lmfit Parameters object
                containing b_i and c_i terms
    """
    
    ntel = corrector.b.shape[0]
    bvec = np.zeros_like(corrector.b)
    cvec = np.zeros_like(corrector.c)
    for i in range(ntel):
        bvec[i] = params["b%d"%(i)]
        cvec[i] = params["c%d"%(i)]
    return bvec,cvec


def get_depth(combiner, Is,):
    """
    Computes a "null depth" analogue (dark/bright)
    to be minimized by the tuning method.
    The masks in the combiner definition
    are used to determine the role of the different outputs.
    
    Parameters:
    -----------
    combiner : A combiner object.
    Is       : An array of intensities
    
    Note: This definition might need some adjustments for
    use in kernel-nullers?
    """
    bright = Is[:,combiner.bright].sum(axis=1)
    dark = Is[:,combiner.dark].sum(axis=1)
    res = (dark/bright)
    return res

def get_Is(params, combiner, corrector, lambs):
    """
    Returns intensities at the combiners' outputs
    taking into account the corrections provided in
    params.
    
    **dcomp is computed automatically be default.**
    
    Parameters:
    -----------
    params   : either
                - A Parameters object from the optimization
                - A tuple of vectors bvec, cvec
    combiner : A combiner object.
    corrector : The corrector object
    lambs    : The wavelengths considered.
    """
    
    if isinstance(params, Parameters):
        bvec, cvec = extract_corrector_params(corrector, params)
    else:
        bvec, cvec = params
    phasor = corrector.get_phasor_from_params(lambs, 
                                             a=None,
                                             b=bvec,
                                             c=cvec)
    #res = combiner.Mcn.dot(phasor)
    res = np.einsum("ikj,ij->ik", combiner.Mcn, phasor)
    Is = np.abs(res)**2
    return Is

def get_contrast_res(params, combiner, corrector, lambs):
    """
    Macro that gets the a residual from parameters 
    for minimizing method.
    """
    Is = get_Is(params, combiner, corrector, lambs)
    res = get_depth(combiner, Is)
    return res



class corrector(object):
    def __init__(self, config, lambs):
        """
        A module that provides beam adjustments
        for the input. It contains amplitude *a*, geometric
        piston *b* and ZnSe piston substitution *c*. Note that
        the ZnSe length replaces some air length.
        
        Parameters:
        -----------
        config:     A parsed config file
        lambs :     The wavelength channels to consider [m]
                    (At the __init__ stage, it is only used for
                    the computation of a mean refractive index for
                    the dispersive material)
        """
        self.config = config
        nznse_file = np.loadtxt(znse_file,delimiter=";")
        self.nznse = interp.interp1d(nznse_file[:,0]*1e-6, nznse_file[:,1],
                                     kind="linear", bounds_error=False )
        diams = self.config.getarray("configuration", "diam")
        n_tel = diams.shape[0]
        # An amplitude factor
        self.a = np.ones(n_tel)
        self.b = np.zeros(n_tel)
        self.c  = np.zeros(n_tel)
        self.nmean= np.mean(self.nznse(lambs))
        self.dcomp = -(self.nmean-1)*self.c
        
    
    def get_phasor(self, lambs):
        """
        Returns the complex phasor corresponding
        to the current a, b, c, and dcomp phasors.
        
        Parameters:
        -----------
        lambs :     The wavelength channels to consider [m]
        """
        ns = self.nznse(lambs)
        alpha = self.a[None,:]*np.exp(-1j*2*np.pi/lambs[:,None]*(self.b[None,:]+self.dcomp[None,:] +self.c[None,:]*(ns[:,None]-1)))
        return alpha
    def get_phasor_s(self, lamb):
        """
        Deprecated
        """
        ns = self.nznse(lambs)
        alpha = self.a*np.exp(-1j*2*np.pi/lamb*(self.b + self.dcomp +self.c*(ns-1)))
        return alpha
    
    def get_phasor_from_params(self, lambs, a=None,
                               b=None, c=None,
                               dcomp=None):
        """
        Similar to get_phasor() but allows to provide the
        parameters as arguments (slower).
        
        Returns the complex phasor corresponding
        to the current a, b, c, and dcomp phasors.
        
        Parameters:
        -----------
        lambs :     The wavelength channels to consider [m]
        a     :     Vector of the amplitude term
        b     :     Vetor of the geometric piston term [m]
        c     :     Vetor of the dispersive piston term [m]
        """
        ns = self.nznse(lambs)
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        if c is None:
            c = self.c
        if dcomp is None:
            dcomp = -(self.nmean-1)*c
        alpha = a[None,:]*np.exp(-1j*2*np.pi/lambs[:,None]*(b[None,:] + dcomp[None,:] +c[None,:]*(ns[:,None]-1)))
        return alpha
    
    def theoretical_phase(self,lambs, proj_opds, wet_atmo):
        """
        Computes the theoretical chromatic phase effect of the
        array geometry projected on axis based on the wet atmosphere
        model.
        
        Parameters:
        -----------
        lambs :     The wavelength channels to consider [m]
        proj_opds  : The projected piston obtained by projection
                    (Get from simulator.obs.get_projected_geometric_pistons)
        wet_atmo   : The wet atmosphere model (see n_air.wet_atmo object)
        """
        nair = n_air.n_air(lambs)
        phase = 2*np.pi/lambs*nair*proj_opds
        return phase
        
    def solve_air(self, lambs, wet_atmo):
        """
        Computes a least squares compensation model (see
        Koresko et al. 2003 DOI: 10.1117/12.458032)
        
        Parameters:
        -----------
        lambs :     The wavelength channels to consider [m]
        wet_atmo   : The wet atmosphere model (see n_air.wet_atmo object)
        """
        nair = n_air.n_air(lambs)
        ns = np.array([nair, self.nznse(lambs)]).T
        A = 2*np.pi/lambs[:,None] * ns
        
        self.S = np.linalg.inv(A.T.dot(A)).dot(A.T)
        return self.S
    
    def tune_static(self, lambs, combiner, apply=True,
                    freeze_params=["b0", "c0", "b2", "c2"]):
        """
        Optimize the compensator to correct chromatism in the 
        model of the combiner. Returns a lmfit solution object.
        If "apply" is set to True, a, b, c, and dcomp are also
        set to the best fit value.
        
        Parameters:
        -----------
        lambs :     The wavelength channels to consider [m]
        combiner :  A combiner object (chromatic)  
        apply    :  Boolean deciding whether to set the local
                    parameters to best fit value (default: True)
        freeze_params : The name of parameters to be freezed.
                    Should be used to account for the larger than
                    necessary number of degrees of freedom.
                    
        Note: For obtaining a more practical direct results, some
        more complicated balancing guidelines should be followed.
        """
        params = Parameters()
        for i in range(self.b.shape[0]):
            params.add("b%d"%(i),value=0., min=-1.0e-3, max=1.0e-3, vary=True)
            params.add("c%d"%(i),value=0., min=-1.0e-3, max=1.0e-3, vary=True)
        params["b0"].vary = False
        params["c0"].vary = False
        params["c2"].vary = False
        params["b2"].vary = False
        sol = minimize(get_contrast_res, params,
               args=(combiner, self, lambs),
               method="leastsq")
        
        self.sol = sol
        if apply:
            bvec, cvec = extract_corrector_params(self, sol.params)
            self.b = bvec
            self.c = cvec
            self.dcomp = -(self.nmean-1)*self.c
        return sol
    def plot_tuning(self,lambs,  npoints = 20):
        from kernuller.diagrams import plot_chromatic_matrix as cmpc
        import matplotlib.pyplot as plt
        
        pltlambrange = np.linspace(np.min(lambs),
                                   np.max(lambs),
                                   20)
        init_phasor = cor.get_phasor(pltlambrange)
        
        fig = cmpc(asim.combiner.M,asim.combiner.lamb, pltlambrange,
        plotout=cor.get_phasor_from_params(pltlambrange, b=cor.b, c=cor.c), minfrac=0.9)
        
        