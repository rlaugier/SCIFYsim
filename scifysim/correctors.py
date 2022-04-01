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

# Brute force test parameter to 
thetas = np.linspace(-np.pi, np.pi, 10000)
comphasor = np.ones(4)[None,:]*np.exp(1j*thetas[:,None])

# Utility functions for optimization
##################################################

from lmfit import Parameters, minimize
def extract_corrector_params(corrector, params):
    """
    Utility function to reconstruct the *b* and *c*
    vectors from the lmfit Parameters object.
    
    **Parameters:**
    
    * corrector : The corrector object
    * params   :  at lmfit Parameters object
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
    
    **Parameters:**
    
    * combiner : A combiner object.
    * Is       : An array of intensities
    
    .. admonition: Note:
    
        This definition might need some adjustments for
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
    
    **Parameters:**
    
    * params   : either
    
                - A Parameters object from the optimization
                - A tuple of vectors bvec, cvec
                
    * combiner : A combiner object.
    * corrector : The corrector object
    * lambs    : The wavelengths considered.
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


def get_es(params, combiner, corrector, lambs):
    """
    Returns the enantiomorph excursion taking into account the corrections provided in
    params.
    
    **Currently works only for double-bracewell 3-4 architecures**
        
    **dcomp is computed automatically by default.**
    
    **Parameters:**
    
    - params   : either
            * A Parameters object from the optimization
            * A tuple of vectors bvec, cvec
    - combiner : A combiner object.
    - corrector : The corrector object
    - lambs    : The wavelengths considered.
    """
    
    if isinstance(params, Parameters):
        bvec, cvec = extract_corrector_params(corrector, params)
    else:
        bvec, cvec = params
    phasor = corrector.get_phasor_from_params(lambs, 
                                             a=None,
                                             b=bvec,
                                             c=cvec)
    thetas = np.linspace(-np.pi, np.pi, 10000)
    comphasor = np.ones(4)[None,:]*np.exp(1j*thetas[:,None])
    amatcomp = np.einsum("ijk, ik -> ijk", combiner.Mcn, phasor)
    allcor = np.einsum("ik, mk -> mik", amatcomp[:,3,:], comphasor) - np.conjugate(amatcomp[:, 4,:])[None,:,:]
    excursion = np.min(np.linalg.norm(allcor, axis=2), axis=0)
    
    return excursion
def get_shape_res(params, combiner, corrector, lambs):
    """
    Macro that gets the a residual from parameters 
    for minimizing method.
    """
    res = get_es(params, combiner, corrector, lambs)
    return res


class corrector(object):
    def __init__(self, config, lambs, file=None):
        """
        A module that provides beam adjustments
        for the input. It contains amplitude *a*, geometric
        piston *b* and ZnSe piston substitution *c*. Note that
        the ZnSe length replaces some air length.
        
        **Parameters:**
        
        * config:     A parsed config file
        * lambs :     The wavelength channels to consider [m]
          (At the __init__ stage, it is only used for
          the computation of a mean refractive index for
          the dispersive material)
        * file A file containing the plate index
                    
        **Internal parameters:**
        
        * a     :     Vector of the amplitude term
        * b     :     Vetor of the geometric piston term [m]
        * c     :     Vetor of the dispersive piston term [m]
        """
        self.config = config
        if file is None:
            nplate_file = np.loadtxt(znse_file,delimiter=";")
        else: 
            nplate_file = file
        self.nplate = interp.interp1d(nplate_file[:,0]*1e-6, nplate_file[:,1],
                                     kind="linear", bounds_error=False )
        diams = self.config.getarray("configuration", "diam")
        n_tel = diams.shape[0]
        # An amplitude factor
        self.a = np.ones(n_tel)
        self.b = np.zeros(n_tel)
        self.c  = np.zeros(n_tel)
        self.nmean= np.mean(self.nplate(lambs))
        self.dcomp = -(self.nmean-1)*self.c
        
        self.prediction_model = n_air.wet_atmo(config)
        
    
    def get_phasor(self, lambs):
        """
        Returns the complex phasor corresponding
        to the current a, b, c, and dcomp phasors.
        
        **Parameters:**
        
        * lambs :     The wavelength channels to consider [m]
        
        **Returns:** alpha
        """
        ns = self.nplate(lambs)
        alpha = self.a[None,:]*np.exp(-1j*2*np.pi/lambs[:,None]*(self.b[None,:]+self.dcomp[None,:] +self.c[None,:]*(ns[:,None]-1)))
        return alpha
    def get_phasor_s(self, lamb):
        """
        Deprecated
        """
        ns = self.nplate(lambs)
        alpha = self.a*np.exp(-1j*2*np.pi/lamb*(self.b + self.dcomp +self.c*(ns-1)))
        return alpha
    def get_raw_phase_correction(self, lambs, b=0,c=0, dcomp=0, model=None):
        """
        Returns the raw (non-wrapped) phase produced by an optical path
        of b[m] in air and c[m] in plate material.
        
        **Parameters**
        
        * lambs :     The wavelength channels to consider [m]
        * a     :     Vector of the amplitude term
        * b     :     Vetor of the geometric piston term [m]
        * c     :     Vetor of the dispersive piston term [m]
        * dcomp :     A length of air to compensate for the plate
        
        
        """
        if model is None:
            model = self.prediction_model
        nair = model.get_Nair(lambs, add=1)
        nplate = self.nplate(lambs)
        return 2*np.pi/lambs*(nair*b + nplate*c)
    def get_dcomp(self, c):
        """
        Returns the theoertical value of dcomp for a given value of compensator
        plate, to correct for the pure piston term introduced.
        """
        dcomp = -(self.nmean-1)*c
        return dcomp
        
    def get_phasor_from_params(self, lambs, a=None,
                               b=None, c=None,
                               dcomp=None):
        """
        Similar to get_phasor() but allows to provide the
        parameters as arguments (slower).
        
        Returns the complex phasor corresponding
        to the current a, b, c, and dcomp phasors.
        
        **Parameters:**
        
        * lambs :     The wavelength channels to consider [m]
        * a     :     Vector of the amplitude term
        * b     :     Vetor of the geometric piston term [m]
        * c     :     Vetor of the dispersive piston term [m]
        """
        ns = self.nplate(lambs)
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        if c is None:
            c = self.c
        if dcomp is None:
            dcomp = self.get_dcomp(c)
        alpha = a[None,:]*np.exp(-1j*2*np.pi/lambs[:,None]*(b[None,:] + dcomp[None,:] +c[None,:]*(ns[:,None]-1)))
        return alpha
    
    def theoretical_phase(self,lambs, proj_opds, model=None, add=1):
        """
        Computes the theoretical chromatic phase effect of the
        array geometry projected on axis based on the wet atmosphere
        model.
        
        **Parameters:**
        
        * lambs :     The wavelength channels to consider [m]
        * proj_opds  : The projected piston obtained by projection
          (Get from simulator.obs.get_projected_geometric_pistons)
        * model      : A model for humid air (see n_air.wet_atmo object).
          If None, defaults to self.model, created upon init.
        * add        : returns n-1+add (add=0 gives the relative
          optical path compared to vacuum)
        
        **Returns:** phase
        """
        nair = model.get_Nair(lambs, add=add)
        phase = 2*np.pi/lambs*nair*proj_opds
        return phase.T
        
    def solve_air(self, lambs, model):
        """
        Computes a least squares compensation model (see
        **Koresko et al. 2003 DOI: 10.1117/12.458032**)
        
        **Parameters:**
        
        * lambs :     The wavelength channels to consider [m]
        * model   : The wet atmosphere model (see n_air.wet_atmo object)
        
        **Returns:** :math:`\Big( \mathbf{A}^T\mathbf{A}\mathbf{A}^T \Big)^{-1}`
        """
        nair = model.get_Nair(lambs)
        ns = np.array([nair, self.nplate(lambs)]).T
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
        
        **Parameters:**
        
        * lambs :     The wavelength channels to consider [m]
        * combiner :  A combiner object (chromatic)  
        * apply    :  Boolean deciding whether to set the local
          parameters to best fit value (default: True)
        * freeze_params : The name of parameters to be freezed.
          Should be used to account for the larger than
          necessary number of degrees of freedom.
                    
        .. admonition:: Note:
            
            For obtaining a more practical direct results, some
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
    
    def tune_static_shape(self, lambs, combiner, apply=True,
                    sync_params=[("b3", "b2", 0.),
                                ("c3", "c2", 0.)],
                    freeze_params=["b0", "c0", "b1", "c1"]):
        """
        Optimize the compensator to correct chromatism in the 
        model of the combiner to obtain enantomporph combinations
        at the outputs. Returns a lmfit solution object.
        If ``apply`` is set to True, a, b, c, and dcomp are also
        set to the best fit value.
        
        **Currently only works for double Bracewell 3-4 architectures.**
        
        **Parameters:**
        
        * lambs :     The wavelength channels to consider [m]
        * combiner :  A combiner object (chromatic)  
        * apply    :  Boolean deciding whether to set the local
          parameters to best fit value (default: True)
          
        .. admonition:: Note:
        
            For obtaining a more practical direct results, some
            more complicated balancing guidelines should be followed.
        
        **Example:**
        
        .. code-block::
            
            sol = asim.corrector.tune_static_shape(asim.lambda_science_range,
                             asim.combiner,
                             sync_params=[("b3", "b2", asim.corrector.b[3] - asim.corrector.b[2]),
                                         ("c3", "c2", asim.corrector.c[3] - asim.corrector.c[2])],
                             apply=True)
        
        """
        params = Parameters()
        print("inside_tuning", self.b, self.c)
        for i in range(self.b.shape[0]):
            params.add("b%d"%(i),value=self.b[i], min=-1.0e-3, max=1.0e-3, vary=True)
            params.add("c%d"%(i),value=self.c[i], min=-1.0e-3, max=1.0e-3, vary=True)
        
        # Should do this in a loop for sync_params
        for tosync in sync_params:
            params[tosync[0]].set(expr=tosync[1]+f"+ {tosync[2]}")
        # If we have 
        #b23 = self.b[3]-self.b[2]
        #c23 = self.c[3]-self.c[2]
        #params["b3"].set(expr=f"b2 + {b23}")
        #params["c3"].set(expr=f"c2 + {c23}")
        for aparam in freeze_params:
            params[aparam].set(vary=False)
        
        #display.display(params)
        sol = minimize(get_shape_res, params,
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
        
        