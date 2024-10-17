import numpy as np
#from sympy.functions.elementary.piecewise import Piecewise
#from kernuller import mas2rad, rad2mas
#from . import utilities
from . import n_air
#from astropy import constants
#from astropy import units
import scipy.interpolate as interp
from pathlib import Path
import logging
logit = logging.getLogger(__name__)
from copy import deepcopy

# import pdb

# from pdb import set_trace


parent = Path(__file__).parent.absolute()
#znse_file = parent/"data/znse_index.csv"
znse_file = parent/"data/znse_Connolly.csv"
caf2_file = parent/"data/caf2_index.csv"



# Brute force test parameter to 
thetas = np.linspace(-np.pi, np.pi, 10000)
comphasor = np.ones(4)[None,:]*np.exp(1j*thetas[:,None])

def get_max_differential(array):
    """
    Gets the maximum differential between beams:
    
    * returns : max(array) - min(array)
    """
    return np.max(array) - np.min(array)

# Utility functions for optimization
##################################################

class material_sellmeier(object):
    def __init__(self, bs, cs=None):
        """
        * bs  : the B terms of the Sellmeier equation
        * cs  : the C terms of the Sellmeier equation [µm^2]

        if cs is not provided, then we expect coefficients given from `bs`
        as (B1, C1, B2, C2, ...)
        
        """
        if cs is None:
            self.bs, self.cs = bs2bcs(bs)
        else:
            self.bs = bs
            self.cs = cs
            
    def get_n(self, lamb, add=0):
        lamb_microns = lamb*1e6
        terms = np.array([ab*lamb_microns**2 / (lamb_microns**2 - ac) \
                        for (ab, ac) in zip(self.bs, self.cs)])
        nr = -1 + np.sqrt(1 + np.sum(terms, axis=0))
        return nr + add

def bs2bcs(bs):
    """
    Transforms a flat list of sellmeier coefficients (B1, C1, B2, C2, ...)
    into two vectors fo coefficients B and C.
    """
    order = bs.shape[0]//2
    b2 = bs.reshape((order, 2))
    return b2[:,0], b2[:,1]    

# Adding builtin material for the linb
bs_e = np.array([2.9804,
              0.02047,
              0.5981,
              0.0666,
              8.9543,
              416.08])
bs_o = np.array([2.6734,
                0.01764,
                1.2290,
                0.05914,
                12.614,
                474.6])

linb_o = material_sellmeier(bs_o)
linb_e = material_sellmeier(bs_e)


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
                  does not support e: e is taken from corrector
                - A tuple of vectors bvec, cvec
                
    * combiner : A combiner object.
    * corrector : The corrector object
    * lambs    : The wavelengths considered.
    """
    
    if isinstance(params, Parameters):
        bvec, cvec = extract_corrector_params(corrector, params)
        evec = corrector.e
    else:
        bvec = params[:,0]
        cvec = params[:,1]
        evec = params[:,2]
        # bvec, cvec = params
    phasor = corrector.get_phasor_from_params(lambs, 
                                             a=None,
                                             b=bvec,
                                             c=cvec,
                                             e=evec)
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

class offband_ft(object):
    def __init__(self, wl_ft, wl_science, wa_true, wa_model=None, corrector=None):
        """
        Builds a fringe tracker object that handles the problems of dispersion
        between the band in use and the band of interest. It also addresses the problems
        inside the band of interest.
        
        **Arguments:**
        
        * wl_ft : The wavelength at which the fringe tracking is done.
          OPD will be set to produce 0 phase at the man FT wavelength
        * wl_science : The wavelength range for which the correction is optimized
        * wa_true : True wet air model
        * wa_model : Modeled wet air model (as measured at the telescopes)
        * corrector : An dispersion corrector object to compute corresponding
          glass and air thickness
          
        **Note:**
        
        Contrary to the corrector object, the corrections are stored in a single 2D
        array, with the b on the row 0 and c on row 1
        """
        self.wl_ft = wl_ft
        self.wl_science = wl_science
        self.wa_true = wa_true
        self.wa_model = wa_model
        self.corrector = corrector
        self.corrector_model = deepcopy(corrector)
        self.corrector_model.ncomp = self.wa_model.get_Nair
        # corrector_model should be used mostly to compute b_ft or make predictions
        # Estimates of the real phases should always use the true corrector
        
        self.refresh_corrector_models()
        
        #self.corrector = corrector
        self.simple_piston_ft = None
        self.phase_correction_ft = None
        self.phase_correction_ft_science = None

    def refresh_corrector_models(self):
        self.S_model_science = self.corrector_model.solve_air_corrector(self.wl_science)
        self.S_model_ft = self.corrector_model.solve_air(self.wl_ft, self.wa_true)# Solving based for the closed_loop FT
        self.S_model_ft_feedforward = self.corrector_model.solve_air_corrector(self.wl_ft)# Solving based on the FT measurements
        self.S_truth_science = self.corrector.solve_air_corrector(self.wl_science)
        self.S_truth_ft = self.corrector.solve_air(self.wl_ft, self.wa_true)# Solving based for the closed_loop FT
        self.S_truth_ft_feedforward = self.corrector.solve_air_corrector(self.wl_ft)# Solving based on the FT measurements
        self.S_gd_model_ft = self.get_S_GD_star(band=self.wl_ft, model=self.wa_model)
        self.S_gd_model_science = self.get_S_GD_star(band=self.wl_science, model=self.wa_model)
        self.S_gd_ft = self.get_S_GD_star(band=self.wl_ft, model=self.wa_true)
        self.S_gd_science = self.get_S_GD_star(band=self.wl_science, model=self.wa_true)
        
        
    def get_ft_correction_on_science(self, pistons, band=None):
        """
        **Arguments**:
        
        * pistons : The value of optical path length missing at 
          the reference plane [m]
        
        **Output**: The phases for the science wavelengths for a closed loop correction
        by the fringe tracker [rad]
        """
        # The true piston that gives the best mean phase in the FT band
        # This should remain true in white fringe phase tracking mode
        # because of closed loop
        self.simple_piston_ft = np.mean(self.corrector.theoretical_piston(self.wl_ft,
                                                                         pistons,
                                                                         model=self.wa_true,
                                                                         add=0), axis=0)
        # The corrected phase:
        self.phase_correction_ft = self.corrector.get_raw_phase_correction(self.wl_ft[:,None],
                                                                            b=self.simple_piston_ft,
                                                                            c=0)
        self.phase_correction_ft_science = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
                                                                                    b=self.simple_piston_ft,
                                                                                    c=0)
        if band is not None:
            return self.corrector.get_raw_phase_correction(band[:,None],
                                                            b=self.simple_piston_ft, c=0)
        else:
            return self.phase_correction_ft_science
    

    
    def update_NCP_corrections(self,pistons, band=None, mode="phase"):
        """
        Called by `get_phase_on_band` and `get_phase_on_science_values` and therefore
        called each time the `director.point` is called.
        
        **Arguments**:
        
        * pistons : The value of optical path length missing at 
          the reference plane [m]

        **Refreshes**:

        * `self.true_phase_on_science`
        self.model_phase_on_science
        self.correction_ft_to_science
        self.phase_seen_by_ft 
        self.b_ft
        self.correction_feedforward
        self.b_science_ideal
        
        * Computing the feedforward correction based on the FT phase:
          - `self.b_ft` : The atmospheric corrector for the FT band
          - `self.correction_feedforward` : The atmospheric corrector for the FT band
        
        * Computing the ideal correction for perfect knowledge of atmosphere
          (If we could measure the actual phase and close a loop)
          - `self.b_science_ideal` : The atmospherci corrector for the science band
          - `self.correction_closed_loop`
        
        * Computing a biased estimation based on a full model
          - `self.b_science_model` : The atmospherci corrector for the science band
          - `self.correction_blind`

        **New implementation**:
        
        In the idea that the phase of phase at entrance of the NOTT chip phi_tot:
        ```
        phi_tot = phi_v + phi_DL + phi_nott                            
                    |        |        |-> Phase from the NOTT LDC
                    |        |----------> Phase from the DL correction
                    |-------------------> Phase from the vacuum piston imbalance
        phi_A = phi_v + phi_DL
          |->  the Asgard phase
        phi_nott = phi_AN + phi_ZN + phi_CN + phi_LN
                    |        |        |        |-> phase from LiNb plates
                    |        |        |----------> phase from CO2 cells
                    |        |-------------------> phase from ZnSe plates
                    |----------------------------> phase from the air DL and TT

        ```

        """
        # In the idea that the phase of phase at entrance of the NOTT chip phi_tot
        # The vacuum phase:
        self.phi_v = (2*np.pi/self.wl_science * pistons).T
        self.phi_v_ft = (2*np.pi/self.wl_ft * pistons).T
        if mode == "phase":
            # Computing the feedforward correction based on the FT phase:
            # This is using the true phase since it is closed loop
            self.b_ft = self.S_truth_ft.dot(self.phi_v_ft).T
            self.phi_DL = self.corrector.get_raw_phase_correction(self.wl_science[:,None], vector=self.b_ft)
            self.phi_DL_ft = self.corrector.get_raw_phase_correction(self.wl_ft[:,None], vector=self.b_ft)
            if band is not None:
                phi_DL_band = self.corrector.get_raw_phase_correction(band[:,None], vector=self.b_ft)
            # This is what we believe Heimdallr has accomplished
            self.b_ft_model = self.S_model_ft.dot(self.phi_v_ft).T
            self.phi_DL_model = self.corrector_model.get_raw_phase_correction(self.wl_science[:,None], vector=self.b_ft_model)
            self.phi_DL_ft_model = self.corrector_model.get_raw_phase_correction(self.wl_ft[:,None], vector=self.b_ft_model)
            if band is not None:
                phi_DL_model_band = self.corrector_model.get_raw_phase_correction(band[:,None], vector=self.b_ft_model)
        elif mode == "group":
            # raise NotImplementedError("Group delay FT tracking not ready yet")
            logit.warning("Early implementation of GD tracking")
            n_air = self.wa_true.get_Nair(self.wl_science, add=1)
            n_air_ft = self.wa_true.get_Nair(self.wl_ft, add=1)
            self.phi_DL = (- 2*np.pi/self.wl_science * self.S_gd_science * n_air * pistons ).T
            self.phi_DL_ft = (- 2*np.pi/self.wl_ft * self.S_gd_ft * n_air_ft * pistons ).T
            if band is not None:
                n_air_band = self.wa_true.get_Nair(band, add=1)
                phi_DL_band = (- 2*np.pi/band * self.S_gd_science * n_air_band * pistons).T
            n_air_model = self.wa_model.get_Nair(self.wl_science, add=1)
            n_air_ft_model = self.wa_model.get_Nair(self.wl_ft, add=1)
            self.phi_DL_model = (- 2*np.pi/self.wl_science * self.S_gd_model_science * n_air_model * pistons ).T
            self.phi_DL_ft_model = (- 2*np.pi/self.wl_ft * self.S_gd_model_ft * n_air_ft_model * pistons ).T
            if band is not None:
                n_air_model_band = self.wa_model.get_Nair(band, add=1)
                phi_DL_model_band = (- 2*np.pi/band * self.S_gd_model_science * n_air_model_band * pistons ).T
        self.phi_asgard_model = self.phi_v - self.phi_DL_model
        self.phi_asgard_true = self.phi_v - self.phi_DL
        if band is not None:
            phi_v_band = (2*np.pi/band * pistons).T
            phi_asgard_true_band = phi_v_band - phi_DL_band
            phi_asgard_model_band = phi_v_band - phi_DL_model_band

        # Now to compute the correction
        # Temp correction
        ###########################
        # 

        # Computing the ideal correction (If we could measure the actual phase and close a loop)
        self.b_science_ideal = self.S_model_science.dot(self.phi_asgard_true).T
        #self.correction_closed_loop = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
        #                                                      b=self.b_science_ideal[0,:],
        #                                                      c=self.b_science_ideal[1,:])
        self.correction_ideal = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
                                                                             vector=self.b_science_ideal)
        
        # Computing a biased estimation based on a full model
        self.b_science_model = self.S_model_science.dot(self.phi_asgard_model).T
        #self.correction_blind = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
        #                                                      b=self.b_science_model[0,:],
        #                                                      c=self.b_science_model[1,:])
        self.correction_blind = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
                                                                       vector=self.b_science_model)

        # self.phi_tot_ft = self.phi_v_ft + self.phi_DL_ft
        # self.phi_tot    = self.phi_v + self.phi_DL
        
        self.phi_nott_ideal = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
                                                vector=self.b_science_ideal)
        self.phi_nott_model = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
                                                vector=self.b_science_model)
        
        if band is not None:
            phi_nott_ideal_band = self.corrector.get_raw_phase_correction(band[:,None],
                                                    vector=self.b_science_ideal)
            phi_nott_model_band = self.corrector.get_raw_phase_correction(band[:,None],
                                                    vector=self.b_science_model)
            return phi_asgard_true_band, phi_asgard_model_band, phi_nott_ideal_band, phi_nott_model_band 



    
        # # Computing the errors in the L' band
        # # The total phase from the piston on the band
        # self.true_phase_on_science = self.corrector.theoretical_phase(self.wl_science, pistons, model=self.wa_true, add=0)
        # if band is not None:
        #     total_phase_on_band = self.corrector.theoretical_phase(band, pistons,
        #                                                           model=self.wa_true, add=0)
        # self.model_phase_on_science = self.corrector.theoretical_phase(self.wl_science, pistons,
        #                                                                model=self.wa_model, add=0)
        # # The phase from piston, only from FT to science (assusming a closed loop on the FT)
        # self.correction_ft_to_science = self.get_ft_correction_on_science(pistons)
        # if band is not None:
        #     correction_to_phase_ft = self.get_ft_correction_on_science(pistons, band=band)
        # # becomes self.phi_v
        # self.phase_seen_by_ft = self.corrector.theoretical_phase(self.wl_ft, pistons, model=self.wa_true, add=0) - self.phase_correction_ft
        
        # # Computing the feedforward correction based on the FT phase:
        # self.b_ft = self.S_model_ft.dot(self.phase_seen_by_ft).T
        # #self.correction_feedforward = asim.corrector.get_raw_phase_correction(self.wl_science[:,None],
        # #                                                      b=self.b_ft[0,:],
        # #                                                      c=self.b_ft[1,:])
        # self.correction_feedforward = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
        #                                                                      vector=self.b_ft)
        
        # # Computing the ideal correction (If we could measure the actual phase and close a loop)
        # self.b_science_ideal = self.S_model_science.dot(self.true_phase_on_science - self.correction_ft_to_science).T
        # #self.correction_closed_loop = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
        # #                                                      b=self.b_science_ideal[0,:],
        # #                                                      c=self.b_science_ideal[1,:])
        # self.correction_closed_loop = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
        #                                                                      vector=self.b_science_ideal)
        
        # # Computing a biased estimation based on a full model
        # self.b_science_model = self.S_model_science.dot(self.model_phase_on_science - self.correction_ft_to_science).T
        # #self.correction_blind = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
        # #                                                      b=self.b_science_model[0,:],
        # #                                                      c=self.b_science_model[1,:])
        # self.correction_blind = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
        #                                                                vector=self.b_science_model)
        # if band is not None:
        #     return total_phase_on_band, correction_to_phase_ft    

    def update_NCP_corrections_old(self,pistons, band=None):
        """
        Called by `get_phase_on_band` and `get_phase_on_science_values` and therefore
        called each time the `director.point` is called.
        
        **Arguments**:
        
        * pistons : The value of optical path length missing at 
          the reference plane [m]

        **Refreshes**:

        * `self.true_phase_on_science`
        self.model_phase_on_science
        self.correction_ft_to_science
        self.phase_seen_by_ft 
        self.b_ft
        self.correction_feedforward
        self.b_science_ideal
        
        * Computing the feedforward correction based on the FT phase:
          - `self.b_ft` : The atmospheric corrector for the FT band
          - `self.correction_feedforward` : The atmospheric corrector for the FT band
        
        * Computing the ideal correction for perfect knowledge of atmosphere
          (If we could measure the actual phase and close a loop)
          - `self.b_science_ideal` : The atmospherci corrector for the science band
          - `self.correction_closed_loop`
        
        * Computing a biased estimation based on a full model
          - `self.b_science_model` : The atmospherci corrector for the science band
          - `self.correction_blind`

        """
        # Computing the errors in the L' band
        # The total phase from the piston on the band
        self.true_phase_on_science = self.corrector.theoretical_phase(self.wl_science, pistons,
                                                                        model=self.wa_true, add=0)
        if band is not None:
            total_phase_on_band = self.corrector.theoretical_phase(band, pistons,
                                                                  model=self.wa_true, add=0)
        self.model_phase_on_science = self.corrector.theoretical_phase(self.wl_science, pistons,
                                                                       model=self.wa_model, add=0)
        # The phase from piston, only from FT to science (assusming a closed loop on the FT)
        self.correction_ft_to_science = self.get_ft_correction_on_science(pistons)
        if band is not None:
            correction_to_phase_ft = self.get_ft_correction_on_science(pistons, band=band)
        self.phase_seen_by_ft = self.corrector.theoretical_phase(self.wl_ft, pistons,
                                                model=self.wa_true, add=0) - self.phase_correction_ft
        
        # Computing the feedforward correction based on the FT phase:
        self.b_ft = self.S_model_ft.dot(self.phase_seen_by_ft).T
        #self.correction_feedforward = asim.corrector.get_raw_phase_correction(self.wl_science[:,None],
        #                                                      b=self.b_ft[0,:],
        #                                                      c=self.b_ft[1,:])
        self.correction_feedforward = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
                                                                             vector=self.b_ft)
        
        # Computing the ideal correction (If we could measure the actual phase and close a loop)
        self.b_science_ideal = self.S_model_science.dot(self.true_phase_on_science - self.correction_ft_to_science).T
        #self.correction_closed_loop = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
        #                                                      b=self.b_science_ideal[0,:],
        #                                                      c=self.b_science_ideal[1,:])
        self.correction_closed_loop = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
                                                                             vector=self.b_science_ideal)
        
        # Computing a biased estimation based on a full model
        self.b_science_model = self.S_model_science.dot(self.model_phase_on_science - self.correction_ft_to_science).T
        #self.correction_blind = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
        #                                                      b=self.b_science_model[0,:],
        #                                                      c=self.b_science_model[1,:])
        self.correction_blind = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
                                                                       vector=self.b_science_model)
        if band is not None:
            return total_phase_on_band, correction_to_phase_ft

    def get_modeled_law(self, max_baseline=133, model=None):
        """
        Evaluate a law for glass and air compensation
        **Arguments:**
        
        * max_baseline : the maximum length tho considerj
        * model : A humid air model for which to draw the plot
        """
        if model is None:
            model = self.wa_model
        path_lengths = np.linspace(-max_baseline, max_baseline, 200)[:,None]
        S_model = self.corrector.solve_air_corrector(self.wl_science)
        
        model_phase_on_science = self.corrector.theoretical_phase(self.wl_science, path_lengths,
                                                                  model=model, add=0)
        
        correction_ft_to_science = self.get_ft_correction_on_science(path_lengths)
        phases = model_phase_on_science - correction_ft_to_science
        b_model = S_model.dot(phases)
        #correction_closed_loop = self.corrector.get_raw_phase_correction(self.wl_science[:,None],
        #                                                      b=self.b_science_ideal[0,:],
        #                                                      c=self.b_science_ideal[1,:])
        
        air_compensation = -(self.corrector.nmean-1)*b_model[1,:]
        air_range = np.max(b_model[0,:] + air_compensation)
        
        #self.model_phase_on_science - self.correction_ft_to_science
        glass_range = np.max(b_model[1,:])

        import matplotlib.pyplot as plt
        
        plt.figure(dpi=200)
        plt.subplot(211)
        plt.plot(path_lengths, b_model[0,:], label="Air length, uncompensated")
        plt.plot(path_lengths, b_model[0,:] + air_compensation, label="Air length, uncompensated")
        plt.legend(fontsize="x-small")
        plt.xlabel("Path length [m]")
        plt.ylabel("Air compensation [m]")
        plt.title(f"Maximum range for {model.name} over {max_baseline:.1f} m\n Air: {air_range*1000:.2f} mm")
        
        plt.subplot(212)
        plt.plot(path_lengths, b_model[1,:], label="Glass length, uncompensated")
        plt.legend(fontsize="x-small")
        plt.xlabel("Path length [m]")
        plt.ylabel("Glass compensation [m]")
        plt.title(f"Maximum glass range: {glass_range*1000:.2f} mm")
        plt.tight_layout()
        plt.show()
        
        
        
    def get_S_GD_star(self, band=None, model=None, resample=None):
        """
        S_GD_star is the transform matrix (flat in this case) that projects
        the phase of vacuum created phases into piston that minimize the
        group delay.

        This is inspired by Tango 1990. In the case of only air conpensation,
        equation 
        
        **Arguments**:
        * band  : if None: will save `self.sld_sxs`
        * pistons : 
        * model    : A specific model with which to compute the correction
        * resample: None (default) to keep the sampling of the band,
          or int to resample band with that number of samples.


        """
        if band is None:
            band = self.wl_ft
        if resample is None:
            wls = band
        else:
            wls = np.linspace(band[0], band[-1], resample)
        if model is None:
            model = self.wa_true
        # sig is the spectroscopic wavenumber 1/lambda
        sig = 1/wls
        nair = model.get_Nair_wn(sig, add=1)
        sigX = sig * nair
        s_dsigX_dsig = np.mean(np.gradient(sigX)/np.gradient(sig))
        s_gd = s_dsigX_dsig /nair**2
        if band is None:
            self.s_gd = s_gd
        return s_gd

    def get_phase_GD_tracking(self, pistons, band=None, model=None, resample=None):
        """
        **Arguments**:

        * band  : if None: will save `self.sld_sxs`
        * pistons : 
        * model    : A specific model with which to compute the correction
        * resample: None (default) to keep the sampling of the band,
          or int to resample band with that number of samples.
        """
        if band is None:
            band = self.wl_ft
        if model is None:
            model = self.wa_true
        s_gd = self.get_S_GD_star(band=band, model=model, resample=resample)

        n_air = model.get_Nair(band, add=1)
        
        phase_GD_target = 2*np.pi/band * pistons * (1- s_gd * n_air)
        if band is None:
            self.phase_GD_target = phase_GD_target
        return phase_GD_target
        
    
    def get_phase_on_band(self, band, pistons,
                                mode="ideal",
                                ft_mode="phase"):
        """Similar to ``get_phase_science_values`` but for an arbitrary band 
        for illustration purposes.
        
        **Arguments**:
        * band  : 
        * pistons : 
        * mode    : Type of correction to apply
            - `ideal` : Compensate with perfect knowledge of the atmospheric
              effects.
            - `blind` : Compensate using the internal atmosphere model
              which may deviate from ground truth.
            - `feedforward` : Assumes the dispersion is being measured at teh FT band
        """
        # `phi_asgard_model_band` normally used only to compute b_xxx
        phi_asgard_true_band,\
        phi_asgard_model_band,\
        phi_nott_ideal_band,\
        phi_nott_model_band = self.update_NCP_corrections(pistons, band=band, mode=ft_mode)

        #true_phase_on_band = self.corrector.theoretical_phase(band, pistons, model=self.wa_true, add=0)
        #model_phase_on_band = self.corrector.theoretical_phase(band, pistons, model=self.wa_model, add=0)
        if mode == "ideal":
            # Assumes a closed loop in the science band
            #correction_closed_loop = self.corrector.get_raw_phase_correction(band[:,None],
            #                                                  b=self.b_science_ideal[0,:],
            #                                                  c=self.b_science_ideal[1,:])
            correction_closed_loop = self.corrector.get_raw_phase_correction(band[:,None],
                                                                            vector=self.b_science_ideal)
            return phi_asgard_true_band - phi_nott_ideal_band - correction_closed_loop
            
        elif mode == "blind":
            # Assumes only atmosphere is good and setpoint of FT is solid
            #correction_blind = self.corrector.get_raw_phase_correction(band[:,None],
            #                                                  b=self.b_science_model[0,:],
            #                                                  c=self.b_science_model[1,:])
            correction_blind = self.corrector.get_raw_phase_correction(band[:,None],
                                                                      vector=self.b_science_model)
            return phi_asgard_true_band - phi_asgard_model_band - correction_blind
            # NB: phi_asgard_model_band is used only for computing b_model
        
        elif mode == "feedforward":
            # Assumes a measurement of dispersion in the FT band
            #correction_feedforward = asim.corrector.get_raw_phase_correction(band[:,None],
            #                                                  b=self.b_ft[0,:],
            #                                                  c=self.b_ft[1,:])
            logit.warning("feedfoward not implemented yet. returns model-model")
            correction_feedforward = self.corrector.get_raw_phase_correction(band[:,None],
                                                                            vector=self.b_ft)
            return phi_asgard_model_band - phi_nott_model_band - correction_feedforward
        
        
        
    def get_phase_science_values(self,pistons,
                                    mode="ideal",
                                    ft_mode="phase"):
        self.update_NCP_corrections(pistons, mode=ft_mode)
        if mode == "ideal":
            # Assumes a closed loop in the science band
            return self.phi_asgard_true - self.phi_nott_ideal
            
        elif mode == "blind":
            # Assumes only atmosphere is good and setpoint of FT is solid
            return self.phi_asgard_true - self.phi_nott_model
        
        elif mode == "feedforward":
            # Assumes a measurement of dispersion in the FT band
            logit.warning("feedfoward not implemented yet. returns model-model")
            return self.phi_asgard_model - self.phi_nott_model

        

def generic_vacuum(lambs, add=1.):
    """
    Add should always be 1.
    """
    return np.ones_like(lambs) + add
def no_material(lambs, add=0):
    """
    Add should always be 0.
    """
    return np.zeros_like(lambs) + add

class corrector(object):
    def __init__(self, config, lambs, file=None, order=3,
                model_comp=None, model_material2=None):
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
        * order  : The order to which do the interpolation of the
          tabulated refractive index of the material 1
        * model_comp : A wet_atmo object model for the material in
          which compensation is made (None -> Vacuum).
        * model_material2 : A wet_atmo object model for the second
          material for compensation is made (None -> no material).
          
                    
        **Internal parameters:**
        
        * a     :     Vector of the amplitude term
        * b     :     Vetor of the geometric piston term [m]
        * c     :     Vetor of the dispersive piston term [m]
        """
        self.config = config
        if file is None:
            nplate_file = np.loadtxt(znse_file, delimiter=";")
        else: 
            nplate_file = np.loadtxt(parent/"data"/file, delimiter=";")
        self.nplate = interp.interp1d(nplate_file[:,0]*1e-6, nplate_file[:,1],
                                     kind=order, bounds_error=False )
        if model_comp is None:
            print("Not spotted ncomp")
            self.ncomp = generic_vacuum
        else:
            self.wa_comp = model_comp
            # print("successfully spotted ncomp")
            self.ncomp = self.wa_comp.get_Nair
            
        if model_material2 is None:
            self.nmat2 = no_material
        else:
            self.nmat2 = model_material2.get_Nair
        self.nmean_mat2 = np.mean(self.nmat2(lambs))
        diams = self.config.getarray("configuration", "diam")
        self.n_tel = diams.shape[0]
        # An amplitude factor
        
        chip_corr = np.zeros((self.n_tel,4)) # (n_tel, n_glasses)
        atmo_corr = np.zeros((self.n_tel,4)) # (n_tel, n_glasses)
        a_corr = np.ones(self.n_tel) # (n_tel, n_glasses)
        
        self.refresh_adjustments(a=a_corr, chip_corr=chip_corr, atmo_corr=atmo_corr,
                                write=True)
        
        self.nmean = np.mean(self.nplate(lambs))

        
        self.prediction_model = n_air.wet_atmo(config)
        #pdb.set_trace()
        
    def refresh_adjustments(self, a=None, chip_corr=None, atmo_corr=None,
                            write=False):
        """
        Updates the values of self.a, self.b... based on the 
        values in self.chip_corr and atmo_corr. Except for a,
        arrays are of shape `(n_tel, n_glasses)` with n_glasses including
        air and air compensation.
        
        **Arguments:**
        * a        : Correction for amplitude
        * chip corr: Correction for the chip [m]
        * atmo_corr: Correction for the atmosphere [m]
        * write : If true, will write down the vectors received
        """
        if a is None:
            a = self.a
        elif write:
            self.a = a
            
        if chip_corr is None:
            chip_corr = self.chip_corr
        elif write:
            self.chip_corr = chip_corr
            
        if atmo_corr is None:
            atmo_corr = self.atmo_corr
        elif write:
            self.atmo_corr = atmo_corr
            
        self.b = chip_corr[:,0] + atmo_corr[:,0] # Air compensation
        self.c  = chip_corr[:,1] + atmo_corr[:,1] # Glass length
        #-(self.nmean-1)*self.c 
        self.dcomp = chip_corr[:,2] + atmo_corr[:,2] # A length of air as compensation for glass
        self.e = chip_corr[:,3] + atmo_corr[:,3] # Additional material (CO2 bellows)
        
    def get_phasor(self, lambs):
        """
        Returns the complex phasor corresponding
        to the current a, b, c, and dcomp phasors.
        
        **Parameters:**
        
        * lambs :     The wavelength channels to consider [m]
        
        **Returns:** alpha
        """
        # the air displacing solid plate:
        np1 = self.nplate(lambs) - self.ncomp(lambs, add=1.)
        # The air displacing second material:
        np2 = self.nmat2(lambs, add=1.) - self.ncomp(lambs, add=1.)
        alpha = self.a[None,:]*np.exp(-1j*2*np.pi/lambs[:,None] * (self.b[None,:] + \
                                                            self.dcomp[None,:] +\
                                                        self.c[None,:] * np1[:,None] +\
                                                        self.e[None,:] * np2[:,None]))
        return alpha
    def get_phasor_s(self, lambs):
        """
        Deprecated
        """
        # the air displacing solid plate:
        np1 = self.nplate(lambs) - self.ncomp(lambs, add=1.)
        # The air displacing second material:
        np2 = self.nmat2(lambs, add=1.) - self.ncomp(lambs, add=1.)
        alpha = self.a*np.exp(-1j*2*np.pi/lambs*(self.b + self.dcomp +\
                                                self.c*np1 + self.e*np2))
        return alpha
    def get_raw_phase_correction(self, lambs,
                                 b=0, c=0, e=0,
                                 dcomp=0, vector=None):
        """
        Returns the raw (non-wrapped) phase produced by an optical path
        of b[m] in air and c[m] in plate material.
        
        **Parameters**
        
        * lambs :     The wavelength channels to consider [m]
        * a     :     Vector of the amplitude term
        * b     :     Vettor of the geometric piston term [m]
        * c     :     Vettor of the dispersive piston term [m]
        * e     :     Vettor of the addtional corretction material term [m]
        * dcomp :     A length of air to compensate for the plate
        * vector :    The vector-form of all dispersive correction 
          shape: (n_materials, n_tel) [m]
        
        
        """
        # the air displacing solid plate:
        np1 = self.nplate(lambs) - self.ncomp(lambs, add=1.)
        # The air displacing second material:
        np2 = self.nmat2(lambs, add=1.) - self.ncomp(lambs, add=1.)
        #if model is None:
        #    model = self.prediction_model
        nair = self.ncomp(lambs, add=1)
        
        if vector is None:
            phase_correction = 2*np.pi/lambs*(nair*(b + dcomp) + np1*c + np2*e)
        else:
            if vector.shape[1] == 3:
                phase_correction = 2*np.pi/lambs*(nair*(vector[:,0] + dcomp) +\
                                                  np1*vector[:,1] + np2*vector[:,2])
            elif vector.shape[1] == 2:
                phase_correction = 2*np.pi/lambs*(nair*(vector[:,0] + dcomp) +\
                                                  np1*vector[:,1])
            elif vector.shape[1] == 1:
                phase_correction = 2*np.pi/lambs*(nair*(vector[:,0] + dcomp))
        return phase_correction

    def get_vector(self):
        """
            get_vector get the vector form of the current corrector setup.
        """
        vector = np.array([self.b, self.c, self.e])
        return vector
    
    def get_dcomp(self, c):
        """
        Returns the theoertical value of dcomp for a given value of compensator
        plate, to correct for the pure piston term introduced.
        **Arguments**:
        * c   : The value of glass plate [m]
        """
        dcomp = -(self.nmean-1)*c
        return dcomp
        
    def get_dcomp_from_vector(self, vector):
        """
        Returns the theoertical value of dcomp for a vector value of compensator
        plate, to correct for the pure piston term introduced.
        **Arguments**:
        * vector : The vector valued corrector position [m]
          (Only c at index 1 is used)
        """
        c = vector[1]
        dcomp = -(self.nmean-1)*c
        return dcomp
        

    def get_phasor_from_params(self, lambs, a=None,
                               b=None, c=None,
                               dcomp=None, e=None):
        """
        Similar to get_phasor() but allows to provide the
        parameters as arguments (slower).
        
        Returns the complex phasor corresponding
        to the current a, b, c, and dcomp phasors.
        
        **Parameters:**
        
        * lambs :     The wavelength channels to consider [m]
        * a     :     Vector of the amplitude term
        * b     :     Vector of the geometric piston term [m]
        * c     :     Vector of the dispersive piston term [m]
        * e     :     Vector of the addtional corretction material term [m]
        * dcomp :     A length of air to compensate for the plate [m]
          if dcomp is None: it will be computed based on `self.c`
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
        if e is None:
            e = self.e      
            
        # pdb.set_trace()
        
        # the air displacing solid plate:
        np1 = self.nplate(lambs) - self.ncomp(lambs, add=1.)
        # The air displacing second material:
        np2 = self.nmat2(lambs, add=1.) - self.ncomp(lambs, add=1.)
        #nplate -1 because it is air displacing glass

        alpha = a[None,:]*np.exp(-1j*2*np.pi/lambs[:,None] * (b[None,:] + \
                                                            dcomp[None,:] +\
                                                        c[None,:] * np1[:,None] +\
                                                        e[None,:] * np2[:,None]))
        return alpha
    
    def theoretical_phase(self,lambs, proj_opds, model=None, add=0, db=False, ref=None):
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
        * ref        : The reference wavelength for which phase is 0.
            - `None`
            - the value of a reference wavelength [m]
            - "center" : use the central bin of lambs
            - "mean" : use the mean of lambs
        
        **Returns:** phase [rad]
        """
        nair = model.get_Nair(lambs, add=add)
        if ref is None:
            nref = 0
        else:
            if isinstance(ref, str):
                if ref == "center":
                    ref = np.array([lambs[lambs.shape[0]//2]])
                elif ref == "mean":
                    ref = np.array([np.mean(lambs)])
            elif isinstance(ref, float):
                ref = np.array([ref,])
            # Here get_Nair expects an array to will return an array
            nref = model.get_Nair(ref, add=add)[0]
        if db:
            pdb.set_trace()
        # otherwise ref is the falue of the reference wavelength
        if len(proj_opds.shape) == 1:
            proj_opds_s = proj_opds[None,:]
        else:
            proj_opds_s = proj_opds.T
            
        phase = 2*np.pi * proj_opds_s * ((nair - nref) / lambs)[:,None]
        return phase
    
    def theoretical_piston(self,lambs, proj_opds, model=None, add=0, db=False, ref=None):
        """
        Computes the theoretical chromatic optical path effect of the
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
        * ref        : The reference wavelength for which opl is 0.
            - `None`
            - the value of a reference wavelength [m]
            - "center" : use the central bin of lambs
            - "mean" : use the mean of lambs
        
        **Returns:** piston [m]
        """
        nair = model.get_Nair(lambs, add=add)
        if ref is None:
            nref = 0
        else:
            if isinstance(ref, str):
                if ref == "center":
                    ref = np.array([lambs[lambs.shape[0]//2]])
                elif ref == "mean":
                    ref = np.array([np.mean(lambs)])
            elif isinstance(ref, float):
                ref = np.array([ref])
            nref = model.get_Nair(ref, add=add)
        # otherwise ref is the falue of the reference wavelength
        if db:
            pdb.set_trace()
        phase = proj_opds * (nair - nref)
        return phase.T
        
    def solve_air(self, lambs, model, corrector_shape=True):
        """
        Computes a least squares compensation model (see
        **Koresko et al. 2003 DOI: 10.1117/12.458032**)
        
        **Parameters:**
        
        * lambs :     The wavelength channels to consider [m]
        * model   : The wet atmosphere model (see n_air.wet_atmo object)
        * corrector_shape : (bool) if True: the matrix will be adjusted to
          give vectors adapted to a corrector with the correct number of materials.
        
        **Returns:** :math:`\Big( \mathbf{A}^T\mathbf{A}\mathbf{A}^T \Big)^{-1}`
        """
        nair = model.get_Nair(lambs, add=1)
        #nplate -1 because it is air displacing glass
        # ns = np.array([nair, (self.nplate(lambs)-1)]).T 
        ns = nair[:,None]
        A = 2*np.pi/lambs[:,None] * ns
        
        self.S = np.linalg.inv(A.T.dot(A)).dot(A.T)
        if corrector_shape:
            n_corrections = np.count_nonzero(np.array([True, 
                                        self.nplate is not no_material,
                                        self.nmat2 is not no_material]))
            S_0 = np.zeros((n_corrections, self.S.shape[1]))
            S_0[0] = self.S[0]
            self.S = S_0
            return self.S
        else:
            return self.S
        
        
                              
                              
    def solve_air_corrector(self, lambs):
        """
        Computes a least squares compensation model for 
        correction with variable thickness plates of material.
        (see **Koresko et al. 2003 DOI: 10.1117/12.458032**)
        
        This will use 
        
        **Parameters:**
        
        * lambs :     The wavelength channels to consider [m]
        
        **Returns:** :math:`\Big( \mathbf{A}^T\mathbf{A}\mathbf{A}^T \Big)^{-1}`
        """
        nair = self.ncomp(lambs,add=1)
        # If the corrector is equipped, add dof of
        # air displacing glass
        if self.nplate is not no_material:
             v2 = self.nplate(lambs,) - nair
        
        # If the corrector is equipped, add dof of
        # air displacing material
        if self.nmat2 is not no_material:
            v3 = self.nmat2(lambs, add=1) - nair
        
        if (self.nplate is not no_material) and (self.nmat2 is not no_material):
            ns = np.array([nair, v2, v3]).T 
        elif (self.nplate is not no_material) and not (self.nmat2 is not no_material):
            ns = np.array([nair, v2]).T 
        elif not (self.nplate is not no_material) and (self.nmat2 is not no_material):
            ns = np.array([nair, v3]).T 
        elif not (self.nplate is not no_material) and not (self.nmat2 is not no_material):
            ns = np.array([nair,]).T 
        else :
            raise NotImplementedError("Combination of materials not found")
            
        A = 2*np.pi/lambs[:,None] * ns
        self.S = np.linalg.inv(A.T.dot(A)).dot(A.T)
        return self.S

    def solve_air_model(self, lambs, models):
        """
        Computes a least squares compensation model (see
        **Koresko et al. 2003 DOI: 10.1117/12.458032**)
        
        **Parameters:**
        
        * lambs :     The wavelength channels to consider [m]
        * model   : The wet atmosphere model (see n_air.wet_atmo object)
        
        **Returns:** :math:`\Big( \mathbf{A}^T\mathbf{A}\mathbf{A}^T \Big)^{-1}`
        """
        
        ns = np.array([amodel.get_Nair(lambs) for amodel in models])
        # One of the components need to have the geometric element ()
        ns[0,:] = ns[0,:] +1
        A = 2*np.pi/lambs[:,None] * ns.T # We do the transpose here
        
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
        #pdb.set_trace()
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
            self.chip_corr = {"b":bvec,
                             "c":cvec,
                             "dcomp":-(self.nmean-1)*cvec}
            # These updates would be removed
            self.b = self.chip_corr["b"]
            self.c = self.chip_corr["c"]
            self.dcomp = self.chip_corr["dcomp"]
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
        """
        Plot of the state of tuning.
        
        **Parameters:**
        * lambs :     The wavelength range to consider to consider [m]
        * npoints :   The number number of points in the range to consider
        """
        from kernuller.diagrams import plot_chromatic_matrix as cmpc
        import matplotlib.pyplot as plt
        
        pltlambrange = np.linspace(np.min(lambs),
                                   np.max(lambs),
                                   npoints)
        init_phasor = cor.get_phasor(pltlambrange)
        