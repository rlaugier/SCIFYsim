import sympy as sp
import numpy as np
import scifysim as sf
import logging
from einops import rearrange
import dask.array as da
import copy
from pdb import set_trace
from scipy.linalg import sqrtm
from astropy import units

import matplotlib.pyplot as plt
import pickle

from scipy.stats import chi2, ncx2, norm
from scipy.optimize import leastsq

logit = logging.getLogger(__name__)

class noiseprofile(object):
    def __init__(self, integ, asim, diffobs_series, verbose=False, n_pixsplit=1):
        """
        An object to dynamically compute a parametric noise floor
        
        **Patameters:**
        
        * integ     : An integrator object augmented by some mean
                        and static measurements (see below)
        * asim      : A simulator object (mostly used to get some combiner
                        and injector parameters)
        * diffobs   : A series of realisations of te differential observable
                        intensity (e.g. `diffobs = np.einsum("ij, mkj->mk",asim.combiner.K, dit_intensity)`)
        
        
        **Creation makes use of:**
        
        * integ.mean_starlight (run ``integ.mean_starlight = np.mean(starlights, axis=0)``)
        * integ.mean_planetlight (run ``integ.mean_planetlight = np.mean(planetlights, axis=0)``)
        * integ.get_static() (run ``integ.static = asim.computed_static``)

        **Notes:**
        
        * `self.sigma_ron` in [e-] 
        * `self.s_enc_bg` is defined in [ph/s/pix]
        * `self.s_dark_current` is defined in [e-/s/pix]
        * `self.s_d` is defined in [ph/s/ch] (that is **per channel**)
        * `self.s_total_bg_current_ch` is in [e-/s/ch]
        
        """
        # Read noise
        self.mynpix = n_pixsplit
        self.eta = integ.eta
        self.sigma_ron = np.sqrt(self.mynpix) * integ.ron
        self.sigma_ron_d = np.sqrt(2) * self.sigma_ron
        self.mask_dark = asim.combiner.dark
        self.mask_bright = asim.combiner.bright
        self.mask_phot = asim.combiner.photometric
        self.K = asim.combiner.K
        
        # Photon noise
        self.dit_0 = asim.injector.screen[0].step_time * integ.n_subexps
        self.m_0, self.mp0 = asim.context.get_mags_of_sim(asim)
        self.F_0 = mag2F(self.m_0)
        self.s_0 = integ.mean_starlight / self.dit_0
        static, dark_current, enclosure = integ.get_static()
        self.s_dark_current = dark_current / self.dit_0
        self.s_enc_bg = enclosure / self.dit_0 / self.eta # Must convert that back in equ. photons
        self.s_d = static / self.dit_0 / self.eta # Must convert that back in equ. photon
        print(f" s_d = {self.s_d} [ph/s] the static output")
        print(f" s_dark_current = {self.s_dark_current} [ph-/s] equivalent the dark current")
        print(f" s_enc_bg = {self.s_enc_bg} [ph-/s] equivalent the enclosure background")
        self.p_0 = integ.mean_planetlight / self.dit_0
        # The total background current per channel
        self.s_total_bg_current_ch = self.mynpix * self.s_dark_current +\
                                    self.eta * self.mynpix * self.s_enc_bg +\
                                    self.eta * self.s_d.sum(axis=1)
        
        #self.sigma_phot = np.sqrt(integ.eta * self.s_0 * self.dit_0)
        #self.sigma_phot_d = np.sqrt(2) * self.sigma_phot

        # Sigma phi
        # Can only be computed on the differential observable
        self.diff_std = np.std(diffobs_series, axis=0)
        self.k_phi = self.diff_std / self.F_0
        # For getting a covariance
        self.diff_cov = np.cov(diffobs_series.T)
        self.k_phi_m = self.diff_cov / self.F_0**2


    def get_noises(self, m_star, dit, matrix=False):
        """Returns the different noises
        
        .. admonition:: Warning!
        
            If matrix is True, then sigma_phi_d is a covariance matrix (beware of variances on diagonal)
        
        **Arguments:**
        
        * m_star       : Magnitude of the star
        * dit          : Detector integration time [s]
        
        **Returns:**
        
        ``sigma_phot_d, sigma_phi_d`` : standard deviation vectors (or
        covariance matrix for ``sigma_phi_d``)
        """
        Fs = mag2F(m_star)
        
        # small s is a flux
        s_s = Fs/self.F_0 * self.s_0
        s_d = self.s_d + self.mynpix*(self.s_dark_current + self.s_enc_bg)[:,None]
        
        
        sigma_phot = np.sqrt(self.eta * dit * (s_d + s_s))
        vark = np.einsum("ki, wi-> wk", sigma_phot**2, np.abs(self.K))
        sigma_phot_d = np.sqrt(vark.flatten())
        # sigma_phot_d = np.sqrt(np.sum(sigma_phot[:,self.mask_dark]**2, axis=-1)) # kernel will need different treatment here
        sigma_phot_photo = sigma_phot[:,self.mask_phot]
        sigma_phot_bright = sigma_phot[:,self.mask_bright]
        
        if matrix:
            Sigma_phi_d = self.eta**2 * self.k_phi_m * Fs**2
            return sigma_phot_d, Sigma_phi_d
        else:
            sigma_phi_d = self.eta * self.k_phi * Fs
            return sigma_phot_d, sigma_phi_d

    def diff_noise_floor_dit(self, m_star, dit, matrix=False):
        """Returns the compound noise floor
        WARNING: if matrix is True, then the result is
        a covariance matrix (beware of variances on diagonal)
        
        **Arguments:**
        
        * m_star       : Magnitude of the star
        * dit          : Detector integratioon time [s]
        
        **Returns:**
        
        Either ``sigma_tot_diff`` or ``cov_tot_diff`` depending on
        ``matrix``.
        """
        
        #print(self.sigma_ron_d.shape)
        #print(sigma_phot_d.shape)
        #print(sigma_phi_d.shape)
        if not matrix:
            sigma_phot_d, sigma_phi_d = self.get_noises(m_star, dit, matrix=matrix)
            sigma_tot_diff = np.sqrt(\
                                    self.sigma_ron_d**2\
                                   + sigma_phot_d**2\
                                   + sigma_phi_d**2)
            return sigma_tot_diff
        else: # Note that in this case a covariance matrix will be used
            sigma_phot_d, Sigma_phi_d = self.get_noises(m_star, dit, matrix=matrix)
            cov_tot_diff = self.sigma_ron_d**2*np.eye(sigma_phot_d.shape[0])\
                                   + np.diag(sigma_phot_d**2)\
                                   + Sigma_phi_d
            return cov_tot_diff
        
    def plot_noise_sources(self, wls, dit=1., starmags=np.linspace(2., 6., 5),
                          show=True, legend_loc="best", legend_font="x-small",
                          ymin=0., ymax=1., n_dits=1):
        
        sigphis = []
        sigphots = []
        for amag in starmags:
            asigphot, asigphi = self.get_noises(amag, dit)
            sigphots.append(1/np.sqrt(n_dits) * asigphot)
            sigphis.append(1/np.sqrt(n_dits) * asigphi)
        sigphots = np.array(sigphots)
        sigphis = np.array(sigphis)
        
        
        mybmap = plt.matplotlib.cm.Blues
        mygmap = plt.matplotlib.cm.YlOrBr
        fig = plt.figure(dpi=200)
        plt.plot(wls, 1/np.sqrt(n_dits) * self.sigma_ron_d * np.ones_like(wls),color="gray",
                 label=r"$\sigma_{RON}$")
        #plt.plot(wls, sigma_phot, label="Photon")
        for i, dump in enumerate(sigphis):
            msg = f"$\sigma_{{inst}}$ (mag = {starmags[i]:.1f})"
            cvalue = i/sigphis.shape[0]
            aline = plt.plot(wls,
                                 sigphis[i,:], label=msg,
                                 color=mybmap(sf.utilities.trois(cvalue, 0.,1., ymin=ymin, ymax=ymax)))
        for i, dump in enumerate(sigphots):
            msg = f"$\sigma_{{phot}}$ (mag = {starmags[i]:.1f})"
            cvalue = i/sigphis.shape[0]
            aline = plt.plot(wls,
                                 sigphots[i,:], label=msg,
                                 color=mygmap(sf.utilities.trois(cvalue, 0.,1., ymin=ymin, ymax=ymax)))
            #plt.text(np.min(wls), asnr[0], msg)
        plt.yscale("log")
        plt.ylabel("Noise $[e^-]$")
        plt.xlabel("Wavelength [m]")
        plt.legend(fontsize=legend_font, loc=legend_loc)
        plt.title(f"Noise for {n_dits} x {dit:.2f} s DITs")
        if show:
            plt.show()
        return fig
    
    def summary(self):
        print("n_pix = ", self.mynpix, "# The number of pixels used per spectral channel")
        print("t_dit_0 = ", self.dit_0, "# The reference exposure time used in MC simulations")
        print("eta = ", self.eta, "# The quantum efficiency of the detector")
        print("sigma_ron_d = ", self.sigma_ron_d, "# The total readout noise per spectral channel for the differential observable(s)")
        print("k_phi = ", self.k_phi_m, "# The instrumental error coefficient")
        print(f" s_d = {self.s_d} [ph/s] the static output")
        print(f" s_dark_current = {self.s_dark_current} [e-/s] the dark current")
        print(f" s_enc_bg = {self.s_enc_bg} [e-/s] the enclosure background")
        

class spectral_context(object):
    def __init__(self, vegafile="config/vega.ini", compensate_chromatic=True,
                    verbose=False):
        """Spectral context for magnitude considerations
        
        **Arguments:**
        
        * vegafile    : Either
            
            - The parsed config file of the original simulator:
              a modified copy will be created for the reference 
              Vega observations.
            - The path *str* to a file for observation of Vega
              in the same spectral configuration.
        
        * compensate_chromatic: Argument to be passed to the new simulator
        * verbose     : whether to give more details
        
        """
        if isinstance(vegafile, str):
            raise AttributeError("no longer supported\
                Please provide a ConfigParser object (asim.config)")
            self.avega = sf.utilities.prepare_all(vegafile, update_params=False,
                            instrumental_errors=False, compensate_chromatic=compensate_chromatic,
                            verbose=verbose)
        elif isinstance(vegafile, sf.parsefile.ConfigParser):
            vega_config = self.create_vega_config(vegafile)
            if vega_config.get("configuration", "location") == "space":
                use = False
            else:
                use = True
            self.avega = sf.utilities.prepare_all(vega_config, update_params=False,
                            instrumental_errors=False,
                            compensate_chromatic=compensate_chromatic,
                            verbose=verbose,
                            update_start_end=use)
            
        else :
            raise TypeError()
        self.thevegassflux = self.avega.src.star.ss.sum(axis=1)
        
    def create_vega_config(self, config, tarloc="Kraz"):
        """
        Builds the config file for a reference observation of Vega
        
        **Arguments:**
        
        * config: A parsed config file from the simulator itself.
        * tarloc: A star used as a proxy to observe close to Zenith.
          Kraz is a good fit for Paranal.
        
        **Returns** a modified parsed ``config`` file that has the same settings but that
        is designed to reproduce an observation of Vega near zenith. Most parameters
        describing the instrument are the same. A shorter sequence is created (only 
        the central one is to be used: index 3). The target is chosen so that it passes
        near zenith at observatory.
        """
        from copy import deepcopy
        veg_config = deepcopy(config)
        veg_config.set("target", "target", "Kraz")
        veg_config.set("target", "star_radius", "2.36")
        veg_config.set("target", "star_distance", "7.68")
        veg_config.set("target", "star_temperature", "9600.")
        veg_config.set("target", "planet_radius", "0.000001")
        veg_config.set("target", "planet_temperature", "0.")
        veg_config.set("target", "n_points", "7")
        return veg_config
    
    def sflux_from_vegamag(self, amag):
        """inverse of vegamag_from_ss()"""
        f = self.thevegassflux*10**(-amag/2.5)
        return f
    def vegamag_from_ss(self, flux):
        """inverse of sflux_form_vegamag()"""
        amag = -2.5*np.log10(flux/self.thevegassflux)
        return amag
    
    
    def get_mags_of_sim(self, asim):
        """Magnitudes of planet and star of a simulator object
        
        **Arguments:**
        
        * asim     : A simulator object
        
        """
        ss_star = asim.src.star.ss
        ss_planet = asim.src.planet.ss

        if len(asim.src.star.ss.shape) == 2:
            ss_star = asim.src.star.ss.sum(axis=1)
        else :
            ss_star = asim.src.star.ss

        if len(asim.src.planet.ss.shape) == 2:
            ss_planet = asim.src.planet.ss.sum(axis=1)
        else :
            ss_planet = asim.src.planet

        m_star = np.mean(self.vegamag_from_ss(ss_star))
        m_planet = np.mean(self.vegamag_from_ss(ss_planet))
        
        return m_star, m_planet

    def get_difmap_flux(self, sim, map=None, mode="e-", K=None,
                        planet=None, single_out=True):
        """
        Get the map in terms of flux of a given planet
        Not kernel-ready
        """
        if K is None:
            K = sim.combiner.K
        if map is None:
            map = sim.maps
        if planet is None:
            planet = sim.src.planet
        # adiffmap = map[:,:,3,:,:] - map[:,:,4,:,:]
        # flux_map = adiffmap / \
        #     sim.vigneting_map.ds_sr * planet.ss.sum(axis=1)[None,:,None,None]
        # else:
        adiffmap = np.einsum("k o , s w o x y -> s w k x y", K, map)
        flux_map = adiffmap / \
            sim.vigneting_map.ds_sr * planet.ss.sum(axis=1)[None,:,None,None,None]
        if single_out:
            flux_map =  flux_map[:,:,0,:,:] 
        if mode == "e-":
            return sim.integrator.eta * flux_map
        elif mode == "ph":
            return flux_map

        
    def get_maxmap(self, map, time_index=None, max=True):
        """Returns the maximum of the map

        **Arguments:**
        * map:  A map object (asim.maps)
        * time_index: An index for the (asim.sequence)
          observing series
        * max : (True) Returns only the max signal of the map
        """
        if time_index is None:
            time_index = map.shape[0]//2
        adiffmap = map[time_index,:,3,:,:] - map[time_index,:,4,:,:]
        if max:
            return np.max(adiffmap, axis=(1,2))
        else:
            return adiffmap

    def get_planet_max_signal(self, sim, planet=None, max=True,
                                time_index=None):
        """Returns the differential flux signal ph/s collected
        for a given planet

        **Arguments:**
        * sim : A director object
        * planet : a source.
        * time_index: An index for the (asim.sequence)
          observing series
        * max : (True) Returns only the max signal of the map
        
        """
        if planet is None:
            planet=sim.src.planet
        if max:
            amax = self.get_maxmap(sim.maps, time_index=time_index)
            planet_signal = amax/sim.vigneting_map.ds_sr * planet.ss.sum(axis=1)
            return planet_signal
        else:
            raise NotImplementedError
        

class simplified_context(spectral_context):
    def __init__(self, mycontext): 
        self.thevegassflux = mycontext.thevegassflux

    def to_pickle(self, apath):
        """
        Saving the context to a pickle file.
        """
        with open(apath, "wb") as thefile:
            pickle.dump(self, thefile)
        
    
def mag2F(mag):
    F = 1*10**(-mag/2.5)
    return F
def F2mag(F):
    mag = -2.5*np.log10(F/1)
    return mag

class BasicETC(object):
    def __init__(self, asim):
        """
        **Parameters:**
        * ipeak : []
        * pdiam : [m]
        * S_collecting [m^2]
        * lambda_science_range [m]
        * diffuse []
        * throughput []
        * eta [e-/ph]
        * fiber_etendue [sr]
        * cold_enclosure [e-/s]
        * dark_current [e-/s]
        * src_names []
        * contribs [ph/s]
        
        """
        asim.integrator.update_enclosure(asim.lambda_science_range)
        self.peak_contributions = np.abs(asim.combiner.Mcn[:,3,:].mean(axis=0))
        self.ipeak = np.sum(self.peak_contributions)**2
        self.pdiam = asim.injector.pdiam * units.m
        self.odiam = asim.injector.odiam * units.m
        self.S_collecting = np.pi * self.pdiam**2/4 - np.pi * self.odiam**2/4
        self.lambda_science_range = asim.lambda_science_range
        self.diffuse = asim.diffuse
        self.throughput = self.diffuse[0].get_downstream_transmission(self.lambda_science_range)
        self.eta = asim.integrator.eta * units.electron / units.photon
        self.fiber_etendue = np.pi * (self.lambda_science_range / self.pdiam)**2
        contribs = []
        self.src_names = []
        self.cold_enclosure = asim.integrator.det_sources[0]
        self.dark_current = asim.integrator.det_sources[1]
        self.dit_0 = asim.config.getfloat("detector", "default_dit") * units.s
        self.ron = asim.integrator.ron * units.electron
        for anelement in self.diffuse:
            contribs.append(self.fiber_etendue \
                            * anelement.get_own_brightness(self.lambda_science_range) \
                            * anelement.get_downstream_transmission(self.lambda_science_range))
            self.src_names.append(anelement.__name__)
        contribs.append(asim.integrator.det_sources[0] \
                        / self.eta)
        self.src_names.append(asim.integrator.det_labels[0])
        contribs.append(asim.integrator.det_sources[1] \
                        * np.ones_like(asim.integrator.det_sources[0])
                        / self.eta)
        self.src_names.append(asim.integrator.det_labels[1])
        self.contribs = np.array(contribs) * units.photon/units.s
        self.contribs_current = self.eta * self.contribs

    @property
    def contribs_variance_rate(self):
        """The variance rate in e-**2/s"""
        variance_rate = self.contribs_current.sum(axis=0).value \
                        * units.electron**2/units.s
        return variance_rate

    def get_min_exp_time(self, planet_mag, planet_T, snr=3., dit_0=None,
                        verbose=False):
        """
        **Arguments**:
        * planet_mag : [Vega mag]
        * planet_T   : [K]
        * snr        : (3) Signal to noise ratio to reach
        * dit_0      : (None) [s]
        * verbose    : (False)
        """
        if dit_0 is None:
            dit_0 = self.dit_0
        if not isinstance(dit_0, units.quantity.Quantity):
            dit_0 = dit_0 * units.s
        total_current_planet, total_noise = self.show_signal_noise(planet_mag, dit=1.,
                                                    T=planet_T, verbose=verbose,
                                                    plot=False, show=False)
        at = snr**2* np.sqrt(2) * self.contribs_variance_rate \
                            /total_current_planet**2
        at2 = snr**2 * np.sqrt(2) \
            * ( self.contribs_variance_rate \
                        + self.ron**2/dit_0) \
              /total_current_planet**2 
        return at, at2


    def planet_photons(self, planet_mag, dit=1., T=None, verbose=False):
        """
        **Arguments**:
        * planet_mag : 
        * dit        : (1.) [s]
        * T          T (None) []
        * verbose    : (False)
        """
        if not isinstance(dit, units.quantity.Quantity):
            dit = dit * units.s
        if T is None:
            self.fd_planet = sf.sources.mag2flux_bin(
                            self.lambda_science_range, planet_mag
            )
        else:
            self.fd_planet = sf.sources.magT2flux_bin(
                            self.lambda_science_range,
                            planet_mag,
                            T=T
            )
        return dit * self.fd_planet

    def show_signal_noise(self, planet_mag, dit=1., T=None, verbose=True,
                        decompose=True, plot=True,
                        show=True):
        """
        **Arguments**:
        * planet_mag : The magnitude of the planet in the science band (Vega).
        * dit        : (1.) The individual detector integration time. Typically
          adjusted for different spectral resolutions. [s]
        * T          : (None) Temperature [K]
        * verbose    : (True)
        * decompose  : (True)
        * plot       : (True)
        * show       : (True)
        """
        planet_photons = self.planet_photons(planet_mag, dit=dit, T=T,
                                            verbose=verbose)
        planet_electrons = planet_photons \
                        * self.ipeak * self.throughput \
                        * self.S_collecting * self.eta 
        if not isinstance(dit, units.quantity.Quantity):
            dit = dit * units.s
        if verbose:
            print(f"Planet photons {np.sum(planet_photons)/dit:.1e} [ph/s] \
                        for a total of {np.sum(planet_photons):.1e} [ph]")
            print(f"Mean total transmission {100*np.mean(self.throughput):.2f} \%")
            print(f"Collecting {self.S_collecting:.1e} [m^2]")
            print(f"Quantum efficiency {self.eta:.2f}")
            print(f"Planet e- {np.sum(planet_electrons)/dit:.1e} [e-/s] \
                        for a total of {np.sum(planet_electrons):.1e} [e-]")
        background_photons = np.sum(self.contribs * dit, axis=0)
        background_electrons = background_photons * self.eta
        background_sigma = np.sqrt(background_electrons.value)
        total_background_sigma = np.sqrt(background_electrons.sum())*units.electron
        if verbose:
            print(f"Background photons {background_photons.sum()/dit:.1e} [ph/s] \
                        for a total of {background_photons.sum():.1e} [ph]")
            print(f"Background electrons {background_electrons.sum()/dit:.1e} [ph/s] \
                        for a total of {background_electrons.sum():.1e} [ph]")
            print(f"Photon noise std {total_background_sigma:.1e} [e-]")
            print(f"SNR min {np.min(planet_electrons/background_sigma)}\
                max {np.max(planet_electrons/background_sigma)}")
        if plot:
            contribs_current = self.contribs_current * dit
            import matplotlib.pyplot as plt
            afig = plt.figure()
            base = np.zeros_like(np.sqrt(contrib2variance(contribs_current[0])))
            print(base.unit)
            for i, acontrib in enumerate(contribs_current):
                acontrib_variance = contrib2variance(acontrib)
                print(acontrib_variance.unit)
                plt.fill_between(self.lambda_science_range,
                    y1=base, y2=np.sqrt(base**2 + acontrib_variance).value,
                    label=f"$\\sigma$ {self.src_names[i]}")
                base = np.sqrt(base**2 + acontrib_variance)
            plt.plot(self.lambda_science_range, 
                    np.sqrt(np.sum(contribs_current, axis=0)).value,
                    color="r", linestyle="--", label="Total background noise")
            plt.plot(self.lambda_science_range,
                    planet_electrons,
                    label=f"Planet L={planet_mag:.1f}, {T:.0f}K", color="k",)
            plt.legend(fontsize="x-small")
            plt.xlabel("Wavelength [m]")
            plt.ylabel("Planet [e-] \n $\\sigma_{background}$ [e-]")
            plt.title(f"DIT = {dit} [s]")
            if show:
                plt.show()
        if plot and not show:
            return planet_electrons, background_sigma, afig
        else:
            return planet_electrons, background_sigma
        
        
        
            
        
def contrib2variance(acontrib, unit=units.electron):
    """
        Assumes incoming signals proportional to electrons.
    Tweaks the unit to a poisson variance (e.g. from e- to e-**2)
    """
    avar = acontrib*unit
    return avar

##################################
# The energy detector test
##################################

def pdet_e(lamb, xsi, rank):
    """
    Computes the residual of Pdet
    
    **Arguments:**
    
    * lamb     : The noncentrality parameter representing the feature
    * xsi      : The location of threshold
    * rank     : The rank of observable
    
    **Returns** the Pdet difference
    """
    respdet = 1 - ncx2.cdf(xsi,rank,lamb)
    return respdet
def residual_pdet_Te(lamb, xsi, rank, targ):
    """
    Computes the residual of Pdet
    
    **Arguments:**
    
    * lamb     : The noncentrality parameter representing the feature
    * targ     : The target Pdet to converge to
    * xsi      : The location of threshold
    * rank     : The rank of observable
    
    **Returns** the Pdet difference. See *Ceau et al. 2019* for more information
    """
    respdet = 1 - ncx2.cdf(xsi,rank,lamb) - targ
    return respdet
def get_sensitivity_Te_old(maps, pfa=0.002, pdet=0.9, postproc_mat=None, ref_mag=10.,
                verbose=True):
    """
    **Deprecated**
    
    Magnitude map at which a companion is detectable for a given map and whitening matrix.
    The throughput for the signal of interest is given by the map at a reference magnitude.
    The effective noise floor is implicitly described by the whitening matrix.
    
    **Parameters**
    
    * maps       : Differential map for a given reference magnitude
    * pfa        : The false alarm rate used to determine the threshold
    * pdet       : The detection probability used to determine the threshold
    * postproc_mat : The matrix of the whitening transformation ($\Sigma^{-1/2}$)
    * ref_mag    : The reference mag for the given map
    * verbose    : Print some information along the way
    
     See *Ceau et al. 2019* for more information
     
    **Returns**
    
    * mags : The magnitude at which Te crosses the threshold
    * fluxs: The fluxes at which Te crosses the threshold
    * Tes  : The values of the test statistic on the map at the reference magnitude
    
    """
    W = postproc_mat
    if len(W.shape) == 3:
        nbkp = W.shape[0]*W.shape[1]
    elif len(W.shape)==2:
        nbkp = W
    else:
        raise NotImplementedError("Wrong dimension for post-processing matrix")
    if len(maps.shape) == 5:
        nt, nwl, nout, ny, nx = maps.shape
    elif len(maps.shape) == 4:
        nt, nwl, nout, no = maps.shape
        raise NotImplementedError("single_datacube")
    else : 
        raise NotImplementedError("Shape not expected")
    f0 = sf.analysis.mag2F(ref_mag) #flux_from_vegamag(ref_mag)
    
    if postproc_mat is not None:
        if len(postproc_mat.shape):
            postproc_mat.shape[0]
    #Xsi is the threshold
    xsi = chi2.ppf(1.-pfa, nbkp)
    lambda0 = 0.2**2 * nbkp
    #The solution lamabda is the x^T.x value satisfying Pdet and Pfa
    sol = leastsq(residual_pdet_Te, lambda0,args=(xsi,nbkp, pdet))# AKA lambda
    lamb = sol[0][0]
    fluxs = []
    mags = []
    Tes = []
    if verbose:
        print("xsi = ", xsi)
        print("nbkp = ", nbkp)
        print("lambda0 = ", lambda0)
        print("lambda = ", lamb)
    #print(W.shape)
    for (i, j), a in np.ndenumerate(maps[0,0,0,:,:]):
        wsig = []
        for k in range(nt):
            mapsig = maps[k,:,:,i,j].flatten()
            wsig.append(1/f0 * W[k,:,:].dot(mapsig))
        wsig = np.array(wsig).flatten()
        #wsig = 1/f0 * np.array([W[k,:,:].dot(maps[k,:,:,i,j].flat) for k in range(nt) ]).reshape((nt*nwl*nout))
        flux = np.sqrt(lamb) / np.sqrt(np.sum(wsig.T.dot(wsig)))
        Te = np.sum((wsig*f0).T.dot(wsig*f0)).astype(np.float64)
        Tes.append(Te)
        fluxs.append(flux)
        mags.append(sf.analysis.F2mag(flux))
    fluxs = np.array(fluxs).reshape((ny, nx))
    mags = np.array(mags).reshape((ny,nx))
    Tes = np.array(Tes).reshape((ny,nx))
    #print(wsig.shape)
    
    return mags, fluxs, Tes
def get_sensitivity_Te(maps, mod=np, pfa=0.002, pdet=0.9, postproc_mat=None, ref_mag=10.,
                verbose=True):
    """
    Magnitude map at which a companion is detectable for a given map and whitening matrix.
    The throughput for the signal of interest is given by the map at a reference magnitude.
    The effective noise floor is implicitly described by the whitening matrix.
    
    **Parameters**
    
    * maps       : Differential map for a given reference magnitude and exposure time
    * mod        : The math module to use for vector ops default=numpy
    * pfa        : The false alarm rate used to determine the threshold
    * pdet       : The detection probability used to determine the threshold
    * postproc_mat : The matrix of the whitening transformation ($\Sigma^{-1/2}$)
    * ref_mag    : The reference mag for the given map
    * verbose    : Print some information along the way
    
     See *Ceau et al. 2019* for more information
     
    **Returns**
    
    * mags : The magnitude at which Te crosses the threshold
    * fluxs: The fluxes at which Te crosses the threshold
    * Tes  : The values of the test statistic on the map at the reference magnitude
    
    """
    W = postproc_mat
    if len(W.shape) == 3:
        nbkp = W.shape[0]*W.shape[1]
    elif len(W.shape)==2:
        nbkp = W.shape[0]
    else:
        raise NotImplementedError("Wrong dimension for post-processing matrix")
    if len(maps.shape) == 5:
        nt, nwl, nout, ny, nx = maps.shape
    elif len(maps.shape) == 4:
        nt, nwl, nout, no = maps.shape
        raise NotImplementedError("single_datacube")
    else : 
        raise NotImplementedError("Shape not expected")
    f0 = sf.analysis.mag2F(ref_mag) #flux_from_vegamag(ref_mag)
    
    if postproc_mat is not None:
        if len(postproc_mat.shape):
            postproc_mat.shape[0]
    #Xsi is the threshold
    xsi = chi2.ppf(1.-pfa, nbkp)
    if verbose:
        print("nbkp = ", nbkp)
        print(xsi)
    lambda0 = 0.2**2 * nbkp
    #The solution lambda is the x^T.x value satisfying Pdet and Pfa
    sol = leastsq(residual_pdet_Te, lambda0,args=(xsi,nbkp, pdet))# AKA lambda
    lamb = sol[0][0]
    w2 = 1/f0 * W
    # Concatenate the wavelengths
    kmap = maps.reshape((maps.shape[0], maps.shape[1]*maps.shape[2], maps.shape[3], maps.shape[4]))
    # Apply the whitening
    wsig = mod.einsum("s u k , s k y x -> s u y x", w2, kmap)
    # Concatenate the observing sequence blocks
    wsig = wsig.reshape(wsig.shape[0]*wsig.shape[1], wsig.shape[2], wsig.shape[3])
    xtx = mod.einsum(" k y x , k y x -> y x ", wsig, wsig)
    fluxs = np.sqrt(lamb) / np.sqrt(xtx)
    mags = sf.analysis.F2mag(fluxs)
    Tes = mod.einsum("k y x , k y x -> y x ", wsig*f0, wsig*f0)
    
    return mags, fluxs, Tes



##################################
# The Neyman-Pearson test
##################################


def get_Tnp_threshold(x, Pfa):
    if len(x.shape) == 1:
        xTx = x.T.dot(x)
    elif len(x.shape) == 5:
        xTx = np.einsum("ijklm, ijklm -> lm", x, x)
    return np.sqrt(xTx)*norm.ppf(1-Pfa)
def get_Tnp_threshold_map(xTx_map, Pfa):
    if len(xTx_map.shape) == 2:
        #xTx = x.T.dot(x)
        return np.sqrt(xTx_map)*norm.ppf(1-Pfa)
def residual_pdet_Tnp(xTx, xsi, targ):
    """
    Computes the residual of Pdet in a NP test.
    
    **Arguments:**
    
    * xTx     : The noncentrality parameter representing the feature
    * targ     : The target Pdet to converge to
    * xsi      : The location of threshold
    
    **Returns** the Pdet difference.
    
    """
    
    Pdet_Pfa = 1 - norm.cdf((xsi - xTx)/np.sqrt(xTx))
    return Pdet_Pfa - targ
def get_sensitivity_Tnp(maps, pfa=0.002, pdet=0.9, postproc_mat=None, ref_mag=10.,
                verbose=False, use_tqdm=True):
    """
    Magnitude map at which a companion is detectable for a given map and whitening matrix.
    The throughput for the signal of interest is given by the map at a reference magnitude.
    The effective noise floor is implicitly described by the whitening matrix.
    
    **Arguments:**
    
    * maps       : Differential map for a given reference magnitude
    * pfa        : The false alarm rate used to determine the threshold
    * pdet       : The detection probability used to determine the threshold
    * postproc_mat : The matrix of the whitening transformation ($\Sigma^{-1/2}$)
    * ref_mag    : The reference mag for the given map
    * verbose    : Print some information along the way
    
    **Returns**: A magnitude map
    
    """
    from scipy.stats import norm
    W = postproc_mat
    if len(W.shape) == 3:
        nbkp = W.shape[0]*W.shape[1]
    elif len(W.shape)==2:
        nbkp = W.shape[0]
    else:
        raise NotImplementedError("Wrong dimension for post-processing matrix")
    if len(maps.shape) == 5:
        nt, nwl, nout, ny, nx = maps.shape
    elif len(maps.shape) == 4:
        nt, nwl, nout, no = maps.shape
        raise NotImplementedError("single_datacube")
    else : 
        raise NotImplementedError("Shape not expected")
    f0 = sf.analysis.mag2F(ref_mag) #flux_from_vegamag(ref_mag)
    # Concatenation of the wavelengths:
    rmap = maps.reshape((maps.shape[0], maps.shape[1]*maps.shape[2], maps.shape[3], maps.shape[4]))
    x_map = 1 * np.einsum("iok, iklm-> iolm",
                     postproc_mat, rmap)
    # Concatenation of the sequence
    x_map = x_map.reshape((x_map.shape[0]*x_map.shape[1], x_map.shape[2], x_map.shape[3]))
    #x_map = rearrange(x_map, "a b c d -> (a b) c d")
    #Xsi is the threshold
    xTx_map = np.einsum("olm, olm -> lm", x_map, x_map)
    xsi_0 = get_Tnp_threshold_map(xTx_map, pfa)
    
    F_map = f0/np.sqrt(xTx_map) * ( norm.ppf(1-pfa, loc=0) - norm.ppf(1-pdet, loc=xTx_map, scale=np.sqrt(xTx_map)))
    mag_map = sf.analysis.F2mag(F_map)
    if verbose:
        plt.figure()
        plt.imshow(mag_map)
        plt.colorbar()
        plt.show()
        print(x_map.shape)
        print(xTx_map.dtype)
        plt.figure()
        plt.imshow(xsi_0)
        plt.colorbar()
        plt.show()
    return mag_map

def get_sensitivity_Tnp_old(maps, pfa=0.002, pdet=0.9, postproc_mat=None, ref_mag=10.,
                verbose=False, use_tqdm=True):
    """
    **Deprecated**
    
    Magnitude map at which a companion is detectable for a given map and whitening matrix.
    The throughput for the signal of interest is given by the map at a reference magnitude.
    The effective noise floor is implicitly described by the whitening matrix.
    
    **Arguments:**
    
    * maps       : Differential map for a given reference magnitude
    * pfa        : The false alarm rate used to determine the threshold
    * pdet       : The detection probability used to determine the threshold
    * postproc_mat : The matrix of the whitening transformation ($\Sigma^{-1/2}$)
    * ref_mag    : The reference mag for the given map
    * verbose    : Print some information along the way
    
    **Returns**: A magnitude map
    
    """
    from scipy.stats import norm
    W = postproc_mat
    if len(W.shape) == 3:
        nbkp = W.shape[0]*W.shape[1]
    elif len(W.shape)==2:
        nbkp = W
    else:
        raise NotImplementedError("Wrong dimension for post-processing matrix")
    if len(maps.shape) == 5:
        nt, nwl, nout, ny, nx = maps.shape
    elif len(maps.shape) == 4:
        nt, nwl, nout, no = maps.shape
        raise NotImplementedError("single_datacube")
    else : 
        raise NotImplementedError("Shape not expected")
    f0 = sf.analysis.mag2F(ref_mag) #flux_from_vegamag(ref_mag)
    
    x_map = 1 * np.einsum("iok, iklm-> iolm",
                     postproc_mat, rearrange(maps, "a b c d e -> a (b c) d e"))
    x_map = rearrange(x_map, "a b c d -> (a b) c d")
    #Xsi is the threshold
    xTx_map = np.einsum("olm, olm -> lm", x_map, x_map)
    xsi_0 = get_Tnp_threshold_map(xTx_map, pfa)
    
    F_map = f0/np.sqrt(xTx_map) * ( norm.ppf(1-pfa, loc=0) - norm.ppf(1-pdet, loc=xTx_map, scale=np.sqrt(xTx_map)))
    mag_map = sf.analysis.F2mag(F_map)
    if verbose:
        plt.figure()
        plt.imshow(mag_map)
        plt.colorbar()
        plt.show()
        print(x_map.shape)
        print(xTx_map.dtype)
        plt.figure()
        plt.imshow(xsi_0)
        plt.colorbar()
        plt.show()
    return mag_map


## Tools for interpretation 

def correlation_map(signal, maps, postproc=None, K=None, n_diffobs=1, verbose=False):
    """
    Returns the raw correlation map of a signal with a map.
    
    **Arguments:**
    
    * signal      : The signal measured on sky shape:
      (n_slots, n_wl, n_outputs)
    * maps        : The maps for comparison shape:
      (n_slots, n_wl, n_outputs, x_resolution, y_resolution)
    * postproc    : The whitening matrix to use 
    * n_diffobs : The number of differential observables for this combiner
      (1 for double bracewell, 3 for VIKiNG)
    
    **Returns**
    
    * cmap1 : The correlation map
    * xtx_map : The normalization map
    """
    assert (len(signal.shape) == 3), "Input shape (slots, wavelengths, outputs)"
    assert (len(maps.shape) == 5), "Maps shape (slots, wavelengths, outputs, position, position)"
    
    print("n_diffobs", n_diffobs)
    print("shape2", maps.shape[2])
    if signal.shape[2] != n_diffobs:
        diffobs = np.einsum("ij, mkj->mki", K, signal)
    elif signal.shape[2] == n_diffobs:
        diffobs = signal
    
    if maps.shape[2] != n_diffobs:
        difmap = np.einsum("ij, kljmn -> klimn", K, maps)
    elif maps.shape[2] == n_diffobs:
        difmap = maps
    if K is not None:
        wmap = np.einsum("ijk, iklm -> ijlm", postproc, rearrange(difmap, "a b c d e -> a (b c) d e"))
        wsig = np.einsum("ijk, ik -> ij", postproc, rearrange(diffobs, "a b c -> a (b c)"))
    else:
        wmap = rearrange(difmap, "a b c d e -> a (b c) d e")
        wsig = rearrange(diffobs, "a b c -> a (b c)")
    if verbose:
        print(wmap.shape)
        print(wsig.shape)
        xsi = chi2.ppf(1.-0.01, wsig.flatten().shape[0])
        print(f"xsi = {xsi}")
        print(f"yty = {wsig.flatten().dot(wsig.flatten())}")
    
    cmap1 = np.einsum("ijkl, ij -> kl", wmap, wsig)
    xtx_map = np.einsum("ijkl, ijkl -> kl", wmap, wmap)
    #cmap2 = np.einsum("ikl, i -> kl", rearrange(wmap, "a b c d -> (a b) c d"), wsig.flatten())
    
    return cmap1, xtx_map


def make_source(params, lambda_range, distance):
    """
    Creates a source from scratch for purpose of model fitting.
    
    **Arguments:**
    
    * params: An lmfit Parameters object containing:
        - "Sep" separation in [mas]
        - "PA" position angle in [deg]
        - "Radius" The radius in [R_sun]
        - "Temperature" The temperature [K]
    * lambda_range : array of the wavelength channels used [m]
    * distance: Distance to the system [pc]
    """
    planet_separation = params["Sep"].value
    planet_position_angle = params["PA"].value
    planet_offsetx = -planet_separation*np.sin(planet_position_angle * np.pi/180)
    planet_offsety =  planet_separation*np.cos(planet_position_angle * np.pi/180)
    planet_offset = (planet_offsetx, planet_offsety)
    mysource = sf.sources.resolved_source(lambda_range, distance,
                                          resolved=False,
                                         radius=params["Radius"].value,
                                         T=params["Temperature"].value,
                                         offset=planet_offset)
    return mysource

def get_planet_residual(my_params, target_signal, asim, dit, K, postproc, diffuse, notres=False):
    """
    **arguments:**
    
    * my_params : An lmfit Parameters object containing:
        - "Sep" separation in [mas]
        - "PA" position angle in [deg]
        - "Radius" The radius in [R_sun]
        - "Temperature" The temperature [K]
    * target_signal : The observed signal to fit
    * asim : The simulator object
    * dit : The detector integration time
    * K : The matrix that transforms the outputs into observables
      (single row for double bracewell)
    * postproc : An array of whitening matrices for each chunk (n_chunk, n_wl x n_k)
    * notres : If ``True`` this will return the non-whitened model signal
      if ``False`` this will return the difference between the whitened model signal
      and the whitened target.
    """
    
    theobs = copy.deepcopy(asim.obs)
    interest = make_source(my_params, asim.lambda_science_range, asim.src.distance)
    #theobs = copy.deepcopy(asim.obs)
    combined = make_th_exps(asim, dit, interest, diffuse, obs=theobs)
    #if not notres:
    #print(K.shape)
    #if len(K.shape)>2:
    #set_trace()
    diff = np.einsum("k o, n w o -> n w k", K, combined)
    
    #print("postproc", postproc.shape)
    #print("diff", diff.shape)
    wsig = np.einsum("ijk, ik -> ij", postproc, rearrange(diff, "a b c -> a (b c)"))
    # Not very efficient
    #set_trace()
    
    if notres:
        return diff
    else:
        wtarg = np.einsum("ijk, ik -> ij", postproc, rearrange(target_signal, "a b c -> a (b c)"))
        resvec = wsig - wtarg
        return resvec

def make_th_exps(asim, dit, interest, diffuse, obs=None):
    """
    Creates a model observation of an idealized source of interest.
    
    Simulator/obs are required to account for array projection, pointing
    and combination scheme.
    
    **Arguments:**
    
    * asim : Simulator object
    * dit : Detector integration time
    * interest : Synthetic source of interest
    * diffuse : The diffuse light chain, used
      to model absorption of instrument/sky
    * obs : A given observatory object (default: None)
    
    **Returns:** (n_chunk, n_wl, n_out) array recorded in the integration time.
    """
    if obs is None:
        obs = copy.deepcopy(asim.obs)
    lights = []
    for i, time in enumerate(asim.sequence):
        obs.point(asim.sequence[i], asim.target)
        injected = asim.injector.best_injection(asim.lambda_science_range)
        injected = injected.T * asim.corrector.get_phasor(asim.lambda_science_range)
        if obs.observatory_location == 'space':
            array = obs.get_projected_array()
            
        else:
            array = obs.get_projected_array(obs.altaz, PA=obs.PA)
        filtered_starlight = diffuse[0].get_downstream_transmission(asim.lambda_science_range)
        collected = filtered_starlight * asim.injector.collecting * dit
        #set_trace()
        lights.append(asim.combine_light(interest, injected, array, collected))
    lights = np.array(lights)
    return lights
