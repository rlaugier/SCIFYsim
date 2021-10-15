import sympy as sp
import numpy as np
import scifysim as sf
import logging
from einops import rearrange

from scipy.linalg import sqrtm

import matplotlib.pyplot as plt

from scipy.stats import chi2, ncx2, norm
from scipy.optimize import leastsq

logit = logging.getLogger(__name__)


class noiseprofile(object):
    def __init__(self, integ, asim, diffobs_series, verbose=False, n_pixsplit=1):
        """
        An object to dynamically compute a parametric noise floor
        
        Patameters:
        -----------
        * integ     : An integrator object augmented by some mean
                        and static measurements (see below)
        * asim      : A simulator object (mostly used to get some combiner
                        and injector parameters)
        * diffobs   : A series of realisations of te differential observable
                        intensity (e.g. `diffobs = np.einsum("ij, mkj->mk",asim.combiner.K, dit_intensity)`)
        
        
        Creation makes use of:
        ----------------------
        * integ.mean_starlight (run `integ.mean_starlight = np.mean(starlights, axis=0)`)
        * integ.mean_planetlight (run `integ.mean_planetlight = np.mean(planetlights, axis=0)`)
        * integ.get_static() (run `integ.static = asim.computed_static`)
        
        """
        # Read noise
        self.mynpix = n_pixsplit
        self.eta = integ.eta
        self.sigma_ron = np.sqrt(self.mynpix) * integ.ron
        self.sigma_ron_d = np.sqrt(2) * self.sigma_ron
        self.mask_dark = asim.combiner.dark
        self.mask_bright = asim.combiner.bright
        self.mask_phot = asim.combiner.photometric
        

        # Photon noise
        self.dit_0 = asim.injector.screen[0].step_time * integ.n_subexps
        self.m_0, self.mp0 = asim.context.get_mags_of_sim(asim)
        self.F_0 = mag2F(self.m_0)
        self.s_0 = integ.mean_starlight / self.dit_0
        self.s_d = integ.get_static() / self.dit_0
        
        self.p_0 = integ.mean_planetlight / self.dit_0
        
        
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
        WARNING: if matrix is True, then sigma_phi_d is
        a covariance matrix (beware of variances on diagonal)
        
        Parameters:
        -----------
        
        * m_star       : Magnitude of the star
        * dit          : Detector integration time [s]
        
        Returns:
        --------
        
        `sigma_phot_d, sigma_phi_d` : standard deviation vectors (or
        covariance matrix for sigma_phi_d)
        """
        Fs = mag2F(m_star)
        
        # small s is a flux
        s_s = Fs/self.F_0 * self.s_0
        s_d = self.s_d
        
        
        sigma_phot = np.sqrt(self.eta * dit * (s_d + s_s))
        sigma_phot_d = np.sqrt(np.sum(sigma_phot[:,self.mask_dark]**2, axis=-1))
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
        
        Parameters:
        -----------
        
        * m_star       : Magnitude of the star
        * dit          : Detector integration time [s]
        
        Returns:
        --------
        
        Either `sigma_tot_diff` or `cov_tot_diff` depending on
        `matrix`.
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
                          ymin=0., ymax=1.):
        
        sigphis = []
        sigphots = []
        for amag in starmags:
            asigphot, asigphi = self.get_noises(amag, dit)
            sigphots.append(asigphot)
            sigphis.append(asigphi)
        sigphots = np.array(sigphots)
        sigphis = np.array(sigphis)
        
        
        mybmap = plt.matplotlib.cm.Blues
        mygmap = plt.matplotlib.cm.YlOrBr
        fig = plt.figure(dpi=200)
        plt.plot(wls, self.sigma_ron_d*np.ones_like(wls), label=r"$\sigma_{RON}$")
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
        plt.title("Noise in single %.2f s DIT"%(dit))
        if show:
            plt.show()
        return fig
    
    def summary(self):
        print("n_pix = ", self.mynpix, "# The number of pixels used per spectral channel")
        print("t_dit_0 = ", self.dit_0, "# The reference exposure time used in MC simulations")
        print("eta = ", self.eta, "# The quantum efficiency of the detector")
        print("sigma_ron_d = ", self.sigma_ron_d, "# The total readout noise per spectral channel for the differential observable(s)")
        print("k_phi = ", self.k_phi_m, "# The instrumental error coefficient")
        
        

class spectral_context(object):
    def __init__(self, vegafile="config/vega.ini"):
        """Spectral context for magnitude considerations
        
        Parameters:
        -----------
        
        vegafile    : The configuration file for observation of Vega
                        in the same spectral configuration
        """

        self.avega = sf.utilities.prepare_all(vegafile, update_params=False,
                            instrumental_errors=False)
        self.thevegassflux = self.avega.src.star.ss.sum(axis=1)

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
        
        Parameters:
        -----------
        
        asim     : A simulator object
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
    
def mag2F(mag):
    F = 1*10**(-mag/2.5)
    return F
def F2mag(F):
    mag = -2.5*np.log10(F/1)
    return mag


##################################
# The energy detector test
##################################

def pdet_e(lamb, xsi, rank):
    """
    Computes the residual of Pdet
    lamb     : The noncentrality parameter representing the feature
    xsi      : The location of threshold
    rank     : The rank of observable
    Returns the Pdet difference
    """
    respdet = 1 - ncx2.cdf(xsi,rank,lamb)
    return respdet
def residual_pdet_Te(lamb, xsi, rank, targ):
    """
    Computes the residual of Pdet
    
    Parameters:
    -----------
    
    * lamb     : The noncentrality parameter representing the feature
    * targ     : The target Pdet to converge to
    * xsi      : The location of threshold
    * rank     : The rank of observable
    
    Returns the Pdet difference. See Ceau et al. 2019 for more information
    """
    respdet = 1 - ncx2.cdf(xsi,rank,lamb) - targ
    return respdet
def get_sensitivity_Te(maps, pfa=0.002, pdet=0.9, postproc_mat=None, ref_mag=10.,
                verbose=True):
    """
    Magnitude map at which a companion is detectable for a given map and whitening matrix.
    The throughput for the signal of interest is given by the map at a reference magnitude.
    The effective noise floor is implicitly described by the whitening matrix.
    
    
    * maps       : Differential map for a given reference magnitude
    * pfa        : The false alarm rate used to determine the threshold
    * pdet       : The detection probability used to determine the threshold
    * postproc_mat : The matrix of the whitening transformation ($\Sigma^{-1/2}$)
    * ref_mag    : The reference mag for the given map
    * verbose    : Print some information along the way
    
     See Ceau et al. 2019 for more information
    
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
    Parameters:
    ----------
    
    * xTx     : The noncentrality parameter representing the feature
    * targ     : The target Pdet to converge to
    * xsi      : The location of threshold
    
    Returns the Pdet difference.
    
    """
    
    Pdet_Pfa = 1 - norm.cdf((xsi - xTx)/np.sqrt(xTx))
    return Pdet_Pfa - targ
def get_sensitivity_Tnp(maps, pfa=0.002, pdet=0.9, postproc_mat=None, ref_mag=10.,
                verbose=False, use_tqdm=True):
    """
    Magnitude map at which a companion is detectable for a given map and whitening matrix.
    The throughput for the signal of interest is given by the map at a reference magnitude.
    The effective noise floor is implicitly described by the whitening matrix.
    
    
    * maps       : Differential map for a given reference magnitude
    * pfa        : The false alarm rate used to determine the threshold
    * pdet       : The detection probability used to determine the threshold
    * postproc_mat : The matrix of the whitening transformation ($\Sigma^{-1/2}$)
    * ref_mag    : The reference mag for the given map
    * verbose    : Print some information along the way
    
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
 

def correlation_map(signal, maps, postproc=None, K=None, n_diffobs=1, verbose=False):
    """
    Returns the raw correlation map of a signal with a map.
    
    Parameters:
    -----------
    
    signal      : The signal measured on sky shape:
                    (n_slots, n_wl, n_outputs)
    maps        : The maps for comparison shape:
                    (n_slots, n_wl, n_outputs, x_resolution, y_resolution)
    """
    assert (len(signal.shape) == 3), "Input shape (slots, wavelengths, outputs)"
    assert (len(maps.shape) == 5), "Maps shape (slots, wavelengths, outputs, position, position)"
    
    if signal.shape[2] != n_diffobs:
        diffobs = np.einsum("ij, mkj->mki", K, signal)
    elif signal.shape[2] == n_diffobs:
        diffobs = signal
    
    if maps.shape[2] != n_diffobs:
        difmap = np.einsum("ij, kljmn -> klimn", K, maps)
    elif maps.shape[2] == n_diffobs:
        difmap = maps
    wmap = np.einsum("ijk, iklm -> ijlm", postproc, rearrange(difmap, "a b c d e -> a (b c) d e"))
    wsig = np.einsum("ijk, ik -> ij", postproc, rearrange(diffobs, "a b c -> a (b c)"))
    print(wmap.shape)
    print(wsig.shape)
    
    cmap1 = np.einsum("ijkl, ij -> kl", wmap, wsig)
    #cmap2 = np.einsum("ikl, i -> kl", rearrange(wmap, "a b c d -> (a b) c d"), wsig.flatten())
    
    return cmap1
