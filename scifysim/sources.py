
import numpy as np
import sympy as sp
from sympy.functions.elementary.piecewise import Piecewise
from kernuller import mas2rad, rad2mas
from . import utilities
from astropy import constants
from astropy import units
import scipy.interpolate as interp
from pathlib import Path

parent = Path(__file__).parent.absolute()

class transmission_emission(object):
    def __init__(self,trans_file="data/MK_trans_sfs.txt", T=285,
                 airmass=False, observatory=None, name="noname"):
        """
        Reproduces a basic behaviour in emission-transmission of a medium in the optical
        path. Inspired by the geniesim 
        
        trans_file    : The path to a transmission data file
        T             : Temperature of the medium used for emission
        airmass       : When True, scales the effect with the airmass provided by the observatory object
        observatory   : The observatory object used to provided the airmass.
        
        WARNING: Unlike for astrophysical sources, when a transmission_emission
        object contains an ss attribute, it already
        takes into account the transmission through the insturment.
        """
        self.__name__ = name
        if isinstance(trans_file, float):
            # Flat value from 1 nm to 50µm
            self.trans_file = np.array([[1.0e-9, trans_file],
                                       [50.0e-6, trans_file]])
        elif isinstance(trans_file, str):
            self.trans_file = np.loadtxt(parent/trans_file)
        else:
            raise ValueError("Requires a float or a file path")
            
        self.trans = interp.interp1d(self.trans_file[:,0], self.trans_file[:,1],
                                     kind="linear", bounds_error=False )
        self.T = T
        self.airmass = airmass
        if self.airmass:
            self.obs = observatory
        
        
    def get_trans_emit(self,wl, bright=False, no=False):
        """
        Return the transmission or brightness of the object depending on th bright keyword
        wl            : The wavelength channels to compute [m]
        bright        : If True: compute the brightness
                        if False: compute the transmission
        """
        if isinstance(wl, np.ndarray):
            if bright:
                result = np.ones_like(wl)
            else:
                result = np.zeros_like(wl)
        else:
            if bright:
                result = 1.
            else:
                result = 0.
        transmission = self.trans(wl)
        if self.airmass :
            sky = transmission**self.obs.altaz.secz.value
        else :
            sky = transmission
        
        # Determining brightness:
        import pdb
        if bright:
            #pdb.set_trace()
            sky = (1-sky) * blackbody.get_B_lamb_ph(wl, self.T) * np.gradient(wl[0,:])
            
        # Add offset of 3.25e-3 Jy/as² to account fot OH emission lines (see Vanzi & Hainaut)
        #if bright:
        #    sky[wl<2.5e-6] = sky[wl<2.5e-6] + 3.25e-3*units.sr.to(units.arcsec**2)
        if np.any(wl<2.5e-6):
            raise NotImplementedError("Must account for OH emisison lines")
        
        return sky
        
        
    def get_mean_trans_emit(self, wl, bandwidth=None, bright=False,n_sub=10):
        """
        Similar begaviour as get_trans_emit() but averages the effect over each spectral channel.
        wl            : The center of each wavelength channel. If bandwidth is not provided
                        (prefered situation) the width of each channel will be set as the spacing
                        between them (through np.gradient())
        bandwidth     : (deprecated) array of floats providing the width of each spectral channe.
        n_sub         : The number of sub-channel to compute for the calculation. A minimum of 10 is recommended.
                        
        """
        #
        # Little shortcut when bandwidth is not provided: infer it based
        # on the band covered by wl
        if bandwidth is None:
            if isinstance(wl, np.ndarray):
                cen_wl = wl.copy()
                bandwidth = np.gradient(wl)
            else:
                bandwidth = np.max(wl) - np.min(wl)
                cen_wl = np.mean(wl)
        else:
            cen_wl = wl
        
        lambda_min = wl - bandwidth/2
        lambda_max = wl + bandwidth/2
        os_wl = np.linspace(lambda_min, lambda_max, n_sub)
        sky = self.get_trans_emit(os_wl, bright=bright)
        mean_sky = np.mean(sky, axis=0)
        return mean_sky
    
    def get_own_brightness(self, wl):
        """
        Used when chaining different media.
        Returns the flux density per solid angle of the object in each wl channel. [ph/m2/sr]
        """
        result = self.get_mean_trans_emit(wl, bright=True)
        return result

    def get_upstream_brightness(self, wl):
        """
        Used when chaining different media.
        Returns the flux density per solid angle of the object in each wl channel [ph/m2/sr],
        upstream of the current object (exclusively),
        including the transmission by the upstream objects (exclusively)
        """
        if self.upstream is None:
            result = np.zeros_like(wl)
        else :
            result = self.upstream.get_total_brightness(wl)
        return result

    def get_total_brightness(self, wl):
        """
        Used when chaining different media.
        Returns the total filtered
        flux density per solid angle of the
        object in each wl channel. [ph/m2/sr] downstream of the object (inclusively)
        """
        result = self.get_own_brightness(wl) + self.get_own_transmission(wl) * self.get_upstream_brightness(wl)
        return result

    def get_own_transmission(self, wl):
        """
        Used when chaining different media.
        Returns the transmission function of the object.
        """
        result = self.get_mean_trans_emit(wl)
        return result

    def get_downstream_transmission(self, wl, inclusive=True):
        """
        Used when chaining different media.
        Returns the total of the transmission function of the chain downstream of the object.
        wl   : The wavelengths to be computed
        inclusive: Whether to includ the transmission of the object itself
        """
        if inclusive:
            own_trans = self.get_own_transmission(wl)
        else :
            # For exclusive transmission, the object's own transmission is set to 1
            own_trans = 1.
        if self.downstream is None:
            result = own_trans
        else:
            result = own_trans * self.downstream.get_downstream_transmission(wl,inclusive=True)
        return result
#sky_link = transmission_emission()

def chain(A, B):
    """
    This function helps define which occulting / emitting source
    is in front ro behind which. 
    A       : An upstream transmission_emission object
    B       : A downstream transmission_emission object.
    """
    A.downstream = B
    B.upstream = A
def set_source(A):
    """
    Defines A as the first element in the chain.
    """
    A.upstream = None
def set_sink(A):
    """
    Defines A as the last element in the chain.
    """
    A.downstream = None


class _blackbody(object):
    def __init__(self, modules="numexpr"):
        """
        Builds the spectral radiance as a function of T and \nu, k, and \lambda
        Call self.get_Bxxx to get the function B = f(lambda , T)
        
           GENIEsim  0         B_lamb_Jy      - result in Jy / sr
                     1         B_lamb_W       -           W / m^2 / m / sr
                     2         B_lamb_ph      -           ph / s / m^2 / m / sr
                     3         B_nu_ph        -           ph / s / m^2 / Hz / sr
        """
        
        self.h, self.nu, self.c, self.k_b, self.T = sp.symbols("h, nu, c, k_b, T", real=True)
        self.lamb = sp.symbols("lambda", real=True)
        self.k_W2Jy = sp.symbols("k_W2Jy", real=True) #The coefficient to convert from W to Jansky
        
        self.thesubs = [(self.c, constants.c.value),# speed of light
                       (self.k_b, constants.k_B.value),# Boltzmann constant (J.K^-1)
                       (self.h, constants.h.value),# Planck constant (J.s)
                       (self.k_W2Jy, 1e26)
                       ]
        
        # For Jy / sr
        self.B_lamb_Jy = self.k_W2Jy*2*self.h*self.c / self.lamb**3 * \
                    1/(sp.exp(self.h*self.c/(self.lamb*self.k_b*self.T)) -1)
        self.get_B_lamb_Jy = utilities.ee(self.B_lamb_Jy.subs(self.thesubs))
        self.get_B_lamb_Jy.lambdify((self.lamb, self.T), modules=modules)
        self.get_B_lamb_Jy.__doc__ = """f(wavelength[m], T[K]) Computes the Planck law in [Jy / sr]"""
        
        # For  W / m^2 / m / sr
        self.B_lamb_W = 2*self.h*self.c**2 / self.lamb**5 * \
                    1/(sp.exp(self.h*self.c/(self.lamb*self.k_b*self.T)) -1)
        self.get_B_lamb_W = utilities.ee(self.B_lamb_W.subs(self.thesubs))
        self.get_B_lamb_W.lambdify((self.lamb, self.T), modules=modules)
        self.get_B_lamb_W.__doc__ = """f(wavelength[m], T[K]) Computes the Planck law in [W / m^2 / m / sr]"""
        
        # For ph / s / m^2 / m / sr
        self.B_lamb_ph = 2*self.c / self.lamb**4 * \
                    1/(sp.exp(self.h*self.c/(self.lamb*self.k_b*self.T)) -1)
        self.get_B_lamb_ph = utilities.ee(self.B_lamb_ph.subs(self.thesubs))
        self.get_B_lamb_ph.lambdify((self.lamb, self.T), modules=modules)
        self.get_B_lamb_ph.__doc__ = """f(wavelength[m], T[K]) Computes the Planck law in [ph / s / m^2 / m / sr]"""
        
        # For ph / s / m^2 / Hz / sr
        self.B_nu_ph = 2/self.lamb**2 / self.lamb**4 * \
                    1/(sp.exp(self.h*self.c/(self.lamb*self.k_b*self.T)) -1)
        self.get_B_nu_ph = utilities.ee(self.B_nu_ph.subs(self.thesubs))
        self.get_B_nu_ph.lambdify((self.lamb, self.T), modules=modules)
        self.get_B_nu_ph.__doc__ = """f(wavelength[m], T[K]) Computes the Planck law in [ph / s / m^2 / Hz / sr]
        Nota: wavelength is still input in [m]"""
        
        
    def Stefan_Boltzmann(self, T, epsilon=1.):
        """
        Quick utility function giving the Stefan-Boltzmann law
        """
        sigma = constants.sigma_sb.value
        M = sigma * epsilon * T**4
        return M
        
        
        
blackbody = _blackbody()

def distant_blackbody(lambda_range, T, dist, radius):
    """
    Returns the flux density for a distant blackbody
    T      : temperature [K]
    dist   : Distance [pc]
    
    """
    dlambda = np.gradient(lambda_range)
    flux_density = blackbody.get_B_lamb_ph(lambda_range, T)*dlambda \
                    * np.pi * ((radius * constants.R_sun) / (dist * constants.pc))**2
    return flux_density

class group(object):
    """
    Just a simple object to structure some data.
    """
    def __init__(self):
        pass

class star_planet_target(object):
    def __init__(self, config, director):
        self.config = config
        self.T_star = self.config.getfloat("target", "star_temperature")
        self.R_star = self.config.getfloat("target", "star_radius")
        self.resolved_star = self.config.getboolean("target", "star_resolved")
        self.T_planet = self.config.getfloat("target", "planet_temperature")
        self.R_planet = self.config.getfloat("target", "planet_radius") * units.Rjup.to(units.Rsun)
        self.distance = self.config.getfloat("target", "star_distance")
        self.planet_separation = self.config.getfloat("target", "planet_sep")
        self.planet_position_angle = self.config.getfloat("target", "planet_pa")
        self.planet_offsetx = -self.planet_separation*np.sin(self.planet_position_angle * np.pi/180)
        self.planet_offsety =  self.planet_separation*np.cos(self.planet_position_angle * np.pi/180)
        self.planet_offset = (self.planet_offsetx, self.planet_offsety)
        print("sep = ", self.planet_separation)
        print("pa = ", self.planet_position_angle)
        print("offset = ", self.planet_offset)

        # Building the transmission chain
        self.t_sky = self.config.getfloat("atmo", "t_sky")
        self.t_vlti = self.config.getfloat("vlti", "T_vlti")
        
        
        # Creating absorbtion / emission chain:
        self.sky = transmission_emission(trans_file="data/MK_trans_sfs.txt",
                                         T=self.t_sky, airmass=True,
                                         observatory=director.obs,
                                         name="Sky")
        self.UT = transmission_emission(trans_file="data/VLTI_UT_trans_sfs.txt", T=self.t_vlti,
                                       name="UT_optics")
        
        n_warm_optics = config.getfloat("optics", "n_warm_optics")
        throughput_warm_optics = config.getfloat("optics", "throughput_warm_optics")
        self.transmission_warm_optics = throughput_warm_optics**n_warm_optics
        self.temp_warm_optics = config.getfloat("optics", "temp_warm_optics")
        n_cold_optics = config.getfloat("optics", "n_cold_optics")
        throughput_cold_optics = config.getfloat("optics", "throughput_cold_optics")
        self.temp_cold_optics = config.getfloat("optics", "temp_cold_optics")
        self.temp_combiner = config.getfloat("optics", "temp_combiner")
        self.throughput_combiner_chip = config.getfloat("optics", "throughput_combiner_chip")
        self.throughput_disp_elmnt = config.getfloat("optics", "throughput_disp_elmnt")
        self.transmission_cold_optics = self.throughput_disp_elmnt\
                                        * throughput_cold_optics**n_cold_optics
        
        self.warm_optics = transmission_emission(trans_file=self.transmission_warm_optics,
                                                T=self.temp_warm_optics, name="Warm Optics")
        self.combiner = transmission_emission(trans_file=self.throughput_combiner_chip,
                                                T=self.temp_combiner, name="Combiner")
        self.cold_optics = transmission_emission(trans_file=self.transmission_cold_optics,
                                                T=self.temp_cold_optics, name="Cold Optics")
        
        chain(self.sky, self.UT)
        chain(self.UT, self.warm_optics)
        chain(self.warm_optics, self.combiner)
        chain(self.combiner, self.cold_optics)
        self.cold_optics.downstream = None
        self.sky.upstream = None
        
        # Create the star and planet source objects
        self.star = resolved_source(director.lambda_science_range,
                                           distance=self.distance, radius=self.R_star, T=self.T_star,
                                           angular_res=12, radial_res=5,
                                           resolved=self.resolved_star)
        self.planet = resolved_source(director.lambda_science_range,
                                             distance=self.distance, radius=self.R_planet, T=self.T_planet,
                                             resolved=False, offset=self.planet_offset)
    @property
    def physical_separation(self):
        """The physical separation between the planet and star (AU)"""
        psep = self.planet_separation*units.mas.to(units.rad)*self.distance*units.pc.to(units.AU)
        return psep
        


class resolved_source(object):
    def __init__(self, lambda_range, distance, radius, T,
                 angular_res=10, radial_res=15, offset=(0.,0.),
                 build_map=True, resolved=True):
        """
        distance             : Distance of the source [pc]
        radius             : The radius of the source [R_sun]
        T                    : Blackbody temperature [K]
        angular_res          : Number of bins in position angle
        radial_res           : Number of bins in radius
        offset               : Offset of the source radial ([mas], [deg])
        build_map            : Whether to precompute a mapped spectrum 
        resolved             : If false, computes a single point-source
        
        
        After building the map, self.ss (wl, pos_x, pos_y) contains the map
        of flux density corresponding to positions self.xx, self.yy
        """
        self.lambda_range = lambda_range
        self.T = T
        self.distance = distance
        self.radius = radius
        self.offset = offset
        # self.ang_radius is in radians
        self.ang_radius = self.radius / (self.distance*units.pc.to(units.R_sun))
        total_solid_angle = np.pi * self.ang_radius**2 # disk section [sr]
        self.total_flux_density = self.distant_blackbody()/ total_solid_angle # [ph / s / m^2 / sr]
        
        if resolved:
            self.build_grid(angular_res, radial_res)
        else: 
            self.build_point()
        if build_map:
            self.build_spectrum_map()
            
            
    def build_point(self,):
        """
        Routine used to construct unresolved source:
        Creates self.xx, self.yy, self.ds, self.theta [rad], self.r [rad]
        Shapes of xx, yy are preserved, but they will contain a single element
        corresponding to the unresolved point source.
        
        """
        
        self.theta, self.r = np.array([[0.]]), np.array([[0.]])
        # Angular positions referenced East of North
        self.xx = -self.r*np.sin(self.theta)*units.rad.to(units.mas) \
                    - self.offset[0]
        self.yy =  self.r*np.cos(self.theta)*units.rad.to(units.mas) \
                    + self.offset[1]
        
        self.ds = np.array([[np.pi * self.ang_radius**2]])
        
    def build_grid(self, angular_res, radial_res):
        """
        Routine used to construct resolved source:
        Creates self.xx, self.yy, self.ds, self.theta, self.r
        
        """
        radial_step = self.ang_radius/radial_res
        self.theta, self.r = np.meshgrid( np.linspace(0., 2*np.pi, angular_res, endpoint=False), np.linspace(0.+radial_step/2, self.ang_radius-radial_step/2, radial_res, endpoint=True) )
        # Angular positions referenced East of North
        self.xx = -self.r*np.sin(self.theta)*units.rad.to(units.mas) \
                    - self.offset[0]
        self.yy =  self.r*np.cos(self.theta)*units.rad.to(units.mas) \
                    + self.offset[1]
        
        self.dr = np.gradient(self.r)[0]
        self.dtheta = np.gradient(self.theta)[1]
        self.ds = self.r*self.dr*self.dtheta
        
    def get_spectrum_map(self):
        """
        Maps self.total_flux_density
        This produces numerical integration over solid angle elements
        """
        themap =  self.total_flux_density[:,None,None]*self.ds[None,:,:,] #self.r[None,:,:]*self.dr[None,:,:]*self.dtheta[None,:,:]
        return themap
    
    def build_spectrum_map(self):
        """
        The map is saved in a flat shape
        xx_f and yy_f are created to be flat versions of the coordinates.
        ss is a total flux (ph/s/m^2) at the entrance of earth atmosphere.
        """
        self.ss = self.get_spectrum_map().value # Fixing a bug that appears in the spectrograph?
        self.ss = self.ss.reshape(self.ss.shape[0], self.ss.shape[1]*self.ss.shape[2])
        self.xx_f = self.xx.flatten()
        self.yy_f = self.yy.flatten()
        self.xx_r = mas2rad(self.xx_f)
        self.yy_r = mas2rad(self.yy_f)
        
    def distant_blackbody(self):
        """
        Returns the flux density for a distant blackbody
        It is a total in ph/s/m2
        """
        dlambda = np.gradient(self.lambda_range)
        #Discretization by spectral bins
        flux_density = blackbody.get_B_lamb_ph(self.lambda_range, self.T)*dlambda \
                        * np.pi * ((self.radius * constants.R_sun) / (self.distance * constants.pc))**2#That gives the scaling of the flux density by the distance
        return flux_density
        


class source(object):
    """
    DEPRECATED: use resolved_source for all source modelisation purposes
    """
    def __init__(self, xx, yy, ss):
        """
        DEPRECATED: use resolved_source for all source modelisation purposes
        """
        self.xx = xx
        self.yy = yy
        self.ss = ss
    def __add__(self,other):
        xx = np.concatenate((self.xx, other.xx))
        yy = np.concatenate((self.yy, other.yy))
        ss = np.concatenate((self.ss, other.ss), axis=1)
        return source(xx, yy, ss)
    def copy(self):
        return source(self.xx.copy(), self.yy.copy(), self.ss.copy())
    
    @classmethod
    def sky_bg(cls, injector, res,  T, lamb_range, crop=1.):
        """
        DEPRECATED: a transmission_emission object for the sky background
        """
        # First build a grid of coordinates
        hskyextent = rad2mas(injector.focal_hrange/injector.focal_length)
        hskyextent = hskyextent*crop
        resol = res #injector.focal_res
        xx, yy = np.meshgrid(
                            np.linspace(-hskyextent, hskyextent, resol),
                            np.linspace(-hskyextent, hskyextent, resol))
        xx = xx.flatten()
        yy = yy.flatten()
        src = cls(xx, yy,
                  np.ones_like(lamb_range)[:,None]*np.ones_like(xx)[None,:])
        src.rr = np.sqrt(src.xx**2 + src.yy**2)
        thebb = blackbody()
        spectrum = thebb.Boflamb(lamb_range, T)
        src.ss = src.ss * spectrum[:,None]
        if injector is not None:
            
            src.mask = injector.LP01.numerical_evaluation(injector.focal_hrange*crop, resol, lamb_range)
            src.mask = src.mask.reshape((src.mask.shape[0], src.mask.shape[1]*src.mask.shape[2]))
            src.ss = src.ss * src.mask
        return src
        
        
        
class src_extended(object):
    def __init__(self, resol, extent, fiber_vigneting=False):
        """
        
        resol              : Number of elements across
        extent             : The extent of the source
        fiber_vigneting    : whether to include fiber vigneting in the luminosity distribution
                            Fiber vigneting should not be included when off-axis injection is simulated
        """
        self.xx, self.yy = np.meshgrid(
                            np.linspace(-extent/2, exten/2, resol),
                            np.linspace(-extent/2, exten/2, resol))
        self.rr = np.sqrt(self.xx**2 + self.yy**2)
        
        
    def uniform_disk(self, radius):
        self.ss = np.zeros_like(self.rr)
        self.ss[self.rr<radus] = 1.
        
        #self.f = Piecewise((1, self.r<=radius),
        #                   (0, self.r>radius))

        
