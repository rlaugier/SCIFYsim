# injection.py

"""
This module revolves around the injector class
At ``__init__()``, injector builds

* one atmo object for each telescope (stored in a list)
* one focuser object (badly named) for each wavelength AND each telescope
* one fiber_head object that builds the map of LP01 mode.

It then creates a generator called ``self.it`` that will return
a series of complex numbers corresponding to injection complex phasors.

**Testing:**

- ``test_fiber`` for the ``fiber_head``
- ``test_injector`` for the ``injector``

"""



import numpy as np
import threading
import time
from pathlib import Path
from scipy.interpolate import interp2d, interp1d
from xaosim import zernike



from pdb import set_trace


# amto_screen is now provided in house to provide seed
#from xaosim import wavefront as wft

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2


import logging

logit = logging.getLogger(__name__)

from . import utilities

parent = Path(__file__).parent.absolute()


class atmo(object):
    '''Atmospheric Kolmogorov-type phase screen.
    
    **Class Attributes:**
    
    
    - csz     : size (csz x csz) of the phase screen       (in pixels)
    - pdiam   : diameter of the aperture within this array (in pixels)
    - rndarr  : uniformly distributed random array         (csz x csz)
    - kolm    : the original phase screen                  (csz x csz)
    - kolm2   : the oversized phase screen    ((csz + pdiam) x (csz + pdiam))
    - qstatic : an optional quasi static aberration    (pdiam x pdiam)
    - rms     : total phase screen rms value           (in nanometers)
    - rms_i   : instant rms inside the pupil           (in nanometers)

    **Comment:**
    
    While the attributes are documented here for reference, the prefered
    way of interacting with them is via the functions defined within the
    class.

    '''
    # ==================================================
    def __init__(self, name="GRAVITY+", csz = 512,
                 psz=200,
                 lsz=8.0, r0=0.2,
                 ro_wl=3.5e-6, L0=10.0,
                 fc=24.5, correc=1.0, seed=None,
                 wind_speed = 1., 
                 wind_angle = 0.1,
                 step_time = 0.01,
                 lo_excess = 0.,
                 pdiam=8.0, config=None):

        ''' Kolmogorov type atmosphere + qstatic error

        **Parameters:**
        
        - name  : a string describing the instrument
        - csz   : the size of the Fourier array
        - psz   : the size of the pupil in  pixels
        - lsz   : the screen linear size (in meters)
        - r0    : the Fried parameter (in meters)
        - L0    : the outer scale parameter (in meters)
        - fc    : the cutoff frequency [lambda/D] defined by the 
          number of actuators
        - correc : the correction factor to apply to controlled
          spatial frequencies.
        - wind_speed : the speed of the phas screen in [m/s]
        - step_time : the time resolution of the simulation [s]
        - config : a parsed config file 
        -----------------------------------------------------
        '''

        if config is None:
            # Deprecated the use of config is preferred
            self.csz     = csz
            self.psz     = psz
            self.lsz     = lsz
            self.pdiam   = pdiam
            self.ppscale = self.pdiam/self.psz
            self.wind_speed = wind_speed
            self.step_time = step_time
            self.r0      = r0
            self.L0      = L0
            self.r0_wl   = r0_wl

            self.rms_i   = 0.0
            self.correc  = correc
            self.lo_excess = lo_excess
            self.fc      = fc
        else :
            self.config = config
            pres = self.config.getint("atmo", "pup_res")
            screen_oversize = self.config.getfloat("atmo", "screen_size")
            pdiams = self.config.getarray("configuration","diam")
            pdiam = pdiams[0]
            fc = self.config.getfloat("atmo", "fc_ao")
            r0 = self.config.getfloat("atmo", "r0")
            L0 = self.config.getfloat("atmo", "Lout")
            correc = self.config.getfloat("atmo", "correc")
            lo_excess = self.config.getfloat("atmo", "lo_excess")
            wind_speed = self.config.getfloat("atmo", "vwind")
            step_time = self.config.getfloat("atmo", "step_time")
            wl_mean = self.config.getfloat("photon", "lambda_cen")
            
            self.csz     = int(pres*screen_oversize)
            self.psz     = pres
            self.pdiam   = pdiam
            self.ppscale = self.pdiam/self.psz
            self.wind_speed = wind_speed
            self.step_time = step_time
            self.lsz     = self.pdiam*screen_oversize
            self.r0      = r0
            self.L0      = L0
            self.r0_wl   = wl_mean

            self.rms_i   = 0.0
            self.correc  = correc
            self.lo_excess = lo_excess
            self.fc      = fc
            
        kolm, self.modul    = atmo_screen(screen_dimension=self.csz, screen_extent=self.lsz,
                                          r0=self.r0, L0=self.L0,
                                          fc=self.fc, correc=self.correc,
                                          lo_excess=self.lo_excess,
                                          pdiam=self.pdiam, seed=seed)
        self.kolm = kolm.real
        
        self.qstatic = np.zeros((self.psz, self.psz))
        self.shm_phs = self.qstatic

        self.kolm2   = np.tile(self.kolm, (2,2))
        #self.kolm2   = self.kolm2[:self.sz+self.pdiam,:self.sz+self.pdiam]


        self.offx = 0. # x-offset on the "large" phase screen array
        self.offy = 0. # y-offset on the "large" phase screen array
        self.t = 0.
        self.update_step_vector(wind_angle=wind_angle)
        self.ttc     = False   # Tip-tilt correction flag
        
        # auxilliary array (for tip-tilt correction)
        self.xx, self.yy  = np.meshgrid(np.arange(self.csz)-self.csz//2,
                                        np.arange(self.csz)-self.csz//2)
        self.xxnorm2 = np.sum(self.xx**2)
        self.yynorm2 = np.sum(self.yy**2)
        
        self.it = self.give()
        
        # Must update screen to obtain the correct scaling and correction
        self.update_screen()
    

    # ==============================================================
    
    def update_step_vector(self,wind_angle=0.1, wind_speed=None, step_time=None):
        """
        ----
        Refresh the step vector that will be applied
        
        **Parameters:**
        
        - wind_angle : the angle of incience of the wind (values around 0.1 rad
          are favoured since they provide long series before repeating)
        - wind_speed : The speed of the moving phase screen.
        - step_time  : The time step of the simulation [s]
        
        ----
        """
        if wind_speed is not None:
            self.wind_speed = wind_speed
        if step_time is not None:
            self.step_time = step_time
        shift_length = self.wind_speed*self.step_time/self.ppscale
        xstep, ystep = shift_length*np.cos(wind_angle), shift_length*np.sin(wind_angle)
        self.step_vector = np.array((ystep, xstep))
        
        
        
    
    def set_qstatic(self, qstatic=None):
        """
        ----
        Defines some quasistatic errors in the wavefront
        
        **Parameters:**
        
        - qstatic    : A static phase screen
        
        ----
        """
        if qstatic is not None:
            if qstatic.shape == (self.csz, self.csz):
                self.qstatic = qstatic
                print("quasi-static phase screen updated!")
            else:
                print("could not update quasi-static phase screen")
                print("array should be %d x %d (dtype=%s)" % (
                    self.csz, self.csz, str(self.qstatic.dtype)))

        else:
            print("no new quasi-static screen was provided!")

    # ==============================================================
    def update_screen(self, correc=None, fc=None, r0=None, L0=None, seed=None):
        ''' ------------------------------------------------
        Generic update of the properties of the phase-screen
        
        **Parameters:**
        
        - correc : *float* The correction factor for the control region
        - fc : *float* The cutoff fequency
        - r0 : *flaot* r0 of the initial phase screen
        
        ------------------------------------------------ '''
        if r0 is not None:
            self.r0 = r0

        if L0 is not None:
            self.L0 = L0

        if correc is not None:
            self.correc = correc

        if fc is not None:
            self.fc = fc
        
        # Converting to a piston screen in microns at this point
        # r0 is now expected at the mean of the wl band
        logit.warning("Update the way the phase screen is converted to a piston screen")
        kolm, self.modul = atmo_screen(screen_dimension=self.csz, screen_extent=self.lsz,
                                          r0=self.r0, L0=self.L0,
                                          fc=self.fc, correc=self.correc,
                                          lo_excess=self.lo_excess,
                                          pdiam=self.pdiam, seed=seed)
        self.kolm    = 1.0e6 * self.r0_wl / (2*np.pi) * kolm.real  # converting to piston in microns

        self.kolm2   = np.tile(self.kolm, (2,2))


    def reset(self):
        self.offx = 0.
        self.offy = 0.
        
    def give(self):
        ''' ------------------------------------------
        Main loop: frozen screen slid over the aperture
        
        This returns a iterator that yelds the phase screen
        for one aperture.
        
        **use:**
        
        .. code-block::
        
            a = atmo.give()
            phscreen = next(a)
            phscreen = next(a)

        **Options:**
        
        -----------------------------------------  '''

        while True:
            self.offx += self.step_vector[1] #2
            self.offy += self.step_vector[0] #1
            self.t += self.step_time
            self.intoffx = int(np.round(self.offx % self.csz))
            self.intoffy = int(np.round(self.offy % self.csz))

            subk = self.kolm2[self.intoffx:self.intoffx+self.psz,
                              self.intoffy:self.intoffy+self.psz].copy()

            if self.ttc is True:
                ttx = np.sum(subk*self.xx) / self.xxnorm2
                tty = np.sum(subk*self.yy) / self.yynorm2
                subk -= ttx * self.xx + tty * self.yy

            self.rms_i = subk.std()
            self.shm_phs = subk + self.qstatic
            yield self.shm_phs.astype(np.float32)
            
# ==================================================================
def atmo_screen(screen_dimension, screen_extent,
                r0, L0,
                fc=25, correc=1.0, lo_excess=0.,
                pdiam=None,seed=None):
    ''' -----------------------------------------------------------
    
    The Kolmogorov - Von Karman phase screen generation algorithm.

    Adapted from the work of Carbillet & Riccardi (2010).
    `<http://cdsads.u-strasbg.fr/abs/2010ApOpt..49G..47C>`_

    Kolmogorov screen can be altered by an attenuation of the power
    by a correction factor *correc* up to a cut-off frequency *fc*
    expressed in number of cycles across the phase screen

    **Parameters:**

    - screen_dimension    : the size of the array to be computed (in pixels)
    - screen_extent     :  the physical extent of the phase screen (in meters)
    - r0     : the Fried parameter, measured at a given wavelength (in meters)
    - L0     : the outer scale parameter (in meters)
    - fc     : DM cutoff frequency (in lambda/D)
    - correc : correction of wavefront amplitude (factor 10, 100, ...)
    - lo_excess: A factor introducing excess low-order averations (mosly tip-tilt)
      Must be striclty 0 =< lo_excess < 1.
    - pdiam  : pupil diameter (in meters)
    - seed   : random seed for the screen (default: None produces a new seed)

    Returns: two independent phase screens, available in the real and 
    imaginary part of the returned array.

    **Remarks:**
    
    If pdiam is not specified, the code assumes that the diameter of
    the pupil is equal to the extent of the phase screen "screen_extent".
    
    ----------------------------------------------------------- '''
    
    #phs = 2*np.pi * (np.random.rand(screen_dimension, screen_dimension) - 0.5)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    phs = rng.uniform(low=-np.pi, high=np.pi, size=(screen_dimension,screen_dimension))

    xx, yy = np.meshgrid(np.arange(screen_dimension)-screen_dimension/2, np.arange(screen_dimension)-screen_dimension/2)
    rr = np.hypot(yy, xx)
    rr = shift(rr)
    rr[0,0] = 1.0

    modul = (rr**2 + (screen_extent/L0)**2)**(-11/12.)
    

    if pdiam is not None:
        in_fc = (rr < fc * screen_extent / pdiam)
    else:
        in_fc = (rr < fc)
        
    #if not np.isclose(lo_excess, 0):
    #    set_trace()
    
    modul[in_fc] /= correc * (1 - lo_excess * np.exp(- 0.5*rr[in_fc]))
    # obtaining unique rr values 
    rru, rru_indices = np.unique(rr, return_index=True)
    amodul = np.sqrt(2*0.0228)*(screen_extent/r0)**(5/6.)*modul.flat[rru_indices]
    
    screen = ifft(modul * np.exp(1j*phs)) * screen_dimension**2
    screen *= np.sqrt(2*0.0228)*(screen_extent/r0)**(5/6.)

    screen -= screen.mean()
    return screen, np.array([rru, amodul]).T

# ======================================================================

            
dtor = np.pi/180.0  # to convert degrees to radians
i2pi = 1j*2*np.pi   # complex phase factor






def Hn(A):
    return np.conjugate(np.transpose(A))


class focuser(object):
    ''' Generic monochoromatic camera class
    
    This class simulates focusing optics. It can simulate injection 
    either:
    
    - by computing the product of the image plane complex amplitude with
      the fiber mode field, or
    - by computing the product of the pupil plane complex amplitude with
      the conjugation of the fiber mode field.
    '''

    # =========================================================================
    def __init__(self, name="SCExAO_chuck", csz=200, ysz=256, xsz=256,
                 pupil=None,screen=None, rm_inj_piston=False,
                 pdiam=7.92, pscale=10.0, wl=1.6e-6):
        ''' Default instantiation of a cam object:

        -------------------------------------------------------------------
        Parameters are:
        ---------------
        - name    : a string describing the camera ("instrument + camera name")
        - csz     : array size for Fourier computations
        - (ys,xs) : the dimensions of the actually produced image
        - pupil   : a csz x csz array containing the pupil
        - pscale  : the plate scale of the image, in mas/pixel
        - wl      : the central wavelength of observation, in meters
        ------------------------------------------------------------------- '''

        self.name = name
        self.csz = csz              # Fourier computation size
        self.ysz = ysz              # camera vertical dimension
        self.xsz = xsz              # camera horizontal dimension
        self.isz = max(ysz, xsz)    # max image dimension

        # possible crop values (to match true camera image sizes)
        self.x0 = (self.isz - self.xsz) // 2
        self.y0 = (self.isz - self.ysz) // 2
        self.x1 = self.x0 + self.xsz
        self.y1 = self.y0 + self.ysz

        if pupil is None:
            self.pupil = ud(csz, csz, csz//2, True)
        else:
            self.pupil = pupil
        self.flatpup = self.pupil.flatten()
        self.pupilsum = np.sum(self.pupil).astype(np.float32)

        self.pdiam  = pdiam                 # pupil diameter in meters
        self.pscale = pscale                # plate scale in mas/pixel
        self.fov = self.isz * self.pscale
        self.wl     = wl                    # wavelength in meters
        self.mu2phase = np.float32(2.0 * np.pi / self.wl / 1e6)  # convert microns to phase
        self.frm0   = np.zeros((ysz, xsz))  # initial camera frame

        self.btwn_pixel = True            # fourier comp. centering option
        self.phot_noise = False            # photon noise flag
        self.signal     = np.float32(1e6)              # default # of photons in frame
        self.corono     = False            # if True: perfect coronagraph
        self.remove_injection_piston = rm_inj_piston
        self.npix = np.count_nonzero(self.pupil)
      
        # final tune-up
        self.update_cam()
        
        self.tip = self.pupil*zernike.mkzer(*zernike.noll_2_zern(2),
                                 self.pupil.shape[0], 100,
                                 limit=False)
        self.tip = np.pi*self.tip/np.max(self.tip)
        self.ntip = self.tip.flatten()/self.tip.flatten().dot(self.tip.flatten())
        self.tilt = self.pupil*zernike.mkzer(*zernike.noll_2_zern(3),
                                  self.pupil.shape[0], 100,
                                  limit=False)
        self.tilt = np.pi*self.tilt/np.max(self.tilt)
        self.ntilt = self.tilt.flatten()/self.tilt.flatten().dot(self.tilt.flatten())

    # =========================================================================
    def update_cam(self, wl=None, pscale=None, between_pixel=None):
        '''
        Change the filter, the plate scale or the centering of the camera

        **Parameters:**
        
        - pscale        : the plate scale of the image, in mas/pixel
        - wl            : the central wavelength of observation, in meters
        - between_pixel : whether FT are centered between four pixels or not
        '''
        wasgoing = False
            
        if wl is not None:
            self.wl = wl
            try:
                del self._A1
            except AttributeError:
                print("sft aux array to be refreshed")
                pass

        if pscale is not None:
            self.pscale = pscale
            try:
                del self._A1
            except AttributeError:
                print("SFT aux array to be refreshed")
                pass
        if between_pixel is not None:
            self.btwn_pixel = between_pixel
            try:
                del self._A1
            except AttributeError:
                print("SFT aux array to be refreshed")
                pass

        self.ld0 = self.wl/self.pdiam*3.6e6/dtor/self.pscale  # l/D (in pixels)
        self.nld0 = self.isz / self.ld0           # nb of l/D across the frame

        tmp = self.sft(np.zeros((self.csz, self.csz)))


    # =========================================================================
    def update_signal(self, nph=1e6):
        ''' Update the strength of the signal

        Automatically sets the *phot_noise* flag to *True*
        *IF* the value provided is negative, it sets the *phot_noise* flag
        back to *False* and sets the signal back to 1e6 photons

        **Parameters:**
        
        - nph: the total number of photons inside the frame
        '''
        if (nph > 0):
            self.signal = np.flaot32(nph)
            self.phot_noise = True
        else:
            self.signal = np.float32(1e6)
            self.phot_noise = False

    # =========================================================================
    def sft(self, A2):
        ''' Class specific implementation of the explicit Fourier Transform

        -------------------------------------------------------------------
        The algorithm is identical to the function in the sft module,
        except that intermediate FT arrays are stored for faster
        computation.

        For a more generic implementation, refer to the sft module of this
        package.

        Assumes the original array is square.
        No need to "center" the data on the origin.
        -------------------------------------------------------------- '''
        try:
            test = self._A1  # look for existence of auxilliary arrays
        except AttributeError:
            logit.info("updating the Fourier auxilliary arrays")
            NA = self.csz
            NB = self.isz
            m = self.nld0
            self._coeff = m/(NA*NB)

            U = np.zeros((1, NB))
            X = np.zeros((1, NA))

            offset = 0
            if self.btwn_pixel is True:
                offset = 0.5
            X[0, :] = (1./NA)*(np.arange(NA)-NA/2.0+offset)
            U[0, :] = (m/NB)*(np.arange(NB)-NB/2.0+offset)

            sign = -1.0

            self._A1 = np.exp(sign*i2pi*np.dot(np.transpose(U), X))
            self._A3 = np.exp(sign*i2pi*np.dot(np.transpose(X), U))
            self._A1 *= self._coeff

        B = (self._A1.dot(A2)).dot(self._A3)
        return np.array(B)

    # =========================================================================
    def getimage(self, phscreen=None):
        ''' Produces an image, given a certain number of phase screens,
        and updates the shared memory data structure that the camera
        instance is linked to with that image

        If you need something that returns the image, you have to use the
        class member method get_image(), after having called this method.

        **Parameters:**
        
        - phscreen:   The piston map in µm
        '''
        #from pdb import set_trace
        # Here 

        phs = np.zeros((self.csz, self.csz), dtype=np.float64)  # phase map
        
        if phscreen is not None:  # a phase screen was provided
            if self.remove_injection_piston:
                phs += phscreen - np.mean(phscreen[self.pupil])
            else:
                phs += phscreen
            
        wf = np.exp(1j*phs*self.mu2phase)
        wf *= np.sqrt(self.signal / self.pupil.sum())  # signal scaling
        wf *= self.pupil                               # apply the pupil mask
        self._phs = phs * self.pupil                   # store total phase
        self.fc_pa = self.sft(wf)                      # focal plane cplx ampl
        return self.fc_pa
    
    def get_inv_image(self, image):
        B = Hn(self._A1).dot(image.dot(Hn(self._A3)))
        return B
    
    def get_injection(self, phscreen):
        """
        Computes the complex injection phasor based on fiber mode in the pupil plane.
        
        **Parameters:**
        
        - phscreen: The piston map in µm
        """
        #set_trace()
        phs = phscreen.flatten()
        phs -= np.mean(phs[self.flatpup])
        # We do only the needed operations thanks to masking
        wf = np.sqrt(self.signal / self.pupilsum) * np.exp(1j*phs[self.flatpup]*self.mu2phase)  # signal scaling
        #wf *= self.pupil                               # apply the pupil mask
        #self._phs = phs * self.pupil                   # store total phase
        #self.fc_pa = self.sft(wf)                      # focal plane cplx ampl
        injected = self.flat_masked_lppup.dot(wf)
        return injected
    
    def get_tilt(self, phscreen=None):
        ''' Measures the tip-tilt measurement corresponding to the wavefront
        provided. The tip-tilt is in a 2-array and the unit is lambda/D 
        where D is the extent of the pupil mask.
        

        **Parameters:**
        
        - phscreen:   The piston map in µm
        '''
        #from pdb import set_trace
        # Here 

        phs = np.zeros((self.csz, self.csz), dtype=np.float64)  # phase map
        
        if phscreen is not None:  # a phase screen was provided
            if self.remove_injection_piston:
                phs += phscreen - np.mean(phscreen[self.pupil])
            else:
                phs += phscreen
            
        phase = self.pupil*phs*self.mu2phase
        tip  = phase.flatten().dot(self.ntip)
        tilt = phase.flatten().dot(self.ntilt)
        #wf *= np.sqrt(self.signal / self.pupil.sum())  # signal scaling
        #wf *= self.pupil                               # apply the pupil mask
        #self._phs = phs * self.pupil                   # store total phase
        #self.fc_pa = self.sft(wf)                      # focal plane cplx ampl
        return np.array([tip, tilt])
        
        
            
            
            
            
class injector(object):
    def __init__(self,pupil="VLT",
                 pdiam=8.,odiam=1., ntelescopes=4, tt_correction=None,
                 no_piston=False, lambda_range=None,
                 NA = 0.23,
                 a = 4.25e-6,
                 ncore = 2.7,
                 focal_hrange=20.0e-6,
                 focal_res=50,
                 pscale = 4.5,
                 interpolation=None,
                 rm_inj_piston=False,
                 seed=None,
                 atmo_config=None):
        """
        Generates fiber injection object.
        
        **Parameters:**
        
        - pupil     : The telescope pupil to consider
        - pdiam     : The pupil diameter
        - ntelescopes : The number of telescopes to inject
        - pupil     : Apupil name or definition
        - tt_correction : Amount of TT to correct (Not implemented yet)
        - no_piston : Remove the effect of piston at the injection 
          (So that it is handled only by the FT.)
        - NA        : Then numerical aperture of the fiber
        - a         : The radius of the core (m)
        - ncore    : The refractive index of the core
        - focal_hrange : The half-range of the focal region to simulate (m)
        - focal_res : The total resolution of the focal plane to simulate
        - pscale    : The pixel scale for imager setup (mas/pix)
        - seed      : Value to pass for random phase screen initialization
        - atmo_config: A parsed config file 
                    
        Use: call ``next(self.it)`` that returns injection phasors
        For more information, look into the attributes.
        
        To get the ideal injection: ``self.best_injection(lambdas)``
        """
        if lambda_range is None:
            self.lambda_range = np.linspace(3.0e-6, 4.2e-6, 6)
        else:
            self.lambda_range = lambda_range
        self.ntelescopes = ntelescopes
        self.pdiam = pdiam
        self.odiam = odiam
        self.collecting = np.pi/4*(self.pdiam**2 - self.odiam**2)
        
        self.atmo_config = atmo_config
        
        ##########
        # Temporary
        logit.warning("Hard-coded variables:")
        self.NA = NA #0.23#0.21
        self.a = a #4.25e-6#3.0e-6
        self.ncore = ncore #2.7
        self.focal_hrange = focal_hrange # 20.0e-6
        self.focal_res = focal_res # 50
        
        
        self.pscale = pscale
        #Now calling the common part of the config
        self.pupil = pupil
        self.interpolation = interpolation
        self.rm_inj_piston = rm_inj_piston
        self._setup(seed=seed)
        # Preparing iterators
        self.it = self.give_fiber()
        if interpolation is not None:
            self.compute_best_injection(interpolation=interpolation)
            self.get_efunc = self.give_interpolated(interpolation=interpolation)
        self.focal_planes = None
        self.injected = None
        
    @classmethod
    def from_config_file(cls, file=None, fpath=None,
                         focal_res=50, pupil=None, seed=None):
        """
        Construct the injector object from a config file
        
        **Parameters:**
        
        file      : A pre-parsed config file
        fpath     : The path to a config file
        nwl       : The number of wl channels
        focal_res : The total resolution of the focal plane to simulate 
        
        Gathers the variables from the config file then calls for a class instance (``__init__()``)
        """
        from scifysim import parsefile
        if file is None:
            logit.debug("Need to read the file")
            assert fpath is not None , "Need to provide at least\
                                        a path to a config file"
            logit.debug("Loading the parsefile module")
            theconfig = parsefile.parse_file(fpath)
            confpath = Path(fpath).parent
        else:
            logit.debug("file provided")
            assert isinstance(file, parsefile.ConfigParser), \
                             "The file must be a ConfigParser object"
            theconfig = file
            if fpath is None:
                logit.error("Now we use fpath to provide the root for appendix config files")
            else:
                confpath = Path(fpath)
        
        # Numerical aperture
        NA = theconfig.getfloat("fiber", "num_app")
        # Core radius
        a = theconfig.getfloat("fiber", "core")
        if np.isclose(a, 0., atol=1.0e-10):
            logit.warning("Core radius is set to 0 : optimize")
            logit.warning("Not implemented in scifysim: set to 4.25")
            a = 4.25e-6
        else:
            a = theconfig.getfloat("fiber", "core")
        # Core index
        ncore = theconfig.getfloat("fiber", "core_index")
        # Wavelength coverage
        lambcen = theconfig.getfloat("photon", "lambda_cen")
        lambwidth = theconfig.getfloat("photon", "bandwidth")
        interpolation = theconfig.get("photon", "injection_spectral_interp")
        #nwls = theconfig.getint("photon", "n_spectral_science")
        nwl = theconfig.getint("photon", "n_spectral_injection")
        if "None" in interpolation:
            nwl = nwls
        elif "nearest":
            pass
        elif "linear" in interpolation:
            if nwl<1:
                logit.warning("Need minimum 2 spectral chanels at injection")
                logit.warning("to get quadratic interpolation (selected %d)"%(nwl))
                nwl = max(2, nwl)
        elif "quadratic" in interpolation:
            if nwl<3:
                logit.warning("Need minimum 3 spectral chanels at injection")
                logit.warning("to get quadratic interpolation (selected %d)"%(nwl))
                nwl = max(3, nwl)
        elif "cubic" in interpolation:
            if nwl<4:
                logit.warning("Need minimum 4 spectral chanels at injection")
                logit.warning("to get cubic interpolation (selected %d)"%(nwl))
        else :
            print("Interpolation: ", interpolation)
            raise ValueError("Unrecognised interpolation")
            
        lambmin = lambcen - lambwidth/2
        lambmax = lambcen + lambwidth/2
        lambda_range = np.linspace(lambmin, lambmax, nwl)
        
        pdiams = theconfig.getarray("configuration","diam")
        pdiam = pdiams[0]
        odiams = theconfig.getarray("configuration", "cen_obs")
        odiam = odiams[0]
        
        separate_atmo_file = theconfig.getboolean("appendix","use_atmo")
        if separate_atmo_file:
            rel_path = theconfig.get("appendix", "atmo_file")
            if confpath is not None:
                logit.warning("file for atmo configuration:")
                logit.warning((confpath/rel_path).absolute())
                atmo_config = parsefile.parse_file(confpath/rel_path)
            else:
                logit.error("confp was not provided")
                raise NameError("Need to provide fpath")
        else:
            logit.warning("Using same file for atmo configuration")
            atmo_config = theconfig
        
            
        #atmo_mode = theconfig.get("atmo", "atmo_mode")
        #if "seeing" in atmo_mode:
        #    r0 = seeing_to_r0(theconfig.getfloat("atmo","seeing"), lambcen)
        #else:
        #    r0 = theconfig.getfloat("atmo", "r0")
        
        
        logit.warning("Setting default focal range")
        logit.warning("focal_hrange=20.0e-6,  pscale = 4.5")
        # Focal scale and range
        focal_res = focal_res
        focal_hrange = theconfig.getfloat("fiber", "focal_hrange")
        pscale = theconfig.getfloat("fiber", "pscale")
        logit.warning("Needs a nice way to build pupils in here")
        if pupil is None:
            pres = theconfig.getint("atmo", "pup_res")
            radius = pres//2
            pupil = tel_pupil(pres, pres, radius, file=theconfig, tel_index=0)
        
        ntelescopes = theconfig.getint("configuration", "n_dish")
        
        rm_inj_piston = theconfig.getboolean("atmo", "remove_injection_piston")
        
        obj = cls(pupil=pupil,
                 pdiam=pdiam, odiam=odiam,
                 ntelescopes=ntelescopes, tt_correction=None,
                 no_piston=False, lambda_range=lambda_range,
                 atmo_config=atmo_config,
                 NA=NA,
                 a=a,
                 ncore=ncore,
                 focal_hrange=focal_hrange,
                 focal_res=focal_res,
                 pscale=pscale,
                 interpolation=interpolation,
                 rm_inj_piston=rm_inj_piston,
                 seed=seed)
        obj.config = theconfig
        return obj
        
        
        
    def _setup(self, seed=None):
        """
        Common part of the setup
        Nota: the seed for creation of the screen is incremented by 1 between pupils
        """
        self.phscreensz = self.pupil.shape[0]
        
        self.screen = []
        self.focal_plane = []
        for i in range(self.ntelescopes):
            if seed is not None:
                theseed = seed + i
            else: 
                theseed = seed
            self.screen.append(atmo(config=self.atmo_config,
                                    seed=theseed,
                                    wind_angle=0.08+0.01*i))
            self.focal_plane.append([focuser(csz=self.phscreensz,
                                             xsz=self.focal_res, ysz=self.focal_res, pupil=self.pupil,
                                             pscale=self.pscale, wl=wl, rm_inj_piston=self.rm_inj_piston) for wl in self.lambda_range])
            self.fiber = fiber_head()
        # Caluclating the focal length of the focuser
        self.focal_length = self.focal_hrange/utilities.mas2rad(self.focal_plane[0][0].fov/2)
        for fp in self.focal_plane:
            for wl in range(self.lambda_range.shape[0]):
                fp[wl].signal = 1.
                fp[wl].phot_noise = False
        self.LP01 = fiber_head()
        self.LP01.full_consolidation(self.NA, self.a, self.ncore)
        ### Still need to figure out the focal scale
        self.lpmap = self.LP01.numerical_evaluation(self.focal_hrange, self.focal_res, self.lambda_range)
        quartiles = []
        for i, amap in enumerate(self.lpmap):
            quartiles.append([i*np.max(amap)/4 for i in range(4)])
        self.map_quartiles = np.array(quartiles)
        
        # Computing the mode of the fiber in the pupil plane for all scopes and wavelengths
        # Although they should be all the same
        lppup = []
        for i, scope in enumerate(self.focal_plane): # Iterating ont the scopes
            focal_wl = []
            for i, fiberwl in enumerate(scope): # Iterating on the wavelengths
                a_lp_pup = fiberwl.pupil * fiberwl.get_inv_image(self.lpmap[i,:,:])
                fiberwl.lppup = a_lp_pup
                fiberwl.flat_masked_lppup = fiberwl.lppup.flatten()[fiberwl.flatpup]   # This is pre-computation/ to optimize the computation
                focal_wl.append(a_lp_pup)
            lppup.append(focal_wl)
        self.lppup = np.array(lppup)
    
    def reset_screen(self):
        for ascreen in self.screen:
            ascreen.reset()
        
    def give_fiber(self,):
        while True:
            focal_planes = []
            for i, scope in enumerate(self.focal_plane):
                thescreen = next(self.screen[i].it)
                focal_wl = []
                for fiberwl in scope:
                    focal_wl.append(fiberwl.getimage(thescreen))
                focal_planes.append(focal_wl)
            focal_planes = np.array(focal_planes)
            self.focal_planes = focal_planes
            self.injected = np.sum(self.focal_planes*self.lpmap[None,:,:,:], axis=(2,3))
            yield self.injected
            
    def give_comparison(self,):
        """
        A test comparison between image and pupil plane injection
        """
        while True:
            focal_planes = []
            thescreens = []
            for i, scope in enumerate(self.focal_plane):
                thescreen = next(self.screen[i].it)
                thescreens.append(thescreen)
                focal_wl = []
                for fiberwl in scope:
                    focal_wl.append(fiberwl.getimage(thescreen))
                focal_planes.append(focal_wl)
            focal_planes = np.array(focal_planes)
            #self.focal_planes = focal_planes
            orig_injected = np.sum(focal_planes*self.lpmap[None,:,:,:], axis=(2,3))
            
            focal_planes = []
            for i, scope in enumerate(self.focal_plane):
                thescreen = thescreens[i]
                focal_wl = []
                for ascope in scope:
                    focal_wl.append(ascope.get_injection(thescreen))
                    #pupil_injected = self.pupil
                focal_planes.append(focal_wl)
            focal_planes = np.array(focal_planes)
            pupil_injected = focal_planes
            #self.focal_planes = focal_planes
            #self.injected = np.sum(self.focal_planes*self.lpmap[None,:,:,:], axis=(2,3))
            yield orig_injected, pupil_injected
        

    def all_inj_phasors(self,lambdas):
        """
        Convenience function that applies interpolation for all the inputs
        """
        outs = np.array([self.einterp[i](lambdas) for i in range(self.ntelescopes)])
        return outs
    def best_injection(self,lambdas):
        """
        Convenience function that applies interpolation for all the inputs
        """
        outs = np.array([self.best_einterp[i](lambdas) for i in range(self.ntelescopes)])
        return outs
            
    def give_interpolated_injection_image_plane(self,interpolation):
        """
        This one will yield the method that interpolates all the injection phasors
        """
        from scipy.interpolate import interp1d
        while True:
            focal_planes = []
            for i, scope in enumerate(self.focal_plane):
                thescreen = next(self.screen[i].it)
                focal_wl = []
                for fiberwl in scope:
                    focal_wl.append(fiberwl.getimage(thescreen))
                focal_planes.append(focal_wl)
            focal_planes = np.array(focal_planes)
            self.focal_planes = focal_planes
            self.injected = np.sum(self.focal_planes*self.lpmap[None,:,:,:], axis=(2,3))
            #self.injected = np.array([])
            self.einterp = [interp1d(self.lambda_range,
                                      self.injected[i,:],kind=interpolation,
                                      fill_value="extrapolate")\
                                                            for i in range(self.ntelescopes)]
            yield self.all_inj_phasors
            
            
    def give_interpolated(self,interpolation):
        """
        This one will yield the method that interpolates all the injection phasors
        """
        from scipy.interpolate import interp1d
        while True:
            injection_values = []
            for i, scope in enumerate(self.focal_plane):
                thescreen = next(self.screen[i].it)
                inj_wl = []
                for fiberwl in scope:
                    inj_wl.append(fiberwl.get_injection(thescreen))
                injection_values.append(inj_wl)
            injection_values = np.array(injection_values)
            #self.focal_planes = focal_planes
            self.injected = injection_values
            #self.injected = np.array([])
            self.einterp = [interp1d(self.lambda_range,
                                      self.injected[i,:],kind=interpolation,
                                      fill_value="extrapolate")\
                                                            for i in range(self.ntelescopes)]
            yield self.all_inj_phasors
        
    def compute_best_injection(self,interpolation):
        """
        This one will yield the method that interpolates all the injection phasors
        """
        from scipy.interpolate import interp1d
        
        focal_planes = []
        for i, scope in enumerate(self.focal_plane):
            thescreen = np.zeros_like(next(self.screen[i].it))
            focal_wl = []
            for fiberwl in scope:
                focal_wl.append(fiberwl.getimage(thescreen))
            focal_planes.append(focal_wl)
        focal_planes = np.array(focal_planes)
        self.focal_planes = focal_planes
        self.injected = np.sum(self.focal_planes*self.lpmap[None,:,:,:], axis=(2,3))
        self.best_einterp = [interp1d(self.lambda_range,
                                  self.injected[i,:],kind=interpolation,
                                  fill_value="extrapolate")\
                                                        for i in range(self.ntelescopes)]
        
        
    def compute_injection_function(self, interpolation="linear", tilt_res=50, tilt_range=2.):
        """
        Computes an interpolation of the injection as a function of a tip-tilt
        
        ``injector.injection_abs(wl [m], offset [lambda/D])``
        """
        from scipy.interpolate import interp2d
        
        meanwl = np.mean(self.lambda_range)
        # tilt_vector goes for 1 lambda/D
        tilt_vector = np.linspace(-meanwl/2*1e6, meanwl/2*1e6,
                                  self.phscreensz)[None,:] * np.ones(self.phscreensz)[:,None]
        offset = np.linspace(0., tilt_range, tilt_res)
        injecteds = []
        for k, theoffset in enumerate(offset):
            thescreen = theoffset * tilt_vector
            focal_planes = []
            for i, scope in enumerate(self.focal_plane):
                focal_wl = []
                for fiberwl in scope:
                    focal_wl.append(fiberwl.getimage(thescreen))
                focal_planes.append(focal_wl)
            focal_planes = np.array(focal_planes)
            injected = np.sum(focal_planes*self.lpmap[None,:,:,:], axis=(2,3))[0]
            #print(injected.dtype)
            injecteds.append(injected)
        injecteds = np.array(injecteds)
        
        self.injection_rate = unsorted_interp2d(self.lambda_range, offset, np.abs(injecteds)**2, kind=interpolation,fill_value=0.)
        self.injection_rate.__doc__ = """rate(wavelength[m], offset[lambda/D])"""
        self.injection_arg = unsorted_interp2d(self.lambda_range, offset, np.angle(injecteds), kind=interpolation, fill_value=0.)
        self.injection_arg.__doc__ = """phase(wavelength[m], offset[lambda/D])"""
        return
    
    def update_screens(self):
        """
        Reset all the screens for the injection
        """
        for ascreen in self.screen:
            ascreen.update_screen()

    
    
    
    
    
    
    
    
    
    
class fringe_tracker(object):
    def __init__(self, theconfig, seed=None, precompute=False):
        """
        Creates the module that simulates the OPD residual from fring tracker performance
        theconfig    : A parsed config file
        seed         : Seed for the generation of random OPDs
        """
        self.config = theconfig
        self.seed = seed
        self.precompute = precompute
        self.reference_file = self.config.get("fringe tracker", "reference_file")
        self.wet_atmosphere = self.config.getboolean("fringe tracker", "wet_atmosphere")
        self.dry_scaling = self.config.getfloat("fringe tracker", "dry_scaling")
        self.wet_scaling = self.config.getfloat("fringe tracker", "wet_scaling")
        self.static_bias_scaling = self.config.getfloat("fringe tracker", "static_bias_scaling")
        logit.warning("Loading keyword n_dish from [configuration] in fringe_tracking")
        self.n_tel = self.config.getint("configuration","n_dish")
        data = np.loadtxt(parent/self.reference_file, comments="#")
        self.ref_freqs = data[:,0]
        self.ref_ps_phase = data[:,1]
        self.ref_ps_disp = data[:,2]
        self.ref_dt = 1/(2*np.max(self.ref_freqs))
        # timestep: the one that will be used in the simulator
        logit.warning("Loading keyword step_time from [atmo] in fringe_tracking")
        self.timestep = self.config.getfloat("atmo", "step_time")
        
        
        
    def prepare_time_series(self,lamb, duration=10, replace=True):
        """
        Call to refresh the time series to use
        duration        : The duration of the time series to prepare
        replace         : Replace the time series (otherwise append)
        """
        logit.warning("Preparing a fringe tracking residual time series")
        element_duration = self.ref_dt * self.ref_ps_phase.shape[0]
        elements_needed = int(duration/element_duration) + 1
        dryps = []
        disps = []
        for k in range(self.n_tel):
            #Building up to the required length
            dryp = None
            disp = None
            for i in range(elements_needed + 1):
                dryp = utilities.random_series_fft(self.ref_ps_disp, matchto=dryp, keepall=True, seed=self.seed)
                dryp = dryp -(1-self.static_bias_scaling)*np.mean(dryp)
                # Make sure to increment the seed every use
                if self.seed is not None:
                    self.seed = self.seed+1
                disp = utilities.random_series_fft(self.ref_ps_disp, matchto=disp, keepall=True, seed=self.seed)
                if self.seed is not None:
                    self.seed = self.seed+1
            dryps.append(dryp)
            disps.append(disp)
        # Assumbling into an array of columns
        dryps = np.array(dryps).T * self.dry_scaling
        disps = np.array(disps).T * self.wet_scaling
        logit.warning("Dry pistons and dispersion residuals scaling refreshed")
        
        self.ref_sample_times = np.arange(0, self.ref_dt * dryps.shape[0], self.ref_dt)
        #desired_sample_times = np.arange(0, duration, self.timestep)
        
        if replace or ():
            self.dry_piston_series = dryps
            self.dispersion_series = disps
        else:
            self.dry_piston_series = np.concatenate((self.dry_piston_series, self.dry_piston_series), axis=0)
            self.dispersion_series = np.concatenate((self.dispersion_series, self.dispersion_series), axis=0)
        self.prepare_interpolation()
        self.phasor = self.iterator(lamb)
        
    def iterator(self, lamb):
        """
        Iterator that yields the phasors of fringe tracker residuals
        The iterator sets for precomputed or direct interpolation depending on the configuration.
        It also sets for wet or dry computation depending on the configuration.
        """
        if not self.precompute:
            available = int(np.max(self.ref_sample_times)/self.timestep)
            if not self.wet_atmosphere:
                i = 0
                while True:
                    if i>=available:
                        i = 0
                        self.prepare_time_series(lamb, duration=10, replace=True)
                    yield self.get_phasor_dry(i, lamb)
                    i += 1
            else:
                logit.error("Wet atmosphere not implemented")
                raise NotImplementedError("Wet atmosphere not implemented")
        else:
            self.interpolate_batch(np.max(self.ref_sample_times))
            if not self.wet_atmosphere:
                precomp_length = self.precomputed_series_piston.shape[0]
                i = 0
                while True:
                    if i>=precomp_length:
                        i = 0
                        self.prepare_time_series(lamb, duration=10, replace=True)
                    yield self.get_phasor_precomputed_dry(i,lamb)
                    i += 1
                    
            else:
                logit.error("Wet atmosphere not implemented")
                raise NotImplementedError("Wet atmosphere not implemented")
        
    def prepare_interpolation(self):
        """
        Mandatory
        Computes the interpolation functions to be used later
        
        save:
        
        ``self.piston_interpolation()``
        ``self.dispersion_interpolation``
        """
        self.piston_interpolation = interp1d(self.ref_sample_times, self.dry_piston_series, axis=0, kind="linear")
        self.dispersion_interpolation = interp1d(self.ref_sample_times, self.dispersion_series, axis=0, kind="linear")

    
    def interpolate_batch(self, duration):
        """
        optional:
        Prepares lookup tables for direct lookup of piston values
        """
        desired_sample_times = np.arange(0, duration, self.timestep)
        self.precomputed_series_piston = self.piston_interpolation(desired_sample_times)
        self.precomputed_series_dispersion = self.dispersion_interpolation(desired_sample_times)
    
    def get_phasor_precomputed_dry(self, i, lamb):
        """
        Precomputed computation are recommended for longer time series when trying to factor-in long timescale effects.
        One can prepare the "low" resolution series, and dump the original dataset.
        """
        phase = self.precomputed_series_dispersion[i,:][:,None] *2*np.pi / lamb[None,:]
        return np.ones_like(phase) * np.exp(1j*phase)
    def get_phasor_dry(self, i, lamb):
        phase = self.piston_interpolation(i*self.timestep)[:,None] * 2 * np.pi / lamb[None,:]
        return np.ones_like(phase) * np.exp(1j*phase)
    
                                                    
    
        
        

        
        
        
        
        
        
        
        
        
        
        

        
    
class unsorted_interp2d(interp2d):
    def __call__(self, x, y, dx=0, dy=0):
        if (len(x) == 1) and (len(y) == 1):
            return interp2d.__call__(self, x, y, dx=dx, dy=dy, assume_sorted=True)
        asx = np.argsort(x)
        usx = np.argsort(asx)
        asy = np.argsort(y)
        usy = np.argsort(asy)
        
        return interp2d.__call__(self, x[asx], y[asy], dx=dx, dy=dy, assume_sorted=True)[usy,:][:,usx]

from scipy.special import j0, k0
from scipy.constants import mu_0, epsilon_0
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import logging

logit = logging.getLogger(__name__)

from scipy.special import j0, j1, k0

from sympy.utilities.lambdify import implemented_function
J0 = implemented_function("J_0", j0)
J1 = implemented_function("J_1", j1)
K0 = implemented_function("K_0", k0)
from sympy.functions.elementary.piecewise import Piecewise
from kernuller import fprint



class gaussian_fiber_head(object):
    def __init__(self, NA=None,
                mfd=None, wl_mfd=None,
                apply=True):
        """
        A class that constructs a gaussian approximation for a fiber mode.
        Numerical evaluations are based on NA or mfd and wl_mfd, depending
        on which is provided.
        ufuncs are stored in:
        ``Hy_r``
        ``Hy_xy``
        
        **Parameters:**
        
        - NA        : The numerical aperture
        - mfd       : The mode field diameter [m]
        - wl_mfd    : The wavelength at which the mfd is measured [m]
        """
        self.r, self.x, self.y = sp.symbols("r, x, y", real=True)
        self.lamb, self.lamb0 = sp.symbols("lambda, lambda_0", real=True, positive=True)
        self.w, self.R, self.k = sp.symbols("w, R, k", real=True)
        # q, the beam parameter
        self.q = 1/(1/self.R - sp.I*self.lamb/(sp.pi*self.w**2))
        qwaist = self.q.subs([(self.R, sp.oo)])
        self.u = 1/self.q*sp.exp(-sp.I*self.k/2*(self.x**2+self.y**2)*1/self.q)
        self.ur = 1/self.q*sp.exp(-sp.I*self.k/2*self.r**2*1/self.q)
        uwaist = self.u.subs([(self.R, sp.oo),
                        (self.k, 2*sp.pi/self.lamb)])
        
        #self.nclad, self.ncore, self.eps0, self.mu = sp.symbols("n_clad n_core epsilon_0, mu", real=True)
        #self.lamb, self.r, self.a  = sp.symbols("lambda r a", real=True)
        #self.U, self.V, self.W = sp.symbols("U V W", real=True)
        #self.c_H = sp.symbols("c_H", real=True)
        self.NA = sp.symbols("N.A.", real=True, positive=True)
        # ? taken from Shaklan and Roddier?
        self.d = sp.symbols("d", real=True) # pupil diameter
       
        self.define_substitutions()
        
        self.build_equation()
        
        self.thesubs = []#self.subcore_na
        if NA is not None:
            self.NA_value = NA
            self.mfd_value = None
            self.wl_mfd_value = None
            self.thesubs.append((self.NA, self.NA_value))
        elif mfd is not None:
            self.mfd_value = mfd
            self.wl_mfd_value = wl_mfd
            self.NA_value = (self.lamb0/(sp.pi*self.w_0)).subs([(self.lamb0, self.mfd_value),
                                                                (self.w_0, mfd/2)])
            self.thesubs.append((self.NA, ))
        #self.consolidate_equation(self.thesubs)
        if apply:
            self.consolidate_equation(self.thesubs)
        
        
    def define_substitutions(self):
        self.w_0 = sp.symbols("w_0", real=True, positive=True)
        self.z = sp.symbols("z", real=True)
        self.zr = sp.pi*self.w_0**2/self.lamb
        self.Rz = (self.z**2 + self.zr**2)/self.z # that one is not used for now
        #fprint(Rz, "R(z) = ")
        self.wz = self.w_0*sp.sqrt(1+self.z**2/self.zr**2)
        #fprint(wz, "w(z) = ")
        # This one is the cool one!
        self.ur_lamb = self.ur.subs([(self.k, 2*sp.pi/self.lamb),
                  (self.w, self.wz),
                  (self.R, self.Rz)])
        self.ncore, self.nclad = sp.symbols("n_{core}, c_{clad}", real=True, positive=True)
        
    def build_equation(self):
        self.Hy = self.ur_lamb.subs([(self.z, 0),
                                    (self.w_0, self.lamb/self.NA)])
        #fprint(Hy,"H_y = ")
        
    def consolidate_equation(self, thesubs):
        self.Hy_consolidated = self.Hy.subs(thesubs) # Apply the subsitutions
        #self.Hy_r = 
        self.Hy_r = sp.lambdify((self.r, self.NA, self.lamb),self.Hy_consolidated)
        self.subr = (self.r, sp.sqrt(self.x**2+self.y**2))
        self.Hy_xy = sp.lambdify((self.x, self.y, self.NA, self.lamb),
                                 self.Hy_consolidated.subs([self.subr]))
    def full_consolidation(self, NA=None,
                mfd=None, wl_mfd=None):
        """
        Completes the consolidation of the lambda function
        with the application parameters:
        
        **Parameters:**
        
        NA      : The numerical aperture
        mfd     : The mode field diameter in [m]
        wl_mfd  : The wavelength at which mfd is measured in [m]
        
        """
        if NA is not None:
            thesubs = [(self.NA, NA),
                      self.subr]
        else :
            thesubs = [(self.w_0, mfd/2),
                       (self.lamb0, wl_mfd),
                      self.subr]
        self.Hy_xylambda = self.Hy_consolidated.subs(thesubs)
        self.Hy_xy_full = sp.lambdify((self.x, self.y,self.lamb),self.Hy_xylambda)
    def numerical_evaluation(self, half_range, nsamples, lambs):
        xx, yy = np.meshgrid(np.linspace(-half_range, half_range, nsamples),
                                      np.linspace(-half_range, half_range, nsamples))
        amap = self.Hy_xy_full(xx[None,:,:], yy[None,:,:], lambs[:,None,None])
        map_total = np.sqrt(np.sum(np.abs(amap)**2, axis=(1,2)))
        self.map = amap / map_total[:,None,None]
        #from pdb import set_trace
        #set_trace()
        return self.map



class fiber_head(object):
    def __init__(self, ):
        """
        A class that constructs helps compute the LP01 mode of
        a fiber.
        
        **ufuncs are stored in:**
        
        - ``Hy_r``
        - ``Hy_xy``
        """
        self.nclad, self.ncore, self.eps0, self.mu = sp.symbols("n_clad n_core epsilon_0, mu", real=True)
        self.lamb, self.r, self.a  = sp.symbols("lambda r a", real=True)
        self.U, self.V, self.W = sp.symbols("U V W", real=True)
        self.c_H = sp.symbols("c_H", real=True)
        self.NA = sp.symbols("N.A.", real=True)
        # ? taken from Shaklan and Roddier?
        self.d = sp.symbols("d", real=True) # pupil diameter
        self.intensity_E0 = 8/(sp.pi*self.d**2)*sp.sqrt(self.mu/self.eps0)*1/self.nclad
        
        self.define_substitutions()
        
        self.build_equation()
        
        self.thesubs = [self.subU_VW,
                                   self.subru_neu,
                                   self.subv,
                                   self.subch,
                                   self.subcore_na,
                                   (self.mu, mu_0),
                                   (self.eps0, epsilon_0)]
        self.consolidate_equation(self.thesubs)
        
        
        
        
    def define_substitutions(self):
        # Defining some basic substitutions from the paper
        self.subv = (self.V, ((2*sp.pi)/self.lamb)*self.a*self.NA) #Expr. of V from N.A.
        self.subna = (self.NA, sp.sqrt(self.ncore**2 -self.nclad**2)) #Expr of NA from nclad & ncore
        self.subcore_na = (self.nclad, sp.sqrt(self.ncore**2-self.NA**2)) #Expr nclad from N.A.
        self.subch = (self.c_H, sp.sqrt(2*self.nclad/sp.pi)*(self.eps0/self.mu)**(1/4)) #c_H from Shaklan&Roddier
        # From Rudolph and Neumann 1976 (cited in Coudé du Forest 1994)
        self.subU_VW = (self.U, sp.sqrt(self.V**2 - self.W**2)) # U from V and W
        self.subru_neu = (self.W, 1.1428*self.V - 0.9960) # The approximation of the transcendental eq.
        
    def build_equation(self):
        self.piece1 = J0(self.U*self.r/self.a)/J0(self.U)
        self.piece2 = K0(self.W*self.r/self.a)/K0(self.W)
        self.common =  self.c_H/sp.sqrt(2)*self.W*J0(self.U)/(self.a*self.V*J1(self.U)) 

        self.Hy = Piecewise((self.common*self.piece1, self.r<self.a),
                            (self.common*self.piece2, self.r>self.a))
        #fprint(Hy,"H_y = ")
        
    def consolidate_equation(self, thesubs):
        self.Hy_consolidated = self.Hy.subs(thesubs) # Apply the subsitutions
        self.Hy_r = sp.lambdify((self.r, self.NA, self.a, self.lamb, self.ncore),self.Hy_consolidated)
        self.x, self.y = sp.symbols("x, y")
        self.subr = (self.r, sp.sqrt(self.x**2+self.y**2))
        self.Hy_xy = sp.lambdify((self.x, self.y, self.NA, self.a, self.lamb, self.ncore),
                                 self.Hy_consolidated.subs([self.subr]))
    def full_consolidation(self, NA, a, ncore):
        """
        Completes the consolidation of the lambda function
        with the application parameters:
        
        - NA  : The numerical aperture
        - a   : The core radius in meters
        - ncore : the refractive index of the core
        """
        thesubs = [(self.NA, NA),
                  (self.a, a),
                  (self.ncore, ncore),
                  self.subr]
        self.Hy_xylambda = self.Hy_consolidated.subs(thesubs)
        self.Hy_xy_full = sp.lambdify((self.x, self.y,self.lamb),self.Hy_xylambda)
    def numerical_evaluation(self, half_range, nsamples, lambs):
        xx, yy = np.meshgrid(np.linspace(-half_range, half_range, nsamples),
                                      np.linspace(-half_range, half_range, nsamples))
        amap = self.Hy_xy_full(xx[None,:,:], yy[None,:,:], lambs[:,None,None])
        map_total = np.sqrt(np.sum(np.abs(amap)**2, axis=(1,2)))
        self.map = amap / map_total[:,None,None]
        #from pdb import set_trace
        #set_trace()
        return self.map


import astropy.units as units
class injection_vigneting(object):
    """
    A shortcut to injection simulation in the case of diffuse sources
    injector     : an injector object to emulate
    res          : 
    """
    def __init__(self, injector, res, crop=1.):
        """
        A shortcut to injection simulation in the case of diffuse sources
        
        **Parameters:**
        
        - injector     : an injector object to emulate
        - res          : The resolution of the map
        - crop         : A factor that scales the FoV of the map
        """
        # First build a grid of coordinates
        
        lambond = (np.mean(injector.lambda_range) / injector.pdiam)*units.rad.to(units.mas)
        self.mas2lambond = 1/lambond
        
        hskyextent = (injector.focal_hrange/injector.focal_length)*units.rad.to(units.mas)
        hskyextent = hskyextent*crop
        self.resol = res #injector.focal_res
        xx, yy = np.meshgrid(
                            np.linspace(-hskyextent, hskyextent, self.resol),
                            np.linspace(-hskyextent, hskyextent, self.resol))
        self.collecting = np.pi/4*(injector.pdiam**2 - injector.odiam**2)
        self.ds = np.mean(np.gradient(xx)[1]) * np.mean(np.gradient(yy)[0]) #In mas^2
        self.ds_sr = (self.ds*units.mas**2).to(units.sr).value # In sr
        self.xx = xx.flatten()
        self.yy = yy.flatten()
        self.rr = np.sqrt(self.xx**2 + self.yy**2)
        self.rr_lambdaond =  self.rr*self.mas2lambond
        print(self.mas2lambond)
        
        if not hasattr(injector, "injection_rate"):
            injector.compute_injection_function("linear", tilt_range=1.)
        self.vig = injector.injection_rate(injector.lambda_range, self.rr_lambdaond)
        self.vig_func = injector.injection_rate
        self.norm = 1/np.max(self.vig)
    def vigneted_spectrum(self, spectrum, lambda_range, exptime):
        """
        spectrum   : Flux density in ph/s/sr/m^2
        """
        factor = self.collecting * self.ds_sr * exptime
        vigneted_spectrum = self.vig_func(lambda_range, self.rr_lambdaond) * (spectrum * factor)[None,:]
        return vigneted_spectrum
        
    
def test_fiber():
    
    
    NA = 0.21
    a = 3.0e-6
    lamb0 = 3.0e-6
    lambmax = 4.2e-6
    
    focrange = 8.0e-6
    lamb_range = np.linspace(lamb0, lambmax, 10)
    xx, yy = np.meshgrid(np.linspace(-focrange, focrange, 100), np.linspace(-focrange, focrange, 100))
    
    myfiber = fiber_head()
    
    for asub in myfiber.thesubs:
        fprint(asub[1], sp.latex(asub[0])+" = ")
    fprint(myfiber.Hy, "H_y = ")
    
    
    print("Fiber for the following parameters")
    print(r"N.A. = %.2f, a = %.1f µm, \lambda = [%.1f..%.1f] (µm)"%
                          (NA, a*1e6, lamb0*1e6, lambmax*1e6))
    
    LPMAP = myfiber.Hy_xy(xx[None,:,:], yy[None,:,:], 0.21, 3.0e-6, lamb_range[:,None, None], 2.7)
    LPMAP = LPMAP/LPMAP.sum()
    myfiber.full_consolidation(0.21, 3.0e-6, 2.7)
    LPMAP2 = myfiber.numerical_evaluation(10.0e-6, 50, lamb_range)
    plt.figure()
    plt.imshow(np.abs(LPMAP2[0,:,:]))
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.imshow(np.abs(LPMAP2[9,:,:]))
    plt.colorbar()
    plt.show()
    rn = np.linspace(0., 8.0e-6, 1000)
    H1D = myfiber.Hy_r(rn, 0.21, 3.0e-6, 3.6e-6, 2.7)
    plt.figure()
    plt.plot(rn, H1D)
    plt.show()
    return myfiber
            
def test_injection(phscreensz=200, r0=8.,
                   interpolation=None, seed=20127):
    """
    Remember to pass seed=None if you want a random initialization
    """
    import xaosim
    # Construct a pupil using xaosim
    apup = xaosim.pupil.VLT(phscreensz, phscreensz,phscreensz/2)
    myinst = injector(pupil=apup, r0=r0,
                     interpolation=interpolation, seed=seed)
    import matplotlib.pyplot as plt
    injected = next(myinst.it)
    #contourlabels = ["0.","0.25","0.5", "O.75"]
    #tweaking the colormap showing the pupil cutout
    current_cmap = plt.matplotlib.cm.get_cmap("coolwarm")
    current_cmap.set_bad(color='black')
    plt.figure(figsize=(8,4),dpi=100)
    for i in range(myinst.ntelescopes):
        plt.subplot(1,myinst.ntelescopes,i+1)
        plt.imshow((myinst.focal_plane[i][0]._phs/myinst.focal_plane[i][0].pupil),
                           cmap=current_cmap)
    plt.show()
    plt.figure(figsize=(8,4),dpi=100)
    for i in range(myinst.ntelescopes):
        plt.subplot(2,myinst.ntelescopes,i+1)
        plt.imshow(np.abs(myinst.focal_planes[i,0]), cmap="Blues")
        CS = plt.contour(myinst.lpmap[0], levels=myinst.map_quartiles[0], colors="black")
        plt.clabel(CS, inline=1, fontsize=6)
        plt.subplot(2,myinst.ntelescopes,i+1+myinst.ntelescopes)
        plt.imshow(np.abs(myinst.focal_planes[i,-1]), cmap="Reds")
        CS = plt.contour(myinst.lpmap[-1], levels=myinst.map_quartiles[-1], colors="black")
        plt.clabel(CS, inline=1, fontsize=6)
    plt.suptitle("Injection focal plane (contours: LP01 mode quartiles)")
    plt.show()

    tindexes = ["Telescope %d"%(i) for i in range(injected.shape[0])]
    plt.figure()
    width = 0.1
    for i in range(injected.shape[1]):
        plt.bar(np.arange(4)+i*width,np.abs(injected[:,i]), width, label="%.1f µm"%(myinst.lambda_range[i]*1e6))
    plt.legend(loc="lower right",fontsize=7, title_fontsize=8)
    plt.ylabel("Injection amplitude")
    plt.title("Injection amplitude for each telescope by WL")
    plt.show()

    plt.figure()
    width = 0.1
    for i in range(injected.shape[1]):
        plt.bar(np.arange(4)+i*width,np.angle(injected[:,i]), width, label="%.1f µm"%(myinst.lambda_range[i]*1e6))
    plt.legend(loc="lower right",fontsize=7, title_fontsize=8)
    plt.ylabel("Injection phase (radians)")
    plt.title("Injection phase for each telescope by WL")
    plt.show()
    
    return myinst

def test_injection_fromfile(phscreensz=200, 
                            fpath="/home/rlaugier/Documents/hi5/SCIFYsim/scifysim/config/default_new_4T.ini",
                            seed=20127):
    """
    **Remember** to pass ``seed=None`` if you want a **random initialization**
    """
    import xaosim
    # Construct a pupil using xaosim
    apup = xaosim.pupil.VLT(phscreensz, phscreensz,phscreensz/2)
    myinst = injector.from_config_file(fpath=fpath,
                                     pupil=apup,
                                     seed=seed)
    import matplotlib.pyplot as plt
    injected = next(myinst.it)
    #contourlabels = ["0.","0.25","0.5", "O.75"]
    #tweaking the colormap showing the pupil cutout
    current_cmap = plt.matplotlib.cm.get_cmap("coolwarm")
    current_cmap.set_bad(color='black')
    plt.figure(figsize=(8,4),dpi=100)
    for i in range(myinst.ntelescopes):
        plt.subplot(1,myinst.ntelescopes,i+1)
        plt.imshow((myinst.focal_plane[i][0]._phs/myinst.focal_plane[i][0].pupil),
                           cmap=current_cmap)
    plt.show()
    plt.figure(figsize=(8,4),dpi=100)
    for i in range(myinst.ntelescopes):
        plt.subplot(2,myinst.ntelescopes,i+1)
        plt.imshow(np.abs(myinst.focal_planes[i,0]), cmap="Blues")
        CS = plt.contour(myinst.lpmap[0], levels=myinst.map_quartiles[0], colors="black")
        plt.clabel(CS, inline=1, fontsize=6)
        plt.subplot(2,myinst.ntelescopes,i+1+myinst.ntelescopes)
        plt.imshow(np.abs(myinst.focal_planes[i,-1]), cmap="Reds")
        CS = plt.contour(myinst.lpmap[-1], levels=myinst.map_quartiles[-1], colors="black")
        plt.clabel(CS, inline=1, fontsize=6)
    plt.suptitle("Injection focal plane (contours: LP01 mode quartiles)")
    plt.show()

    tindexes = ["Telescope %d"%(i) for i in range(injected.shape[0])]
    plt.figure()
    width = 0.1
    for i in range(injected.shape[1]):
        plt.bar(np.arange(4)+i*width,np.abs(injected[:,i]), width, label="%.1f µm"%(myinst.lambda_range[i]*1e6))
    plt.legend(loc="lower right",fontsize=7, title_fontsize=8)
    plt.ylabel("Injection amplitude")
    plt.title("Injection amplitude for each telescope by WL")
    plt.show()

    plt.figure()
    width = 0.1
    for i in range(injected.shape[1]):
        plt.bar(np.arange(4)+i*width,np.angle(injected[:,i]), width, label="%.1f µm"%(myinst.lambda_range[i]*1e6))
    plt.legend(loc="lower right",fontsize=7, title_fontsize=8)
    plt.ylabel("Injection phase (radians)")
    plt.title("Injection phase for each telescope by WL")
    plt.show()
    
    return myinst
    
    

# An implementation from Ruiliier 2005 (thèse)
# WIP
def optimize_a():
    import sympy as sp
    rho, beta, alpha = sp.symbols("rho, beta, alpha", real=True, positive=True)
    D, omega0, lamb, f = sp.symbols("D, omega_0, lambda, f", real=True, positive=True)
    rhoexpr = 2*((sp.exp(-beta**2) - sp.exp(-beta**2*alpha**2) ) / (beta*sp.sqrt(1-alpha**2) ) )**2
    betaexpr = sp.pi*D*omega0/(2*lamb*f)

    rhoexpr = rhoexpr.subs([(beta, betaexpr)])
    sp.solve(rhoexpr.diff(omega0), omega0)
    return


def tel_pupil(n,m, radius, file=None, pdiam=None,
              odiam=None, spiders=True, between_pix=True, tel_index=0):
    ''' ---------------------------------------------------------
    returns an array that draws the pupil of the VLT
    at the center of an array of size (n,m) with radius "radius".
    
    This is an approximation to reduce number of parameters:
    offset an angle are approximated. See xaosim.pupil.VLT for original
    
    Parameters describing the pupil were deduced from a pupil mask
    description of the APLC coronograph of SPHERE, by Guerri et al, 
    2011. 

    `<http://cdsads.u-strasbg.fr/abs/2011ExA....30...59G>`_
    
    --------------------------------------------------------- '''
    import xaosim
    if file is not None:
        if pdiam is None:
            pdiam = file.getarray("configuration","diam")[tel_index]
        if odiam is None:
            odiam = file.getarray("configuration","cen_obs")[tel_index]
    else:
        if (odiam is None) or (pdiam is None):
            raise ValueError("Provide either a file or kw values")
            
    # Those are fill values, and should not be very important
    thick  = 0.04              # adopted spider thickness (meters)
    offset = odiam #1.11              # spider intersection offset (meters)
    beta   = 50.      #50.5           # spider angle beta
    
    apupil = xaosim.pupil.four_spider_mask(m, n, radius, pdiam, odiam, 
                            beta, thick, offset, spiders=spiders,
                            between_pix=between_pix)

    return apupil
def test_injection_function(asim):
    import matplotlib.pyplot as plt
    asim.injector.compute_injection_function("linear", tilt_range=1.)
    os = np.linspace(0,1., 500)
    plt.figure()
    wl = np.linspace(3e-6, 4e-6, 8)
    rates = asim.injector.injection_rate(wl, os)
    args = asim.injector.injection_arg(wl, os)
    for i, awl in enumerate(wl) :
        plt.plot(os, rates[:,i], label=r"%.1f $\mu m$"%(awl*1e6))
    plt.legend()
    plt.show()

    plt.figure()
    for i, awl in enumerate(wl) :
        plt.plot(os, args[:,i], label=r"%.1f $\mu m$"%(awl*1e6))
    plt.legend()
    plt.show()
    
    

def seeing_to_r0(seeing, wl):
    """
    seeing : seeing in arcseconds
    wl     : wl in m
    """
    r0 = wl / (seeing/3600*np.pi/180) 
    return r0
    

# ===========================================================
# ===========================================================