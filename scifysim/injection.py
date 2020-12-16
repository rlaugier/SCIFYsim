# injection.py

"""
This module revolves around the injector class
At __init__(), injector builds
* one atmo object for each telescope (stored in a list)
* one focuser object (badly named) for each wavelength AND each telescope
* one fiber_head object that builds the map of LP01 mode.
It then creates a generator called `self.it` that will return
a series of complex numbers corresponding to injection complex phasors.

Testing:
test_fiber for the `fiber_head`
test_injector for the `injector`

"""



import numpy as np
import threading
import time
from pathlib import Path
# amto_screen is now provided in house to provide seed
#from xaosim import wavefront as wft

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2


import logging

logit = logging.getLogger(__name__)

from . import utilities

def seed_numpy(myseed):
    rs = np.random.RandomState(myseed)
    

def seeing_to_r0(seeing, wl):
    """
    seeing : seeing in arcseconds
    wl     : wl in m
    """
    r0 = wl / (seeing/3600*np.pi/180) 
    return r0
    

# ===========================================================
# ===========================================================








class atmo(object):
    '''Atmospheric Kolmogorov-type phase screen.

    ====================================================================

    Class Attributes:
    ----------------
    - csz     : size (csz x csz) of the phase screen       (in pixels)
    - pdiam   : diameter of the aperture within this array (in pixels)
    - rndarr  : uniformly distributed random array         (csz x csz)
    - kolm    : the original phase screen                  (csz x csz)
    - kolm2   : the oversized phase screen    ((csz + pdiam) x (csz + pdiam))
    - qstatic : an optional quasi static aberration    (pdiam x pdiam)
    - rms     : total phase screen rms value           (in nanometers)
    - rms_i   : instant rms inside the pupil           (in nanometers)

    Comment:
    -------
    While the attributes are documented here for reference, the prefered
    way of interacting with them is via the functions defined within the
    class.
    ====================================================================

    '''
    # ==================================================
    def __init__(self, name="GRAVITY+", csz = 512,
                 psz=200,
                 lsz=8.0, r0=0.2, L0=10.0,
                 fc=24.5, correc=1.0, seed=None,
                 pdiam=8.0, config=None):

        ''' Kolmogorov type atmosphere + qstatic error

        -----------------------------------------------------
        Parameters:
        ----------
        - name  : a string describing the instrument
        - csz   : the size of the Fourier array
        - psz   : the size of the pupil in  pixels
        - lsz   : the screen linear size (in meters)
        - r0    : the Fried parameter (in meters)
        - L0    : the outer scale parameter (in meters)
        -----------------------------------------------------
        '''

        if config is None:
            self.csz     = csz
            self.psz     = psz
            self.lsz     = lsz
            self.pdiam   = pdiam
            self.r0      = r0
            self.L0      = L0

            self.rms_i   = 0.0
            self.correc  = correc
            self.fc      = fc
        else :
            self.config = config
            pres = self.config.getint("atmo", "pup_res")
            screen_oversize = self.config.getfloat("atmo", "screen_size")
            pdiams = self.config.getarray("configuration","diam")
            pdiam = pdiams[0]
            r0 = self.config.getfloat("atmo", "r0")
            L0 = self.config.getfloat("atmo", "Lout")
            correc = self.config.getfloat("atmo", "correc")
            
            self.csz     = int(pres*screen_oversize)
            self.psz     = pres
            self.pdiam   = pdiam
            self.lsz     = self.pdiam*screen_oversize
            self.r0      = r0
            self.L0      = L0

            self.rms_i   = 0.0
            self.correc  = correc
            self.fc      = fc
            
        self.kolm    = atmo_screen(self.csz, self.lsz, self.r0, self.L0, self.fc, self.correc, pdiam=self.pdiam, seed=seed).real
        
        self.qstatic = np.zeros((self.psz, self.psz))
        self.shm_phs = self.qstatic

        self.kolm2   = np.tile(self.kolm, (2,2))
        #self.kolm2   = self.kolm2[:self.sz+self.pdiam,:self.sz+self.pdiam]



        self.offx = 0 # x-offset on the "large" phase screen array
        self.offy = 0 # y-offset on the "large" phase screen array

        self.ttc     = False   # Tip-tilt correction flag
        
        # auxilliary array (for tip-tilt correction)
        self.xx, self.yy  = np.meshgrid(np.arange(self.csz)-self.csz//2,
                                        np.arange(self.csz)-self.csz//2)
        self.xxnorm2 = np.sum(self.xx**2)
        self.yynorm2 = np.sum(self.yy**2)
        
        self.it = self.give()
    

    # ==============================================================
    def set_qstatic(self, qstatic=None):
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
        
        ------------------------------------------------ '''
        if r0 is not None:
            self.r0 = r0

        if L0 is not None:
            self.L0 = L0

        if correc is not None:
            self.correc = correc

        if fc is not None:
            self.fc = fc
            
        self.kolm    = atmo_screen(
            self.csz, self.lsz, self.r0, self.L0, self.fc, self.correc, seed=seed).real

        self.kolm2   = np.tile(self.kolm, (2,2))

        if self.keepgoing is False:
            # case that must be adressed:
            # amplitude changed when atmo is frozen!
            subk = self.kolm2[self.offx:self.offx+self.psz,
                              self.offy:self.offy+self.psz].copy()
            
            if self.ttc is True:            
                ttx = np.sum(subk*self.xx) / self.xxnorm2
                tty = np.sum(subk*self.yy) / self.yynorm2
                subk -= ttx * self.xx + tty * self.yy

            self.rms_i = subk.std()
            self.shm_phs = subk + self.qstatic


    def give(self):
        ''' ------------------------------------------
        Main loop: frozen screen slid over the aperture

        Options:
        ---------
        -----------------------------------------  '''

        while True:
            self.offx += 2
            self.offy += 1
            self.offx = self.offx % self.csz
            self.offy = self.offy % self.csz

            subk = self.kolm2[self.offx:self.offx+self.psz,
                              self.offy:self.offy+self.psz].copy()

            if self.ttc is True:
                ttx = np.sum(subk*self.xx) / self.xxnorm2
                tty = np.sum(subk*self.yy) / self.yynorm2
                subk -= ttx * self.xx + tty * self.yy

            self.rms_i = subk.std()
            self.shm_phs = subk + self.qstatic
            yield self.shm_phs
            
# ==================================================================
def atmo_screen(isz, ll, r0, L0, fc=25, correc=1.0, pdiam=None,seed=None):
    ''' -----------------------------------------------------------
    The Kolmogorov - Von Karman phase screen generation algorithm.

    Adapted from the work of Carbillet & Riccardi (2010).
    http://cdsads.u-strasbg.fr/abs/2010ApOpt..49G..47C

    Kolmogorov screen can be altered by an attenuation of the power
    by a correction factor *correc* up to a cut-off frequency *fc*
    expressed in number of cycles across the phase screen

    Parameters:
    ----------

    - isz    : the size of the array to be computed (in pixels)
    - ll     :  the physical extent of the phase screen (in meters)
    - r0     : the Fried parameter, measured at a given wavelength (in meters)
    - L0     : the outer scale parameter (in meters)
    - fc     : DM cutoff frequency (in lambda/D)
    - correc : correction of wavefront amplitude (factor 10, 100, ...)
    - pdiam  : pupil diameter (in meters)

    Returns: two independent phase screens, available in the real and 
    imaginary part of the returned array.

    Remarks:
    -------
    If pdiam is not specified, the code assumes that the diameter of
    the pupil is equal to the extent of the phase screen "ll".
    ----------------------------------------------------------- '''
    
    #phs = 2*np.pi * (np.random.rand(isz, isz) - 0.5)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    phs = rng.uniform(low=-np.pi, high=np.pi, size=(isz,isz))

    xx, yy = np.meshgrid(np.arange(isz)-isz/2, np.arange(isz)-isz/2)
    rr = np.hypot(yy, xx)
    rr = shift(rr)
    rr[0,0] = 1.0

    modul = (rr**2 + (ll/L0)**2)**(-11/12.)

    if pdiam is not None:
        in_fc = (rr < fc * ll / pdiam)
    else:
        in_fc = (rr < fc)

    modul[in_fc] /= correc
    
    screen = ifft(modul * np.exp(1j*phs)) * isz**2
    screen *= np.sqrt(2*0.0228)*(ll/r0)**(5/6.)

    screen -= screen.mean()
    return(screen)

# ======================================================================

            
dtor = np.pi/180.0  # to convert degrees to radians
i2pi = 1j*2*np.pi   # complex phase factor













class focuser(object):
    ''' Generic monochoromatic camera class

    ===========================================================================
    The camera is connected to the other objects (DM- and atmosphere- induced
    wavefronts) via shared memory data structures, after instantiation of
    the camera, when it is time to take an image.

    Thoughts:
    --------

    I am also considering another class to xaosim to describe the astrophysical
    scene and the possibility to simulate rather simply images of complex
    objects.

    Using a generic convolutive approach would work all the time, but may be
    overkill for 90% of the use cases of this piece of software so I am not
    sure yet.
    ===========================================================================

    '''

    # =========================================================================
    def __init__(self, name="SCExAO_chuck", csz=200, ysz=256, xsz=256,
                 pupil=None,screen=None,
                 pdiam=7.92, pscale=10.0, wl=1.6e-6):
        ''' Default instantiation of a cam object:

        -------------------------------------------------------------------
        Parameters are:
        --------------
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

        self.pdiam  = pdiam                 # pupil diameter in meters
        self.pscale = pscale                # plate scale in mas/pixel
        self.fov = self.isz * self.pscale
        self.wl     = wl                    # wavelength in meters
        self.frm0   = np.zeros((ysz, xsz))  # initial camera frame

        self.btwn_pixel = False            # fourier comp. centering option
        self.phot_noise = False            # photon noise flag
        self.signal     = 1e6              # default # of photons in frame
        self.corono     = False            # if True: perfect coronagraph
      
        # final tune-up
        self.update_cam()

    # =========================================================================
    def update_cam(self, wl=None, pscale=None, between_pixel=None):
        ''' -------------------------------------------------------------------
        Change the filter, the plate scale or the centering of the camera

        Parameters:
        - pscale        : the plate scale of the image, in mas/pixel
        - wl            : the central wavelength of observation, in meters
        - between_pixel : whether FT are centered between four pixels or not
        ------------------------------------------------------------------- '''
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

        Parameters:
        ----------
        - nph: the total number of photons inside the frame
        ------------------------------------------------------------------- '''
        if (nph > 0):
            self.signal = nph
            self.phot_noise = True
        else:
            self.signal = 1e6
            self.phot_noise = False

    # =========================================================================
    def get_image(self, ):
        return(self.shm_cam)

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
        -------------------------------------------------------------------

        Parameters:
        ----------
        - atmo    : (optional) atmospheric phase screen
        - qstatic : (optional) a quasi-static aberration
        ------------------------------------------------------------------- '''

        # nothing to do? skip the computation!

        mu2phase = 4.0 * np.pi / self.wl / 1e6  # convert microns to phase

        phs = np.zeros((self.csz, self.csz), dtype=np.float64)  # phase map
        
        if phscreen is not None:  # a phase screen was provided
            phs += phscreen
            
        wf = np.exp(1j*phs*mu2phase)

        wf *= np.sqrt(self.signal / self.pupil.sum())  # signal scaling
        wf *= self.pupil                               # apply the pupil mask
        self._phs = phs * self.pupil                   # store total phase
        self.fc_pa = self.sft(wf)                      # focal plane cplx ampl
        return self.fc_pa



    # =========================================================================
    def nogive(self, dm_shm=None, atmo_shm=None):

        ''' ----------------------------------------
        Thread (infinite loop) that monitors changes
        to the DM, atmo, and qstatic data structures
        and updates the camera image.

        Parameters:
        ----------
        - dm_shm    : shared mem file for DM
        - atmo_shm  : shared mem file for atmosphere
        - qstat_shm : shared mem file for qstatic error
        --------------------------------------------

        Do not use directly: use self.start_server()
        and self.stop_server() instead.
        ---------------------------------------- '''
        dm_cntr = 0      # counter to keep track of updates
        atm_cntr = 0     # on the phase screens
        dm_map = None    # arrays that store current phase
        atm_map = None   # screens, if they exist
        nochange = True  # lazy flag!

        # 1. read the shared memory data structures if present
        # ----------------------------------------------------
        if dm_shm is not None:
            dm_map = shm(dm_shm)

        if atmo_shm is not None:
            atm_map = shm(atmo_shm)

        # 2. enter the loop
        # ----------------------------------------------------
        while self.keepgoing:
            nochange = True  # lazy flag up!

            if atm_map is not None:
                test = atm_map.get_counter()
                atmomap = atm_map.get_data()
                if test != atm_cntr:
                    atm_cntr = test
                    nochange = False
            else:
                atmomap = None

            self.make_image(phscreen=atmomap, dmmap=dmmap, nochange=nochange)
            
            
            
            
            
            
            
            
            
            
            
            
class injector(object):
    def __init__(self,pupil="VLT",
                 pdiam=8., ntelescopes=4, tt_correction=None,
                 no_piston=False, lambda_range=None,
                 NA = 0.23,
                 a = 4.25e-6,
                 ncore = 2.7,
                 focal_hrange=20.0e-6,
                 focal_res=50,
                 pscale = 4.5,
                 interpolation=None,
                 seed=None,
                 atmo_config=None):
        """
        Generates fiber injection object.
        pupil     : The telescope pupil to consider
        pdiam     : The pupil diameter
        ntelescopes : The number of telescopes to inject
        pupil     : Apupil name or definition
        tt_correction : Amount of TT to correct (Not implemented yet)
        no_piston : Remove the effect of piston at the injection 
                    (So that it is handled only by the FT.)
        NA        : Then numerical aperture of the fiber
        a         : The radius of the core (m)
        ncore    : The refractive index of the core
        focal_hrange : The half-range of the focal region to simulate (m)
        focal_res : The total resolution of the focal plane to simulate
        pscale    : The pixel scale for imager setup (mas/pix)
        seed      : Value to pass for random phase screen initialization
        atmo_config: A parsed config file 
                    
        Use: call `next(self.it)` that returns injection phasors
        For more information, look into the attributes.
        
        To get the ideal injection: self.best_injection(lambdas)
        """
        if lambda_range is None:
            self.lambda_range = np.linspace(3.0e-6, 4.2e-6, 6)
        else:
            self.lambda_range = lambda_range
        self.ntelescopes = ntelescopes
        self.pdiam = pdiam
        
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
        file      : A pre-parsed config file
        fpath     : The path to a config file
        nwl       : The number of wl channels
        focal_res : The total resolution of the focal plane to simulate 
        
        Gathers the variables from the config file then calls for a class instance (__init__())
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
        if np.isclose(a, 0.):
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
        focal_hrange = 20.0e-6
        pscale = 4.5
        logit.warning("Needs a nice way to build pupils in here")
        if pupil is None:
            pres = theconfig.getint("atmo", "pup_res")
            radius = pres//2
            pupil = tel_pupil(pres, pres, radius, file=theconfig, tel_index=0)
        
        ntelescopes = theconfig.getint("configuration", "n_dish")
        if ntelescopes is not 4:
            raise NotImplementedError("Currently only supports 4 telescopes")
            
        obj = cls(pupil=pupil,
                 pdiam=pdiam, ntelescopes=ntelescopes, tt_correction=None,
                 no_piston=False, lambda_range=lambda_range,
                 atmo_config=atmo_config,
                 NA=NA,
                 a=a,
                 ncore=ncore,
                 focal_hrange=focal_hrange,
                 focal_res=focal_res,
                 pscale=pscale,
                 interpolation=interpolation,
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
                                    seed=theseed))
            self.focal_plane.append([focuser(csz=self.phscreensz,
                                             xsz=self.focal_res, ysz=self.focal_res, pupil=self.pupil,
                                             pscale=self.pscale, wl=wl) for wl in self.lambda_range])
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
            
    def give_interpolated(self,interpolation):
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
        
        injector.injection_abs(wl [m], offset [lambda/D])
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
        
        self.injection_abs = interp2d(self.lambda_range, offset, np.abs(injecteds), kind=interpolation)
        self.injection_arg = interp2d(self.lambda_range, offset, np.angle(injecteds), kind=interpolation)
        return
            

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







class fiber_head(object):
    def __init__(self, ):
        """
        A class that constructs helps compute the LP01 mode of
        a fiber.
        ufuncs are stored in:
        Hy_r
        Hy_xy
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
        NA  : The numerical aperture
        a   : The core radius in meters
        ncore : the refractive index of the core
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
        self.map = amap / np.sqrt(np.sum(np.abs(amap)**2, axis=(1,2)))[:,None,None]
        return self.map
        
    
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
    Remember to pass seed=None if you want a random initialization
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

    http://cdsads.u-strasbg.fr/abs/2011ExA....30...59G
    
    
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
