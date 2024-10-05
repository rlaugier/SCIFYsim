import sympy as sp
import numpy as np
import scifysim as sf
import kernuller
import logging

logit = logging.getLogger(__name__)


class combiner(object):
    def __init__(self,expr, thesubs,
                 mask_bright=None,
                 mask_dark=None,
                 mask_photometric=None,
                 lamb=None):
        
        self.Na = expr.shape[1]
        self.M = expr
        self.bright = mask_bright
        self.dark = mask_dark
        self.photometric = mask_photometric
        self.baseline_subs = thesubs
        self.lamb = lamb
        
        #X = sp.MatrixSymbol('X', 2, 4)
        X = sp.Matrix(sp.symbols('X:{}'.format(self.Na), real=True))
        Y = sp.Matrix(sp.symbols('Y:{}'.format(self.Na), real=True))
        self.Xm = sp.Matrix([[X,Y]])

        lamb = sp.symbols("lambda", real=True)
        self.k = sp.symbols("k", real=True)
        self.alpha = sp.symbols("alpha", real=True)
        self.beta = sp.symbols("beta", real=True)
        self.A = sp.Matrix([[self.alpha],
                       [self.beta]])
        # The error phasor:
        # Prone to change: vectorization of lambda would require e(lambda) in sympy!
        # Will need to build that instead of the scipy.interp1D
        self.e = sp.Matrix(sp.symbols('e:{}'.format(self.Na), real=False))
        self.E = sf.utilities.vec2diag(self.e)

        # The source amplitude
        s = sp.symbols("s", real=True)
        # Source amplitude is not managed here
        thesubs.append((s, 1))



        geomphaseterm = sp.I*self.k*self.Xm@self.A
        # The pointing matrix:
        self.P = sp.Matrix([sp.exp(term) for term in geomphaseterm])
        self.T_clean = self.M@self.E@(s*self.P)
        self.T_subs = self.T_clean.subs(thesubs)
        
        self.encaps = sf.utilities.ee(self.T_subs)
        # Here, lambdifying for the parameters
        self.encaps.lambdify((self.alpha, self.beta,self.Xm ,self.k, self.e), modules="numexpr")
        
        # For the chromatic combiners
            
    def chromatic_matrix(self, wl):
        """
        Work in progress. This helps define the chromatic behaviour of the combiner even
        when it is define achromatic.
        
        **Parameters:**
        
        * wl      : An array of wavelengths for which to compute the matrix
        
        **Result:** Several matrices are stored in `combiner`
        
        * self.Mcs    : a symbolic formula for the combiner matrix (2D sympy array)
          as a function of the wavenumber k
        * self.Mcl    : The matrix as a lambda-function of (k, 0) 
          The extra 0 parameter passed helps to cast the correct shape
        * self.Mcn    : The numpy ndarray of the matrix of interest
          shape is (n_wl, n_out, n_in)
        
        """
        # Defining a chromatic version of the matrix
        k = sp.symbols("k")
        mysubs = [(self.lamb, 2*sp.pi/k)]
        ks = 2*np.pi/wl
        self.Mcs = self.M.subs(mysubs)
        self.Mcl = sf.utilities.lambdifyz((k,), self.Mcs, modules="numpy")
        self.Mcn = np.moveaxis(self.Mcl(ks, 0), 2, 0)
        logit.warning("Comuputed chromatic combiner matrix with the following shape:")
        logit.warning(self.Mcn.shape)
        
    def pseudo_chromatic_matrix(self, wl):
        """
        Work in progress. This helps define the chromatic behaviour of the combiner even
        when it is define achromatic
        """
        # Defining a chromatic version of the matrix
        sigma, = self.M.free_symbols
        lamb = sp.symbols("lambda")
        k = sp.symbols("k")
        mysubs = [(sigma, 0.1),
                  (lamb, 2*sp.pi/k)]
        ks = 2*np.pi/wl
        self.Mcs = self.M.subs(mysubs)
        self.Mcl = sf.utilities.lambdifyz((k,), self.Mcs, modules="numpy")
        self.Mcn = np.moveaxis(self.Mcl(ks, 0), 2, 0)
        logit.warning("Comuputed chromatic combiner matrix with the following shape:")
        logit.warning(self.Mcn.shape)
        
        
    def refresh_array(self, thearray):
        """
        This method recomputes a disposable encapsulated function for the give pointing.
        """
        thesubs = []
        for symbol, val in zip(self.Xm, thearray.flatten()):
            thesubs.append((symbol, val))
        self.T_pointed = self.T_subs.subs(thesubs)
        self.pointed_encaps = sf.utilities.ee(self.T_pointed)
        self.pointed_encaps.lambdify((self.alpha, self.beta, self.k, self.e), modules="numexpr")
        
                
    @classmethod
    def angel_woolf(cls, file, ph_shifters=(0,sp.pi/2)):
        
        M, bright, dark, photo = sf.combiners.angel_woolf_ph(ph_shifters=ph_shifters,
                                                            include_masks=True)
        #Photometric tap
        logit.warning("Here, forced to assume I have only one symbol: sigma")
        for symbol in M.free_symbols:
            sigma = symbol
        
        thesigma = file.getfloat("configuration", "photometric_tap")
        thesubs = [(sigma, thesigma)]
        obj = cls(M, thesubs,
                  mask_bright=bright,
                  mask_dark=dark,
                  mask_photometric=photo)
        return obj
    @classmethod
    def from_config(cls, file, ph_shifters=(0,0)):
        """
        A 
        
        **Parameters:**

        - file         : The config file
        - ph_shifters  : Phase shifters between first and second stage default: (0,0)
                        These correspond to the "internal modulation"

        **Config keywords:**

        * ``configuration.combiner`` 						Name of the combiner architecture
        * ``configuration.photometric_tap`` 		Fraction of light (power) sent to the photometric channels

        * ``configuration.input_phase_offset``	Additional phase offset as part of the design of the combiner

        **Accepted combiners:**
        
        Managed by ``configuration.combiner``

        - angel_woolf_ph,
        - VIKiNG
        - GLINT
        - GRAVITY
        - bracewell_ph
        - bracewell

        """
        hasph = False
        lamb = sp.symbols("lambda")
        thesubs = []
        combiner_type = file.get("configuration", "combiner")
        tap_ratio = file.getfloat("configuration", "photometric_tap")
        phase_offset_type = file.get("configuration", "input_phase_offset")
        
        # Type of input phase offset
        if phase_offset_type == "none":
            input_offset = 0
        elif phase_offset_type == "achromatic":
            input_offset = sp.pi/2
        elif phase_offset_type == "geometric":
            wavelength = file.getfloat("photon", "lambda_cen")
            piston = wavelength/4 # this gives pi/2 for the central wavelength
            p, lamb, aphi = sf.combiners.p, sf.combiners.lamb, sf.combiners.aphi
            input_offset = aphi.subs([(p, piston)])
            #phase = sf.combiners.piston2phase(piston,lamb)
        else :
            raise KeyError("Phase shift type not recognized")
        
        coupler_techno = file.get("configuration", "coupler_techno")
        #(KG_MMI, Sharma_asym, Tepper_direct)
        if coupler_techno == "KG_MMI":
            Mc = sf.combiners.M_KG
        elif coupler_techno == "Sharma_asym":
            Mc = sf.combiners.M_Sharma
        elif coupler_techno == "Tepper_direct":
            Mc = sf.combiners.M_Tepper
        elif coupler_techno == "perfect":
            Mc = sf.combiners.pcoupler
        else:
            logit.error("combiner type not understood")
        
            
        # Switch on combiner type
        if combiner_type == "angel_woolf_ph":
            hasph = True
            M, bright, dark, photo = sf.combiners.angel_woolf_ph(ph_shifters=[0.0, sp.pi/2],
                                                            include_masks=True, tap_ratio=tap_ratio)
        elif combiner_type == "angel_woolf_ph_chromatic":
            hasph = True
            M, bright, dark, photo = sf.combiners.angel_woolf_ph_chromatic(Mc=Mc, ph_shifters=ph_shifters,
                                                            include_masks=True, tap_ratio=tap_ratio,
                                                            input_ph_shifters=input_offset*np.array([0,1,0,1]))
            if M.free_symbols == set():
                lamb = sp.symbols("lambda")
            else:
                lamb, = M.free_symbols
        # ------------------------------------------------- 
        elif combiner_type == "nott_kernel_null":
            hasph = True
            M, bright, dark, photo = sf.combiners.nott_kernel_null(Mc=Mc, include_masks=True, tap_ratio=tap_ratio)
            if M.free_symbols == set():
                lamb = sp.symbols("lambda")
            else:
                lamb, = M.free_symbols
                
        elif combiner_type == "nott_symmetric":
            hasph = True
            M, bright, dark, photo = sf.combiners.nott_symmetric(Mc=Mc, include_masks=True, tap_ratio=tap_ratio)
            if M.free_symbols == set():
                lamb = sp.symbols("lambda")
            else:
                lamb, = M.free_symbols
                
        elif combiner_type == "nott_nuller":
            hasph = True
            M, bright, dark, photo = sf.combiners.nott_nuller(Mc=Mc, include_masks=True, tap_ratio=tap_ratio)
            if M.free_symbols == set():
                lamb = sp.symbols("lambda")
            else:
                lamb, = M.free_symbols   
        # -----------------------------------------------
        elif combiner_type == "GLINT":
            hasph = True
            M, bright, dark, photo = sf.combiners.GLINT(include_masks=True,
                                                        tap_ratio=tap_ratio)
        elif combiner_type == "VIKiNG":
            hasph = True
            M, bright, dark, photo = sf.combiners.VIKiNG(include_masks=True,
                                                         tap_ratio=tap_ratio)
        elif combiner_type ==  "bracewell_ph":
            hasph = True
            M, bright, dark, photo = sf.combiners.bracewell_ph(ph_shifters=ph_shifters,
                                                            include_masks=True)
        elif combiner_type == "bracewell":
            hasph = False
            M, bright, dark, photo = sf.combiners.bracewell_ph(include_masks=True,
                                                            tap_ratio=0)
            M = M[1:3,:]
            bright = bright[1:3]
            dark = dark[1:3]
            photo = photo[1:3]

        elif combiner_type == "GRAVITY":
            hasph = False
            M = sf.combiners.GRAVITY(Mc=Mc, ph_shifter_type=phase_offset_type,wl=wavelength)
        elif combiner_type == "KN_3T":
            hasph=False
            M, bright, dark, photometric = sf.combiners.kernel_nuller_3T(include_masks=True)
        elif combiner_type == "KN_4T":
            hasph=False
            M, bright, dark, photometric = sf.combiners.kernel_nuller_4T(include_masks=True)
        elif combiner_type == "KN_5T":
            hasph=False
            M, bright, dark, photometric = sf.combiners.kernel_nuller_5T(include_masks=True)
        elif combiner_type == "KN_6T":
            hasph=False
            M, bright, dark, photometric = sf.combiners.kernel_nuller_6T(include_masks=True)
        else:
            logit.error("Nuller type not recognized")
            raise KeyError("Nuller type not found")
        
        if not hasph:
            bright = None
            dark = None
            photo = None
        
        obj = cls(M, thesubs,
                 mask_bright=bright,
                 mask_dark=dark,
                 mask_photometric=photo,
                 lamb=lamb)
        return obj

def test_combiner(combiner, nwl=10):
    hr = kernuller.mas2rad(16)
    xx, yy = np.meshgrid(np.linspace(-hr, hr, 1024),
                     np.linspace(-hr, hr, 1024))
    if combiner.Na == 4:
        array = kernuller.VLTI
    elif combiner.Na == 6:
        array = kernuller.CHARA
        
    #Prone to change: wavelength vecorization incomplete
    es = np.ones(combiner.Na)
    # This one is for vectorized wavelength (only yet assumes no injection-wl dependency)
    #amap = combiner.encaps(xx[:,:,None], yy[:,:,None], array.flatten(), 2*np.pi/3.5e-6*np.ones(10)[None,None,:],
    #                      np.ones(combiner.Na))
    lambda_range = np.linspace(3.0e-6, 4.2e-6, 10)
    amap = np.array([combiner.encaps(xx[:,:], yy[:,:], array.flatten(), np.array([2*np.pi/thelambda])[None,None,:],
                          np.ones(combiner.Na)) for thelambda in lambda_range])
    return amap
def test_angel_woolf():
    fpath = "/home/rlaugier/Documents/hi5/SCIFYsim/scifysim/config/default_new_4T.ini"
    logit.warning("Hard path here!")
    theconfig = sf.parsefile.parse_file(fpath)
    acombiner = combiner.angel_woolf(theconfig)
    print("free symbols initial",acombiner.T_clean.free_symbols)
    print("free symbols",acombiner.T_subs.free_symbols)
    amap = test_combiner(acombiner)
    print(amap.shape)
    return amap