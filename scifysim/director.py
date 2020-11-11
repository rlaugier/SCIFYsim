
import kernuller
import scifysim as sf
import numpy as np
from tqdm import tqdm
import logging

logit = logging.getLogger(__name__)

class simulator(object):
    def __init__(self, location=None, array=None,
                tarname="No name", tarpos=None, n_spec_ch=100):
        """
        The object meant to assemble and operate the simulator
        location    : The location of the observatory (None defaults to VLTI)
        array       : The pupil array geometry (either "VLTI", "CHARA"... or ndarray
                        (see kernuller.VLTI for the format))
        tarname     : Name of the target
        tarpos      : RADEC position of the target
        
        n_spec_ch : Then number of sprectral chanels simulated
        
        """
        logit.info("Setting up the array")
        if (array is None) and (location is None):
            raise NotImplementedError("You must provide an array")
        
        if "VLTI" in array:
            self.array = kernuller.VLTI
            self.multi_dish = True
        elif "CHARA" in array:
            self.array = kernuller.CHARA
            self.multi_dish = True
        #elif combiner is not None:
        #    logit.warning("The array will be defined by default based on location")
        #    pass
        else :
            # The array is provided as an ndarray
            logit.info("Setting an arbitrary array")
            if not isinstance(array, np.ndarray):
                logit.error("Array name not understood")
                logit.error("'CHARA' or 'VLTI' or a numpy ndarray")
                raise ValueError("The array parameter must be \
                            the name of a supported arary\
                            or a numpy ndarray.")
                                 
            if array.shape[1] is not 2:
                raise ValueError("Shape of array must be (n, 2)")
            self.array = array
            
        # Setting up the location
        logit.info("Setting up the location")
        if location is None:
            logit.warning("Location not provided")
            if "VLTI" in array:
                logit.warning("Location defaulted to Paranal")
                location = "paranal"
            elif "CHARA" in array:
                logit.warning("Location defaulted to CHARA")
                location = "CHARA"
        else:
            # If location is provided.
            self.obs = sf.observatory.observatory(self.array, location=location)
            
        # Target informations
        self.tarname = tarname
        self.tarpos = tarpos
        self.n_spec_ch = n_spec_ch
        
        self.sequence = None
        
        
    
    @classmethod
    def from_config(cls, file=None, fpath=None,):
        """
        Construct the injector object from a config file
        file      : A pre-parsed config file
        fpath     : The path to a config file
        nwl       : The number of wl channels
        focal_res : The total resolution of the focal plane to simulate 
        """
        from scifysim import parsefile
        if file is None:
            logit.debug("Need to read the file")
            assert fpath is not None , "Need to proved at least\
                                        a path to a config file"
            logit.debug("Loading the parsefile module")
            theconfig = parsefile.parse_file(fpath)
        else:
            logit.debug("file provided")
            assert isinstance(file, parsefile.ConfigParser), \
                             "The file must be a ConfigParser object"
            theconfig = file
        
        location = theconfig.get("configuration", "location")
        array = theconfig.get("configuration", "config")
        n_spec_ch = theconfig.getint("photon", "n_spectral_science")
        
        # Defining the target
        mode = theconfig.get("target", "mode")
        if "name" in mode:
            tarname = theconfig.get("target", "target")
            tarpos = sf.observatory.astroplan.FixedTarget.from_name(tarname)
        else:
            raise NotImplementedError("Some day we will be able to do it by RADEC position")
        
        
        obj = cls(location=location, array=array,
                 tarname=tarname, tarpos=tarpos, n_spec_ch=n_spec_ch)
        obj.config = theconfig
        return obj
    
    
    
    def prepare_injector(self, file=None, ntelescopes=None, tt_correction=None,
                 no_piston=None, lambda_range=None,
                 r0=None,
                 NA=None,
                 a=None,
                 ncore=None,
                 focal_hrange=None,
                 focal_res=None,
                 pscale=None,
                 interpolation=None,
                 injector=None):
        """
        Either: 
            * Provide all parameters
            * Provide a config file
            * provide the injector
        """
        if file is not None:
            logit.warning("tt_correction not implemented")
            self.tt_correction = tt_correction
            logit.warning("no_piston not implemented")
            self.no_piston = no_piston
            self.injector = sf.injection.injector.from_config_file(file=file)
            # Recovering some 
        else:
            if injector is None:
                logit.warning("No file provided, please prepare injector manually")
                logit.warning("Then pass it as kwarg")
                return
            else:
                self.injector = injector
            return
        # Recovering some high level parameters:
        self.ntelescopes = self.injector.ntelescopes
        # Preparing science spectral chanels
        self.lambda_science_range = np.linspace(self.injector.lambda_range[0],
                                               self.injector.lambda_range[-1],
                                               self.n_spec_ch)
        pass
    def prepare_combiner(self, file, **kwargs):
        """
        Constructs self.combiner
        """
        self.combiner_type = file.get("configuration","combiner")
        self.ph_tap = file.get("configuration", "photometric_tap")
        
        if "angel_woolf_ph" in self.combiner_type:
            self.combiner = sf.combiner.combiner.angel_woolf(file,**kwargs)
        else:
            raise NotImplementedError("Only Angel and Woolf combiners for now")
        
    def prepare_spectrograph(self, file):
        pass
    
        
    def make_exposure(self, injection_gen , texp=1., t_co=2.0e-3, time=None):
        """
        Simulate an exposure
        texp      : Exposure time (seconds)
        t_co      : Coherence time (seconds) 
        """
        self.n_subexps = int(texp/t_co)
        taraltaz = self.obs.observatory_location.altaz(time, target=self.target)
        array = self.obs.get_projected_array(taraltaz)
        integrator = sf.spectrograph.integrator()
        for i in tqdm(range(self.n_subexps)):
            injected = next(self.injector.get_efunc)(self.lambda_science_range)
            # lambdified argument order matters! This should remain synchronous
            # with the lambdify call
            combined = np.array([self.combiner.encaps(self.source.xx, self.source.yy,
                                    array.flatten(), 2*np.pi/thelambda, injected[:,i])
                                     for i, thelambda in enumerate(self.lambda_science_range)])
            # incoherently combining over sources
            # Warning: modifying the array
            combined = np.sum(combined*np.conjugate(combined), axis=(1))
            integrator.accumulate(combined)
        mean, std = integrator.compute_stats()
        result = integrator.acc
    def prepare_sequence(self, file=None, times=None, n_points=20, remove_daytime=False):
        """
        Prepare an observing sequence
        """
        if file is not None:
            logit.info("Building sequence from new config file")
            pass
        else:
            logit.info("Building sequence from main config file")
            file = self.file
        
        self.seq_start_string = file.get("target", "seq_start")
        self.seq_end_string = file.get("target", "seq_end")
        self.sequence = self.obs.build_observing_sequence(times=[self.seq_start_string, self.seq_end_string],
                            npoints=n_points, remove_daytime=remove_daytime)
        self.target = sf.observatory.astroplan.FixedTarget.from_name(self.tarname)
            
        pass
    def make_sequence(self):
        """
        Run an observing sequence
        """
        pass
    def build_map(self):
        """
        Create sensitivity map
        """
        seq_exists = hasattr(self, sequence)
        if not seq_exists:
            # If no sequence was prepared, build the map for observations at zenith
            pass
        else:
            # Build the series of maps
            pass
        
    def __call__(self):
        pass
    
def test_director():
    logit.warning("hard path in the test")
    asim = simulator.from_config(fpath="/home/rlaugier/Documents/hi5/SCIFYsim/scifysim/config/default_new_4T.ini")
    asim.prepare_injector(asim.config)
    asim.prepare_combiner(asim.config)