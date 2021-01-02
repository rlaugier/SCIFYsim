
import kernuller
import scifysim as sf
import numpy as np
from tqdm import tqdm
import logging

from kernuller import mas2rad, rad2mas

logit = logging.getLogger(__name__)

class simulator(object):
    def __init__(self,file=None, fpath=None):
                 #location=None, array=None,
                #tarname="No name", tarpos=None, n_spec_ch=100):
        """
        The object meant to assemble and operate the simulator. Construct the injector object from a config file
        file      : A pre-parsed config file
        fpath     : The path to a config file
        
        """
        from scifysim import parsefile
        if file is None:
            logit.debug("Need to read the file")
            assert fpath is not None , "Need to proved at least\
                                        a path to a config file"
            logit.debug("Loading the parsefile module")
            self.config = parsefile.parse_file(fpath)
        else:
            logit.debug("file provided")
            assert isinstance(file, parsefile.ConfigParser), \
                             "The file must be a ConfigParser object"
            self.config = file
        
        self.location = self.config.get("configuration", "location")
        self.array_config = self.config.get("configuration", "config")
        raw_array = eval("kernuller.%s"%(self.array_config))
        self.order = self.config.getarray("configuration", "order").astype(np.int16)
        self.array = raw_array[self.order]
        self.n_spec_ch = self.config.getint("photon", "n_spectral_science")
        
        # Defining the target
        mode = self.config.get("target", "mode")
        if "name" in mode:
            self.tarname = self.config.get("target", "target")
            self.tarpos = sf.observatory.astroplan.FixedTarget.from_name(self.tarname)
        else:
            raise NotImplementedError("Some day we will be able to do it by RADEC position")
            
        self.multi_dish = self.config.getboolean("configuration", "multi_dish")
        

        #self.obs = sf.observatory.observatory(self.array, location=location)
        
        self.sequence = None
        
    def prepare_observatory(self, file=None, fpath=None):
        """
        Preprare the observatory object for the simulator.
        """
        if file is not None:
            theconfig = file
        elif fpath is not None:
            theconfig = parsefile.parse_file(fpath)
        else:
            logit.warning("Using the config file of the simulator for the observatory")
            theconfig = self.config
        self.obs = sf.observatory.observatory(config=theconfig)
        
        logit.warning("Undecided whether to store array in simulator or observatory")
        assert np.allclose(self.obs.statlocs, self.array)
        

    
    
    
    def prepare_injector(self, file=None, fpath=None,
                 ntelescopes=None, tt_correction=None,
                 no_piston=None, lambda_range=None,
                 NA=None,
                 a=None,
                 ncore=None,
                 focal_hrange=None,
                 focal_res=None,
                 pscale=None,
                 interpolation=None,
                 res = 50,
                 crop = 1.,
                 injector=None, seed=None):
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
            self.injector = sf.injection.injector.from_config_file(file=file, fpath=fpath, seed=seed)
            # Recovering some 
        else:
            if injector is None:
                logit.warning("No file provided, please prepare injector manually")
                logit.warning("Then pass it as kwarg")
                return
            else:
                self.injector = injector

        # Recovering some high level parameters:
        self.ntelescopes = self.injector.ntelescopes
        # Preparing science spectral chanels
        self.lambda_science_range = np.linspace(self.injector.lambda_range[0],
                                               self.injector.lambda_range[-1],
                                               self.n_spec_ch)
        
        # Peparing the vigneting function for the field of view and diffuse sources
        self.injector.vigneting = sf.injection.injection_vigneting(self.injector, res, crop)
        
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
    
        
    def make_exposure(self, injection_gen , texp=1., time=None):
        """
        Simulate an exposure
        texp      : Exposure time (seconds)
        t_co      : Coherence time (seconds) 
        """
        t_co = self.injector.atmo.step_time
        self.n_subexps = int(texp/t_co)
        #taraltaz = self.obs.observatory_location.altaz(time, target=self.target)
        #taraltaz, tarPA = self.obs.get_position(self.target, time)
        self.obs.point()
        array = self.obs.get_projected_array(taraltaz, PA=tarPA)
        integrator = sf.spectrograph.integrator()
        for i in tqdm(range(self.n_subexps)):
            injected = next(self.injector.get_efunc)(self.lambda_science_range)
            # lambdified argument order matters! This should remain synchronous
            # with the lambdify call
            combined = np.array([self.combiner.encaps(self.source.xx, self.source.yy,
                                    array.flatten(), 2*np.pi/thelambda, injected[:,m])
                                     for m, thelambda in enumerate(self.lambda_science_range)])
            # incoherently combining over sources
            # Warning: modifying the array
            combined = np.sum(np.abs(combined*np.conjugate(combined)), axis=(2))
            integrator.accumulate(combined)
        mean, std = integrator.compute_stats()
        return integrator
    
    def make_metrologic_exposure(self, interest, star, diffuse, texp=1., t_co=2.0e-3, time=None):
        """
        Simulate an exposure
        texp      : Exposure time (seconds)
        t_co      : Coherence time (seconds) 
        """
        self.n_subexps = int(texp/t_co)
        #taraltaz = self.obs.observatory_location.altaz(time, target=self.target)
        #taraltaz, tarPA = self.obs.get_position(self.target, time)
        
        #Pointing should be done already
        array = self.obs.get_projected_array(self.obs.altaz, PA=self.obs.PA)
        integrator = sf.spectrograph.integrator()
        
        integrator.static_xx = self.injector.vigneting.xx
        integrator.static_yy = self.injector.vigneting.yy
        integrator.static =  []
        for asource in diffuse:
            aspectrum = asource.get_downstream_absorbtion(self.lambda_science_range) \
                            * asource.get_own_brightness(self.lambda_science_range)
            # Collecting surface appears here
            vigneted_spectrum = self.injector.collecting \
                            * self.injector.vigneting.vigneted_spectrum(aspectrum,
                                                            self.lambda_science_range,
                                                            t_co)
            static_output = np.array([self.combiner.encaps(mas2rad(self.injector.vigneting.xx), mas2rad(self.injector.vigneting.yy),
                                    array.flatten(), 2*np.pi/thelambda, np.ones(self.ntelescopes, dtype=np.complex128))
                                     for m, thelambda in enumerate(self.lambda_science_range)])
            #print(static_output.shape)
            static_output = static_output.swapaxes(0,2) * vigneted_spectrum[:,None,:]
            static_output = np.sum(np.abs(static_output*np.conjugate(static_output)), axis=0)
            integrator.static.append(static_output)
                
        
        logit.warning("Currently no vigneting (requires a normalization of vigneting)")
        filtered_starlight = diffuse[0].get_downstream_absorbtion(self.lambda_science_range)
        # collected will convert from [ph / s /m^2] to [ph]
        collected = filtered_starlight * self.injector.collecting * t_co
        integrator.starlight = []
        integrator.planetlight = []
        for i in tqdm(range(self.n_subexps)):
            injected = next(self.injector.get_efunc)(self.lambda_science_range)
            # lambdified argument order matters! This should remain synchronous
            # with the lambdify call
            thexx = star.xx_f
            theyy = star.yy_f
            
            combined_starlight = np.array([self.combiner.encaps(mas2rad(thexx), mas2rad(theyy),
                                    array.flatten(), 2*np.pi/thelambda, injected[:,m])
                                     for m, thelambda in enumerate(self.lambda_science_range)])
            # getting a number of photons
            combined_starlight = combined_starlight * star.ss[:,None,:] * collected[:,None,None]
            combined_starlight = np.sum(np.abs(combined_starlight*np.conjugate(combined_starlight)), axis=(2))
            integrator.starlight.append(combined_starlight)
            
            combined_planetlight = np.array([self.combiner.encaps(mas2rad(interest.xx_f), mas2rad(interest.yy_f),
                                    array.flatten(), 2*np.pi/thelambda, injected[:,m])
                                     for m, thelambda in enumerate(self.lambda_science_range)])
            # getting a number of photons
            combined_planetlight = combined_planetlight * interest.ss[:,None,:] * collected[:,None,None]
            combined_planetlight = np.sum(np.abs(combined_planetlight*np.conjugate(combined_planetlight)), axis=(2))
            integrator.planetlight.append(combined_planetlight)
            
            # incoherently combining over sources
            # Warning: modifying the array
            #combined = np.sum(np.abs(combined*np.conjugate(combined)), axis=(2))
            #integrator.accumulate(combined)
            
        #mean, std = integrator.compute_stats()
        integrator.starlight = np.array(integrator.starlight)
        integrator.planetlight = np.array(integrator.planetlight)
        return integrator
    
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