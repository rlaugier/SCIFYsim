
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
        
        # Defining the target
        mode = theconfig.get("target", "mode")
        if "name" in mode:
            self.tarname = theconfig.get("target", "target")
            self.tarpos = sf.observatory.astroplan.FixedTarget.from_name(self.tarname)
        else:
            raise NotImplementedError("Some day we will be able to do it by RADEC position")
            
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
        self.R = self.lambda_science_range / np.gradient(self.lambda_science_range)
        
        # Peparing the vigneting function for the field of view and diffuse sources
        self.injector.vigneting = sf.injection.injection_vigneting(self.injector, res, crop)
        
        pass
    
    
    def prepare_combiner(self, file, **kwargs):
        """
        Constructs self.combiner
        """
        self.combiner_type = file.get("configuration","combiner")
        self.ph_tap = file.get("configuration", "photometric_tap")
        
        self.combiner = sf.combiner.combiner.from_config(file,**kwargs)
        
        
    def prepare_fringe_tracker(self, file=None, seed=None):
        if file is None:
            file = self.config
        self.fringe_tracker = sf.injection.fringe_tracker(file, seed=seed)
    def prepare_sources(self, file=None):
        """
        Prepares a src object containing star, planet, sky and UT.
        file     : A parsed config file (default: None will reuse the config file in self.)
        """
        if file is None:
            file = self.config
        self.src = sf.sources.star_planet_target(file, self)
    def prepare_integrator(self, config=None, keepall=False,
                           n_sources=4, infinite_well=False):
        """
        Prepares the integraro object that rules the pixel properties
        of the detector.
        config    : A parsed config file (default: None will reuse the config file in self.)
        keepall   : [boolean] Whether to keep each of the steps accumulated.
                            False is recommended for faster computation and memory efficiency
        n_sources : Number of sources sequencially accumulated. (deprecated, do not
                            rely on this number)
        """
        if config is None:
            config = self.config
        self.integrator = sf.spectrograph.integrator(config=config,
                                                     keepall=keepall,
                                                     n_sources=n_sources,
                                                     infinite_well=infinite_well)
        
        
    def prepare_spectrograph(self, config=None, n_chan=None):
        """
        Prepares the spectrograph object that maps the light over the detector.
        file     : A parsed config file (default: None will reuse the config file in self.)
        n_chan   : The number of outputs from the chip. If None: tries to get it from
                 the shape of self.combiner.M
        """
        if config is None:
            config = self.config
        if n_chan is None:
            n_chan = self.combiner.M.shape[0]
        self.spectro = sf.spectrograph.spectrograph(config,
                                                 self.lambda_science_range,
                                                 n_chan=n_chan)
        
    
    
    def make_metrologic_exposure(self, interest, star, diffuse,
                                 texp=1., t_co=2.0e-3, time=None,
                                 monitor_phase=True, dtype=np.float32):
        """
        Warning: at the moment, this assumes pointing has already been performed.
        
        Simulate an exposure with source of interest and sources of noise
        interest  : sf.sources.resolved_source object representing the source of intereest (planet)
        star      : sf.sources.resolved_source object representing the star
        diffuse   : a list of sf.transmission_emission objects linked in a chain.
        texp      : Exposure time (seconds)
        t_co      : Coherence time (seconds) 
        monitor_phase : If true, will also record values of phase errors from injection and fringe tracking
        
        Chained diffuse objects are used both for transmission and thermal background. 
        Result is returned as an integrator object. Currently the integrator object is only a
        vessel for the individual subexposures, and not used to do the integration.
        """
        t_co = self.injector.screen[0].step_time
        self.n_subexps = int(texp/t_co)
        self.integrator.n_subexps = self.n_subexps
        #taraltaz = self.obs.observatory_location.altaz(time, target=self.target)
        #taraltaz, tarPA = self.obs.get_position(self.target, time)
        
        #Pointing should be done already
        array = self.obs.get_projected_array(self.obs.altaz, PA=self.obs.PA)
        
        self.integrator.static_xx = self.injector.vigneting.xx
        self.integrator.static_yy = self.injector.vigneting.yy
        self.integrator.reset()
        self.integrator.static =  []
        self.integrator.static_list = []
        for asource in diffuse:
            aspectrum = asource.get_downstream_transmission(self.lambda_science_range) \
                            * asource.get_own_brightness(self.lambda_science_range) \
            # Collecting surface appears here
            vigneted_spectrum = self.injector.collecting \
                            * self.injector.vigneting.vigneted_spectrum(aspectrum,
                                                            self.lambda_science_range,
                                                            t_co)
            # vigneted_spectrum also handles the integration in time and space
            static_output = np.array([self.combiner.encaps(mas2rad(self.injector.vigneting.xx), mas2rad(self.injector.vigneting.yy),
                                    array.flatten(), 2*np.pi/thelambda, np.ones(self.ntelescopes, dtype=np.complex128))
                                     for m, thelambda in enumerate(self.lambda_science_range)])
            #print(static_output.shape)
            static_output = (static_output.swapaxes(0,2) * np.sqrt(vigneted_spectrum[:,None,:]))
            static_output = np.sum(np.abs(static_output*np.conjugate(static_output), dtype=dtype), axis=0)
            self.integrator.static.append(static_output.T)
            self.integrator.static_list.append(asource.__name__)
                
        
        logit.warning("Currently no vigneting (requires a normalization of vigneting)")
        filtered_starlight = diffuse[0].get_downstream_transmission(self.lambda_science_range)
        # collected will convert from [ph / s /m^2] to [ph]
        collected = filtered_starlight * self.injector.collecting * t_co
        self.integrator.starlight = []
        self.integrator.planetlight = []
        self.integrator.ft_phase = []
        self.integrator.inj_phase = []
        self.integrator.inj_amp = []
        logit.warning("Ugly management of injection/tracking")
        for i in tqdm(range(self.n_subexps)):
            injected = next(self.injector.get_efunc)(self.lambda_science_range)
            tracked = next(self.fringe_tracker.phasor)
            if monitor_phase:
                self.integrator.ft_phase.append(np.angle(tracked[:,0]))
                self.integrator.inj_phase.append(np.angle(injected[:,0]))
                self.integrator.inj_amp.append(np.abs(injected[:,0]))
            injected = injected * tracked
            # lambdified argument order matters! This should remain synchronous
            # with the lambdify call
            thexx = star.xx_f
            theyy = star.yy_f
            
            combined_starlight = np.array([self.combiner.encaps(mas2rad(thexx), mas2rad(theyy),
                                    array.flatten(), 2*np.pi/thelambda, injected[:,m])
                                     for m, thelambda in enumerate(self.lambda_science_range)])
            # getting a number of photons
            combined_starlight = combined_starlight * np.sqrt(star.ss[:,None,:] * collected[:,None,None])
            combined_starlight = np.sum(np.abs(combined_starlight*np.conjugate(combined_starlight), dtype=dtype), axis=(2))
            self.integrator.starlight.append(combined_starlight)
            
            combined_planetlight = np.array([self.combiner.encaps(mas2rad(interest.xx_f), mas2rad(interest.yy_f),
                                    array.flatten(), 2*np.pi/thelambda, injected[:,m])
                                     for m, thelambda in enumerate(self.lambda_science_range)])
            # getting a number of photons
            combined_planetlight = combined_planetlight * np.sqrt(interest.ss[:,None,:] * collected[:,None,None])
            combined_planetlight = np.sum(np.abs(combined_planetlight*np.conjugate(combined_planetlight), dtype=dtype), axis=(2))
            self.integrator.planetlight.append(combined_planetlight)
            
            # incoherently combining over sources
            # Warning: modifying the array
            # combined = np.sum(np.abs(combined*np.conjugate(combined)), axis=(2))
            # integrator.accumulate(combined)
            
        #mean, std = self.integrator.compute_stats()
        self.integrator.starlight = np.array(self.integrator.starlight, dtype=dtype)
        self.integrator.planetlight = np.array(self.integrator.planetlight, dtype=dtype)
        self.integrator.ft_phase = np.array(self.integrator.ft_phase, dtype=dtype)
        self.integrator.inj_phase = np.array(self.integrator.inj_phase, dtype=dtype)
        self.integrator.inj_amp = np.array(self.integrator.inj_amp, dtype=dtype)
        return self.integrator
    
    def make_exposure(self, interest, star, diffuse,
                                 texp=1., t_co=2.0e-3, time=None,
                                 monitor_phase=True,
                                 use_tqdm=False,
                                 dtype=np.float32,
                                 spectro=None):
        """
        Warning: at the moment, this assumes pointing has already been performed.
        
        Simulate an exposure with source of interest and sources of noise
        interest  : sf.sources.resolved_source object representing the source of intereest (planet)
        star      : sf.sources.resolved_source object representing the star
        diffuse   : a list of sf.transmission_emission objects linked in a chain.
        texp      : Exposure time (seconds)
        t_co      : Coherence time (seconds) 
        monitor_phase : If true, will also record values of phase errors from injection and
                        fringe tracking
        dtype     : Data type to use for results
        spectro   : spectrograph object to use. If None, the method will assume one pixel 
                        per output and per spectral channel
        
        Chained diffuse objects are used both for transmission and thermal background. 
        Result is returned as an integrator object. Currently the integrator object is only a
        vessel for the individual subexposures, and not used to do the integration.
        """
        t_co = self.injector.screen[0].step_time
        self.n_subexps = int(texp/t_co)
        
        #Pointing should be done already
        array = self.obs.get_projected_array(self.obs.altaz, PA=self.obs.PA)
        self.integrator.static_xx = self.injector.vigneting.xx
        self.integrator.static_yy = self.injector.vigneting.yy
        self.integrator.reset()
        self.integrator.static =  []
        for asource in diffuse:
            aspectrum = asource.get_downstream_transmission(self.lambda_science_range) \
                            * asource.get_own_brightness(self.lambda_science_range) \
            # Collecting surface appears here
            vigneted_spectrum = self.injector.collecting \
                            * self.injector.vigneting.vigneted_spectrum(aspectrum,
                                                            self.lambda_science_range,
                                                            t_co)
            # vigneted_spectrum also handles the integration in time and space
            static_output = np.array([self.combiner.encaps(mas2rad(self.injector.vigneting.xx), mas2rad(self.injector.vigneting.yy),
                                    array.flatten(), 2*np.pi/thelambda, np.ones(self.ntelescopes, dtype=np.complex128))
                                     for m, thelambda in enumerate(self.lambda_science_range)])
            #print(static_output.shape)
            static_output = (static_output.swapaxes(0,2) * np.sqrt(vigneted_spectrum[:,None,:]))
            static_output = np.sum(np.abs(static_output*np.conjugate(static_output), dtype=dtype), axis=0)
            self.integrator.static.append(static_output.T)
        
        logit.warning("Currently no vigneting (requires a normalization of vigneting)")
        filtered_starlight = diffuse[0].get_downstream_transmission(self.lambda_science_range)
        # collected will convert from [ph / s /m^2] to [ph]
        collected = filtered_starlight * self.injector.collecting * t_co
        self.integrator.ft_phase = []
        self.integrator.inj_phase = []
        self.integrator.inj_amp = []
        logit.warning("Ugly management of injection/tracking")
        if use_tqdm:
            it_subexp = tqdm(range(self.n_subexps))
        else :
            it_subexp = range(self.n_subexps)
        for i in it_subexp:
            injected = next(self.injector.get_efunc)(self.lambda_science_range)
            tracked = next(self.fringe_tracker.phasor)
            if monitor_phase:
                self.integrator.ft_phase.append(np.angle(tracked[:,0]))
                self.integrator.inj_phase.append(np.angle(injected[:,0]))
                self.integrator.inj_amp.append(np.abs(injected[:,0]))
            injected = injected * tracked
            # lambdified argument order matters! This should remain synchronous
            # with the lambdify call
            thexx = star.xx_f
            theyy = star.yy_f
            
            combined_starlight = np.array([self.combiner.encaps(mas2rad(thexx), mas2rad(theyy),
                                    array.flatten(), 2*np.pi/thelambda, injected[:,m])
                                     for m, thelambda in enumerate(self.lambda_science_range)])
            # getting a number of photons
            combined_starlight = combined_starlight * np.sqrt(star.ss[:,None,:] * collected[:,None,None])
            combined_starlight = np.sum(np.abs(combined_starlight*np.conjugate(combined_starlight), dtype=dtype), axis=(2))
            if spectro is not None:
                self.integrator.accumulate(combined_starlight)
            
            combined_planetlight = np.array([self.combiner.encaps(mas2rad(interest.xx_f), mas2rad(interest.yy_f),
                                    array.flatten(), 2*np.pi/thelambda, injected[:,m])
                                     for m, thelambda in enumerate(self.lambda_science_range)])
            # getting a number of photons
            combined_planetlight = combined_planetlight * np.sqrt(interest.ss[:,None,:] * collected[:,None,None])
            combined_planetlight = np.sum(np.abs(combined_planetlight*np.conjugate(combined_planetlight), dtype=dtype), axis=(2))
            if spectro is not None:
                self.integrator.accumulate(combined_planetlight)
            
            # incoherently combining over sources
            # Warning: modifying the array
            # combined = np.sum(np.abs(combined*np.conjugate(combined)), axis=(2))
            # self.integrator.accumulate(combined)
            for k, astatic in enumerate(self.integrator.static):
                self.integrator.accumulate(astatic)
        
        self.integrator.static_list = []
        for k, astatic in enumerate(self.integrator.static):
            self.integrator.static_list.append(diffuse[k].__name__)
            
        #mean, std = self.integrator.compute_stats()
        self.integrator.ft_phase = np.array(self.integrator.ft_phase).astype(dtype)
        self.integrator.inj_phase = np.array(self.integrator.inj_phase).astype(dtype)
        self.integrator.inj_amp = np.array(self.integrator.inj_amp).astype(dtype)
        return self.integrator
    
    def prepare_sequence(self, file=None, times=None, remove_daytime=False):
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
        n_points = file.getint("target", "n_points")
        self.sequence = self.obs.build_observing_sequence(times=[self.seq_start_string, self.seq_end_string],
                            npoints=n_points, remove_daytime=remove_daytime)
        self.target = sf.observatory.astroplan.FixedTarget.from_name(self.tarname)
            
        pass
    def make_sequence(self):
        """
        Run an observing sequence
        """
        pass
    def build_all_maps(self, mapres=200, mapcrop=1.,
                       dtype=np.float32):
        """
        Builds the transmission maps for the combiner for all the pointings
        on self.target at the times of self.target
        mapres        : The resolution of the map in pixels
        mapcrop       : Adjusts the extent of the map
        Returns (also stored in self.map) a transmission map of shape:
        [n_sequence,
        n_wl_chanels,
        n_outputs,
        mapres,
        mapres]
        
        To get final flux of a point source :
        Map/ds_sr * p.ss * DIT
        ds_sr can be found in director.vigneting_map
        """
        vigneting_map = sf.injection.injection_vigneting(self.injector, mapres, mapcrop)
        # Warning: this vigneting map for the maps are in the simulator.vigneting_map
        # not to be confused with simulator.injector.vigneting
        self.vigneting_map = vigneting_map
        maps = []
        for i, time in enumerate(self.sequence):
            self.obs.point(self.sequence[i], self.target)
            amap = self.make_map(i, vigneting_map, dtype=dtype)
            maps.append(amap)
        maps = np.array(maps)
        self.mapshape = (len(self.sequence),
                    self.lambda_science_range.shape[0],
                    maps.shape[2],
                    mapres,
                    mapres)
        self.maps = maps.reshape(self.mapshape)
        extent = [np.min(vigneting_map.xx),
                  np.max(vigneting_map.xx),
                  np.min(vigneting_map.yy),
                  np.max(vigneting_map.yy)]
        self.map_extent = extent
        
        
    def make_map(self, blockindex, vigneting_map, dtype=np.float32):
        """
        Create sensitivity map in ph/s/sr per spectral channel.
        To get final flux of a point source :
        Map/ds_sr * p.ss * DIT
        blockindex : The index in the observing sequence to create the map
        vigneting_map : The vigneting map drawn used for resolution
        """
        self.obs.point(self.sequence[blockindex], self.target)
        array = self.obs.get_projected_array(self.obs.altaz, PA=self.obs.PA)
        #injected = self.injector.best_injection(self.lambda_science_range)
        vigneted_spectrum = \
                vigneting_map.vigneted_spectrum(np.ones_like(self.lambda_science_range),
                                                  self.lambda_science_range, 1.)
        # Caution: line is not idempotent!
        vigneted_spectrum = np.swapaxes(vigneted_spectrum, 0,1)
        static_output = np.array([self.combiner.encaps(mas2rad(vigneting_map.xx),
                                        mas2rad(vigneting_map.yy),
                                        array.flatten(), 2*np.pi/thelambda,
                                        np.ones(self.ntelescopes, dtype=np.complex128)) \
                                         for m, thelambda in enumerate(self.lambda_science_range)])
        static_output = static_output * np.sqrt(vigneted_spectrum[:,None,:])
        # lambdified argument order matters! This should remain synchronous
        # with the lambdify call
        # incoherently combining over sources
        # Warning: modifying the array
        combined = np.abs(static_output*np.conjugate(static_output)).astype(np.float32)
        return combined
        
    def __call__(self):
        pass
    
def test_director():
    logit.warning("hard path in the test")
    asim = simulator.from_config(fpath="/home/rlaugier/Documents/hi5/SCIFYsim/scifysim/config/default_new_4T.ini")
    asim.prepare_injector(asim.config)
    asim.prepare_combiner(asim.config)