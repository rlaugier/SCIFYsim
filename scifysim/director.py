
import kernuller
import scifysim as sf
import numpy as np
from tqdm import tqdm
import logging
from kernuller import mas2rad, rad2mas
from astropy import units
from pdb import set_trace
from pathlib import Path
from copy import deepcopy

import dask.array as da
# import jax.numpy as jp

parent = Path(__file__).parent.absolute()

logit = logging.getLogger(__name__)


class simulator(object):
    def __init__(self,file=None, fpath=None):
                 #location=None, array=None,
                #tarname="No name", tarpos=None, n_spec_ch=100):
        """
        The object meant to assemble and operate the simulator. 
        Construct the injector object from a config file
        
        **Parameters:**
        
        * file      : A pre-parsed config file. See ``parsefile`` submodule
        * fpath     : The path to a config file
        
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
        raw_array, array_config = sf.utilities.get_raw_array(self.config)
        self.array_config = array_config; del array_config
        if self.location == "space": # At this point we only want the "statlocs"
            raw_array = raw_array[0,:,:]
            self.space = True
        else:
            self.space = False
        
        self.order = self.config.getarray("configuration", "order").astype(np.int16)
        self.array = raw_array[self.order]
        self.n_spec_ch = self.config.getint("photon", "n_spectral_science")
        
        self.multi_dish = self.config.getboolean("configuration", "multi_dish")
        
        self.transverse_dispersion = True
        self.longitudinal_dispersion = True
        

        #self.obs = sf.observatory.observatory(self.array, location=location)
        
        self.sequence = None
        
    def prepare_observatory(self, file=None, fpath=None, statlocs=None):
        """
        Preprare the observatory object for the simulator.
        
        **Parameters:**
        
        * file : A parsed config file (see parsefile)
        * fpath: (string) A path to the config file to parse
        * statlocs : The station locations (optional)
          (east, north) for each aperture shape is (Na, 2)
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
        elif "coords" in theconfig.get("target", "mode"):
            self.tarname = theconfig.get("target", "target")
            mycoords = sf.observatory.SkyCoord(self.tarname, unit=units.deg)
            self.target = sf.observatory.astroplan.FixedTarget(mycoords)
        else :
            logit.error("Did not understand the mode for target")
            raise ValueError("Provide a valid mode for the target creation")
        if theconfig.get("configuration", "location") == "space":
            self.obs = sf.observatory.SpaceObservatory(config=theconfig)
            self.obs.wet_atmo = None
        else:
            if self.multi_dish:
                self.obs = sf.observatory.observatory(config=theconfig, statlocs=statlocs)
            elif self.multi_dish is False:
                self.obs = sf.observatory.ObservatoryAltAz(config=theconfig, statlocs=statlocs)
            # This is the true atmospheric model
            self.obs.wet_atmo = sf.wet_atmo(self.config)
        

    
    
    
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
                 crop = 0.7,
                 injector=None, seed=None):
        """
        **Either:**
        
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
        self.k = 2*np.pi/self.lambda_science_range
        self.R = self.lambda_science_range / np.gradient(self.lambda_science_range)
        
        # Peparing the vigneting function for the field of view and diffuse sources
        self.injector.vigneting = sf.injection.injection_vigneting(self.injector, res, crop)
        
        pass
    
    
    def prepare_combiner(self, file, **kwargs):
        """
        Constructs ``self.combiner``
        
        **Parameters:**
        
        * file : A parsed config file (see parsefile)
        * **kwargs to pass to the combiner ``__init__`` method
        
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
        
        * file     : A parsed config file (default: None will reuse the config file in self.)
        """
        if file is None:
            file = self.config
        self.src = sf.sources.star_planet_target(file, self)
    def prepare_integrator(self, config=None, keepall=False,
                           n_sources=4, infinite_well=False):
        """
        Prepares the integraro object that rules the pixel properties
        of the detector.
        
        **Parameters:**
        
        * config    : A parsed config file (default: None will reuse the config file in self.)
        * keepall   : [boolean] Whether to keep each of the steps accumulated.
          False is recommended for faster computation and memory efficiency
        * n_sources : Number of sources sequencially accumulated. (deprecated, do not
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
        
        **Parameters:**
        
        * file     : A parsed config file (default: None will reuse the config file in self.)
        * n_chan   : The number of outputs from the chip. If None: tries to get it from
          the shape of ``self.combiner.M``
        """
        if config is None:
            config = self.config
        if n_chan is None:
            n_chan = self.combiner.M.shape[0]
        self.spectro = sf.spectrograph.spectrograph(config,
                                                 self.lambda_science_range,
                                                 n_chan=n_chan)
        
    def prepare_corrector(self, config=None, optimize=True,
                         model_air=None):
        """
        Prepare the corrector object for the simulator, based
        on a config file.
        
        **Parameters:**
        
        * config   : Either:
        
                    - None (default) to use the simulators config
                    - A parsed config file
                    
        * optimize : Boolean. If True, will optimize both depth and shape
        * apply    : Boolean. If True, apply the optimization 
        * model_air: Provide a model for delay line air different from the true air
          if None (default) the model will be exactly the properties of air.
        """
        if config is None:
            config = self.config
        myair = sf.wet_atmo(config)
        if model_air is None:
            mymodel = deepcopy(myair)
        else:
            mymodel = model_air
            
        if config.get("corrector","mode") == "znse":
            print("----------------------------------------")
            print("Switching to znse")
            logit.error("Correcting with only glass")
            glass_file = config.get("corrector", "glass_file")
            self.corrector = sf.correctors.corrector(config,
                                                 self.lambda_science_range,
                                                model_comp=myair,
                                                file=glass_file)
        elif config.get("corrector","mode") == "znse_co2":
            print("----------------------------------------")
            print("Switching to znse+co2")
            logit.error("Correcting with only glass and CO2")
            co2_for_compensation = sf.wet_atmo(temp=myair.temp, pres=myair.pres,
                                               rhum=0., co2=1.0e6)
            glass_file = config.get("corrector", "glass_file")
            self.corrector = sf.correctors.corrector(config,
                                                 self.lambda_science_range,
                                                model_comp=myair,
                                                model_material2=co2_for_compensation,
                                                file=glass_file)
        
        elif config.get("corrector","mode") == "dispersed":
            raise NotImplementedError("Dispersed not implemented")
        else:
            raise NotImplementedError("Did not understand the corrector type")
            
        if optimize is not False:
            # First tune the null depth, then the shape parameter
            asol = self.corrector.tune_static(self.lambda_science_range,
                                              combiner=self.combiner, apply=optimize)
            # Usage of sync_params to preserve a fixed difference (the one already existing) between
            # two of the parameters, preserving the adjustment made previously.
            sol = self.corrector.tune_static_shape(self.lambda_science_range,
                             self.combiner,
                             sync_params=[("b3", "b2", self.corrector.b[3] - self.corrector.b[2]),
                                         ("c3", "c2", self.corrector.c[3] - self.corrector.c[2])],
                             apply=True)
        
        ft_wavelengths = config.getarray("fringe tracker", "wl_ft")
        wl_ft = np.linspace(ft_wavelengths[0], ft_wavelengths[-1], 6)
        self.offband_model = sf.correctors.offband_ft(wl_ft, self.lambda_science_range,
                                                      wa_true=myair, wa_model=mymodel,
                                                     corrector=self.corrector)

    def backup_ldc(self):
        self.bu_corrector = deepcopy(self.corrector)
        self.bu_offband = deepcopy(self.offband_model)

    def restore_ldc(self):
        self.corrector = deepcopy(self.bu_corrector)
        self.offband_model = deepcopy(self.bu_offband)
        
    def point(self, time, target, refresh_array=False, disp_override=None,
                    long_disp_override=None, ld_mode_override=None,
                    ft_mode="phase"):
        """
        Points the array towards the target. Updates the combiner
        
        **Parameters:**
        
        * time    : The time to make the observation
        * target  : The skycoord object to point
        * refresh_array : Whether to call a lambdification of the array
        * disp_override : None (default) follows the value given by self.transverse_dispersion;
          True force the transverse dispersion;
          False deactivate transverse dispersion.
        """
        # Figuring out what to do for dispersion
        if self.space:
            disp_override = False
            long_disp_override = False
            
        if disp_override is None:
            disp = self.transverse_dispersion
        elif disp_override is True:
            disp = True
        elif disp_override is False:
            disp = False
        if long_disp_override is None:
            long_disp = self.longitudinal_dispersion
        elif long_disp_override is True:
            long_disp = True
        elif long_disp_override is False:
            long_disp = False
        if ld_mode_override is None:
            ld_mode = "ideal"
        else:
            ld_mode = ld_mode_override
        
        self.obs.point(time, target)
        self.reset_static()
        thearray = self.obs.get_projected_array()
        if refresh_array:
            self.combiner.refresh_array(thearray)
        
        
        if not self.space:
            # Handling transverse dispersion
            altaz = self.obs.altaz
            zenith_angle = (np.pi/2)*units.rad - altaz.alt.to(units.rad)
        zero_screen = np.zeros_like(self.injector.focal_plane[0][0].screen_bias).flatten()
        for i in range(len(self.injector.focal_plane)):
            if disp:
                a = self.injector.focal_plane[i][0]
                # Refreshing the asmospheric dispersion
                Xs = np.linspace(-a.pdiam/2, a.pdiam/2, a.csz)
                Ys = np.linspace(-a.pdiam/2, a.pdiam/2, a.csz)
                XX, YY = np.meshgrid(Xs, Ys)

                #print("zenith_angle", zenith_angle.to(units.deg))
                ppixel_pistons = YY * np.tan(zenith_angle.value)

                # Note: we do not account for the main image offset here
                injwl = self.injector.lambda_range
                #import pdb
                #pdb.set_trace()
                pup_phases = self.corrector.theoretical_phase(injwl, ppixel_pistons.flatten(),
                            model=self.obs.wet_atmo,
                            add=0,
                            db=False,
                            ref="center").reshape((injwl.shape[0], *ppixel_pistons.shape))
                pup_op = pup_phases*injwl[:,None,None]/(2*np.pi)
            for j in range(len(self.injector.focal_plane[i])):
                if disp:
                    self.injector.focal_plane[i][j].screen_bias = pup_op[j,:,:].flatten()*1e6
                    # import pdb
                    # pdb.set_trace()
                else:
                    self.injector.focal_plane[i][j].screen_bias = zero_screen
                    
            # Handling longitudinal dispersion
        self.pistons = self.obs.get_projected_geometric_pistons()
        if long_disp:
            # For faster computation:
            # self.ph_disp = self.offband_model.get_ft_correction_on_science(self.pistons)
            logit.debug("Pointing including longitudinal dispersion")
            self.ph_disp = self.offband_model.get_phase_science_values(self.pistons,
                                                                        mode=ld_mode,
                                                                        ft_mode=ft_mode)
            self.phasor_disp = np.exp(1j*self.ph_disp)
        else:
            assert self.pistons.shape == (self.ntelescopes,1), f"pistons shape {self.pistons.shape}"
            logit.debug("Pointing with no longitudinal dispersion")
            self.ph_disp = np.zeros_like(self.offband_model.get_phase_science_values(self.pistons,
                                                                                    mode=ld_mode,
                                                                                    ft_mode=ft_mode))            
            self.phasor_disp = np.exp(1j*self.ph_disp)   


    def make_metrologic_exposure(self, interest, star, diffuse,
                                 int_sources=None,
                                 texp=1., t_co=2.0e-3, time=None,
                                 monitor_phase=True, dtype=np.float32,
                                 perfect=False):
        """
        Simulate an exposure with source of interest and sources of noise
        
        .. Warning::
        
            At the moment, this assumes pointing has already been performed.
        
        
        
        **Parameters:**
        
        * interest  : sf.sources.resolved_source object representing the source of intereest (planet)
        * star      : sf.sources.resolved_source object representing the star
        * diffuse   : a list of sf.transmission_emission objects linked in a chain.
        * texp      : Exposure time (seconds)
        * t_co      : Coherence time (seconds) 
        * monitor_phase : If true, will also record values of phase errors from injection and fringe tracking
        
        Chained diffuse objects are used both for transmission and thermal background. 
        Result is returned as an integrator object. Currently the integrator object is only a
        vessel for the individual subexposures, and not used to do the integration.
        
        **Returns** ``self.integrator``
        """
        t_co = self.injector.screen[0].step_time
        self.n_subexps = int(texp/t_co)
        self.integrator.n_subexps = self.n_subexps
        self.integrator.t_co = t_co
        if not hasattr(self.integrator, "cold_bg"):
            self.integrator.update_enclosure(self.lambda_science_range)
        #taraltaz = self.obs.observatory_location.altaz(time, target=self.target)
        #taraltaz, tarPA = self.obs.get_position(self.target, time)
        
        #Pointing should be done already
        array = self.obs.get_projected_array()
        
        self.integrator.static_xx_r = mas2rad(self.injector.vigneting.xx)
        self.integrator.static_yy_r = mas2rad(self.injector.vigneting.yy)
        if int_sources is not None:
            ## Preparint the post-filtering source
            if not hasattr(int_sources[0], "dflux"):
                for asource in int_sources:
                    # Compute the flux from the source in ph/sr per intergration subframe t_co
                    asource.dflux = t_co * asource.get_total_emission(self.lambda_science_range, bright=True)
                    asource.ss = asource.dflux[:,None]
                
        ## Preparing the sources
        if not hasattr(diffuse[0], "xx_r"):
            for asource in diffuse:
                asource.xx_r = self.integrator.static_xx_r
                asource.yy_r = self.integrator.static_yy_r
                aspectrum = asource.get_downstream_transmission(self.lambda_science_range, inclusive=False) \
                                * asource.get_own_brightness(self.lambda_science_range) \
                # Collecting surface appears here
                vigneted_spectrum = self.injector.vigneting.vigneted_spectrum(aspectrum,
                                                                self.lambda_science_range,
                                                                t_co)
                asource.ss = vigneted_spectrum.T
        perfect_injection = np.ones((self.lambda_science_range.shape[0], self.ntelescopes))
        dummy_collected = np.ones(self.lambda_science_range.shape[0])
        self.integrator.reset()
        self.integrator.static =  []
        self.integrator.static_list = []
        for asource in diffuse:
            # vigneted_spectrum also handles the integration in time and space
            static_output = self.combine_light(asource, perfect_injection, array, dummy_collected)
            self.integrator.static.append(static_output)
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
        
        for i in tqdm(range(self.n_subexps)):
            injected = self.phasor_disp.T * next(self.injector.get_efunc)(self.lambda_science_range)
            tracked = next(self.fringe_tracker.phasor)
            if perfect:
                injected = self.injector.best_injection(self.lambda_science_range)
                tracked = np.ones_like(tracked)
            if monitor_phase:
                # Here we take it only at the shortest wl
                self.integrator.ft_phase.append(np.angle(tracked[:,0]))
                self.integrator.inj_phase.append(np.angle(injected[:,0]))
                self.integrator.inj_amp.append(np.abs(injected[:,0]))
            injected = (injected * tracked).T * self.corrector.get_phasor(self.lambda_science_range)
            combined_starlight = self.combine_light(star, injected, array, collected)
            self.integrator.starlight.append(combined_starlight)
            combined_planetlight = self.combine_light(interest, injected, array, collected)
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
        if perfect:
            logit.warning("Idealized injection")
            print("### WARNING: idealized injection used")
            print("mean phasor : ", np.mean(injected.T,axis=1))
        return self.integrator
    def combine_light(self, asource, injected, input_array, collected,
                      dosum=True, dtype=np.float64):
        """
        Computes the combination for a discretized light source object 
        for all the wavelengths of interest.
        Returns the intensity in all outputs at all wavelengths.
        
        **Parameters:**
        
        * asource     : Source object or transmission_emission object
          including ss and xx_r attributes
        * injected    : complex phasor for the instrumental errors
        * input_array  : The projected geometric configuration of the array
          use observatory.get_projected_array()
        * collected   : The intensity across wavelength and the difference source origins
        * dtype       : The data type to use (use np.float32 for maps)
        
        """
        # Ideally, this collected operation should be factorized over all subexps
        intensity = asource.ss * collected[:,None]
        amplitude = np.sqrt(intensity)
        b = []
        if dtype is np.float64:
            for alpha, beta, anamp in zip(asource.xx_r, asource.yy_r, amplitude.T):
                geom_phasor = self.geometric_phasor(alpha, beta, input_array)
                c = self.combine(injected, geom_phasor, amplitude=anamp)
                b.append(c)
        else:
            for alpha, beta, anamp in zip(asource.xx_r, asource.yy_r, amplitude.T):
                geom_phasor = self.geometric_phasor(alpha, beta, input_array)
                c = self.combine_32(injected, geom_phasor, amplitude=anamp)
                b.append(c)
        b = np.array(b)
        if dosum:
            outputs = np.sum(np.abs(b)**2, axis=0)
        else:
            #set_trace()
            outputs = np.abs(b)**2
        return outputs
    
    def combine_light_dask(self, asource, injected, input_array, collected,
                      dosum=True, map_index=0):
        """
        Computes the combination for a discretized light source object 
        for all the wavelengths of interest. The computation is done 
        out of core using dask.
        Returns the intensity in all outputs at all wavelengths.
        
        **inputs:**
        
        * asource     : Source object or transmission_emission object
          including ss and xx_r attributes as dask arrays
        * injected    : complex phasor for the instrumental errors
        * input_array  : The projected geometric configuration of the array
          use observatory.get_projected_array()
        * collected   : Dask version of the intensity across wavelength
          and the difference source origins
        * map_index   : The index of the pointing in the sequence. Mostly use for numbering
                        of the temporary disk file
        
        .. note:: In "dask" mode the `collected` and `source.ss` are expected as dask arrays
        
        """
        # Ideally, this collected operation should be factorized over all subexps
        intensity = asource.ss * collected[:,None]
        amplitude = np.sqrt(intensity)
        b = []

        #xx_r = da.from_array(asource.xx_r)
        #yy_r = da.from_array(asource.yy_r)
        #damplitude = da.from_array(amplitude)
        geometric_phasor = self.geometric_phasor_dask(asource.xx_r,
                                                      asource.yy_r,
                                                      input_array)
        #print("Computing and writing geometric_phasor", flush=True)
        da.to_zarr(geometric_phasor , f"/tmp/full_geometric_phasor_{map_index}.zarr", overwrite=True)
        #print("Done", flush=True)
        #print("Loading", flush=True)
        geometric_phasor =  da.from_zarr(f"/tmp/full_geometric_phasor_{map_index}.zarr")
        #print("Done", flush=True)
        myinput = injected[None,:,:] * geometric_phasor
        myinput = myinput * amplitude.T[:,:,None]
        b = da.einsum("w o i, f w i -> f w o", self.combiner.Mcn, myinput)
        if dosum:
            outputs = np.sum(np.abs(b)**2, axis=0)
        else:
            outputs = np.abs(b)**2
        return outputs

    def combine_light_dask2(self, asource, injected, input_array, collected,
                      dosum=True, map_index=0):
        """
        Computes the combination for a discretized light source object 
        for all the wavelengths of interest. The computation is done 
        out of core using dask.
        Returns the intensity in all outputs at all wavelengths.
        
        **inputs:**
        
        * asource     : Source object or transmission_emission object
          including ss and xx_r attributes as dask arrays
        * injected    : complex phasor for the instrumental errors
        * input_array  : The projected geometric configuration of the array
          use observatory.get_projected_array()
        * collected   : Dask version of the intensity across wavelength
          and the difference source origins
        * map_index   : The index of the pointing in the sequence. Mostly use for numbering
                        of the temporary disk file
        
        .. note:: In "dask" mode the `collected` and `source.ss` are expected as dask arrays
        
        """
        # Ideally, this collected operation should be factorized over all subexps
        intensity = asource.ss * collected[:,None]
        amplitude = np.sqrt(intensity)
        b = []

        geometric_phasor = self.geometric_phasor_dask(asource.xx_r,
                                                      asource.yy_r,
                                                      input_array)
        # da.to_zarr(geometric_phasor , f"/tmp/full_geometric_phasor_{map_index}.zarr", overwrite=True)
        # geometric_phasor =  da.from_zarr(f"/tmp/full_geometric_phasor_{map_index}.zarr")
        myinput = injected[None,:,:] * geometric_phasor
        myinput = myinput * amplitude.T[:,:,None]
        b = da.einsum("w o i, f w i -> f w o", self.combiner.Mcn, myinput)
        if dosum:
            outputs = np.sum(np.abs(b)**2, axis=0)
        else:
            outputs = np.abs(b)**2
        return outputs

    def combine_32(self, inst_phasor, geom_phasor, amplitude=None,):
        """
        Computes the output INTENSITY based on the input AMPLITUDE
        This version provides a result in np.float32 format for smaller
        memory footprint (for maps)
        
        **Parameters:**
        
        * inst_phasor   : The instrumental phasor to apply to the inputs
          dimension (n_wl, n_input)
        * geom_phasor   : The geometric phasor due to source location
          dimension (n_wl, n_input)
        * amplitude     : Complex amplitude of the spectrum of the source
          dimension (n_wl, n_src)
        
        **Returns** Output complex amplitudes
        
        """
        #from pdb import set_trace
        if amplitude is None:
            amplitude = np.ones_like(self.k, dtype=np.float32)
        myinput = inst_phasor*geom_phasor*amplitude[:,None]
        output_amps = np.einsum("ijk,ik->ij",self.combiner.Mcn.astype(np.complex64), myinput)
        #set_trace()
        return output_amps
    def combine(self, inst_phasor, geom_phasor, amplitude=None):
        """
        Computes the output INTENSITY based on the input AMPLITUDE
        or maps)
        
        **Parameters:**
        
        * inst_phasor   : The instrumental phasor to apply to the inputs
          dimension (n_wl, n_input)
        * geom_phasor   : The geometric phasor due to source location
          dimension (n_wl, n_input)
        * amplitude     : Complex amplitude of the spectrum of the source
          dimension (n_wl, n_src)
        
        **Returns** Output complex amplitudes
        
        """
        if amplitude is None:
            amplitude = np.ones_like(self.k)
        myinput = inst_phasor*geom_phasor*amplitude[:,None]
        output_amps = np.einsum("ijk,ik->ij",self.combiner.Mcn, myinput)
        return output_amps
    
    def geometric_phasor(self, alpha, beta, anarray):
        """
        Returns the complex phasor corresponding to the locations
        of the family of sources
        
        **Parameters:**
        
        * alpha         : The coordinate matched to X in the array geometry
        * beta          : The coordinate matched to Y in the array geometry
        * anarray       : The array geometry (n_input, 2)
        
        **Returns** : A vector of complex phasors
        """
        a = np.array((alpha, beta), dtype=np.float64)
        phi = self.k[:,None] * anarray.dot(a)[None,:]
        b = np.exp(1j*phi)
        return b

    

    def geometric_phasor_dask(self, alphas, betas, anarray):
        """
        Returns the complex phasor corresponding to the locations
        of the family of sources
        
        **Parameters:**
        * alpha         : The coordinate matched to X in the array geometry
          as a dask array (field positions)
        * beta          : The coordinate matched to Y in the array geometry
          as a dask array (field positions)
        * anarray       : The array geometry (n_input, 2)
        """
        # making a dask array of field positions (nfield, 2)
        alphabeta = da.from_array((alphas, betas)).T
        k = da.from_array(self.k)
        phi = k[None, :, None] * da.einsum( "a d, f d -> f a",  anarray, alphabeta)[:,None,:]
        z = da.exp(1j*phi)
        return z

    
    def make_exposure(self, interest, star, diffuse,
                                 texp=1., t_co=2.0e-3, time=None,
                                 monitor_phase=True,
                                 use_tqdm=False,
                                 dtype=np.float32,
                                 spectro=None):
        """
        Warning: at the moment, this assumes pointing has already been performed.
        
        Simulate an exposure with source of interest and sources of noise
        
        **Parameters:**
        
        * interest  : sf.sources.resolved_source object representing the source of intereest (planet)
        * star      : sf.sources.resolved_source object representing the star
        * diffuse   : a list of sf.transmission_emission objects linked in a chain.
        * texp      : Exposure time (seconds)
        * t_co      : Coherence time (seconds) 
        * monitor_phase : If true, will also record values of phase errors from injection and
          fringe tracking
        * dtype     : Data type to use for results
        * spectro   : spectrograph object to use. If None, the method will assume one pixel 
          per output and per spectral channel
        
        Chained diffuse objects are used both for transmission and thermal background. 
        Result is returned as an integrator object. Currently the integrator object is only a
        vessel for the individual subexposures, and not used to do the integration.
        """
        t_co = self.injector.screen[0].step_time
        self.n_subexps = int(texp/t_co)
        
        #Pointing should be done already
        array = self.obs.get_projected_array()
        self.computed_static_xx = self.injector.vigneting.xx
        self.computed_static_yy = self.injector.vigneting.yy
        self.integrator.reset()
        self.integrator.n_subexps = self.n_subexps
        self.integrator.t_co = t_co
        if not hasattr(self.integrator, "cold_bg"):
            self.integrator.update_enclosure(self.lambda_science_range)
        # Check existence of pre-computed static signal: This part should remain static per pointing
        if not hasattr(self, "computed_static"): # Pointings should delete this attribute
            if not hasattr(diffuse[0], "xx_r"):
                # Populate the source.xx_r, and source.ss attributes
                self.integrator.static_xx_r = mas2rad(self.injector.vigneting.xx)
                self.integrator.static_yy_r = mas2rad(self.injector.vigneting.yy)
                for asource in diffuse:
                    asource.xx_r = self.integrator.static_xx_r
                    asource.yy_r = self.integrator.static_yy_r
                    aspectrum = asource.get_downstream_transmission(self.lambda_science_range, inclusive=False) \
                                    * asource.get_own_brightness(self.lambda_science_range) \
                    # Collecting surface appears here
                    vigneted_spectrum = self.injector.vigneting.vigneted_spectrum(aspectrum,
                                                                    self.lambda_science_range,
                                                                    t_co)
                    asource.ss = vigneted_spectrum.T
                    
            dummy_collected = np.ones(self.lambda_science_range.shape[0])
            perfect_injection = np.ones((self.lambda_science_range.shape[0], self.ntelescopes))
            
            self.computed_static =  []
            for asource in diffuse:
                static_output = self.combine_light(asource, perfect_injection, array, dummy_collected)
                self.computed_static.append(static_output)
        self.integrator.starlight = np.zeros_like(self.computed_static[0])
        self.integrator.planetlight = np.zeros_like(self.integrator.starlight)
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
            self.integrator.exposure += t_co
            injected = self.phasor_disp.T * next(self.injector.get_efunc)(self.lambda_science_range)
            tracked = next(self.fringe_tracker.phasor)
            if monitor_phase:
                self.integrator.ft_phase.append(np.angle(tracked[:,0]))
                self.integrator.inj_phase.append(np.angle(injected[:,0]))
                self.integrator.inj_amp.append(np.abs(injected[:,0]))
            injected = (injected * tracked).T * self.corrector.get_phasor(self.lambda_science_range)
            # lambdified argument order matters! This should remain synchronous
            # with the lambdify call
            combined_starlight = self.combine_light(star, injected, array, collected)
            if spectro is None:
                self.integrator.accumulate(combined_starlight)
                self.integrator.starlight += combined_starlight
            
            combined_planetlight = self.combine_light(interest, injected, array, collected)
            if spectro is None:
                self.integrator.accumulate(combined_planetlight)
                self.integrator.planetlight += combined_planetlight
            
            # incoherently combining over sources
            # Warning: modifying the array
            # combined = np.sum(np.abs(combined*np.conjugate(combined)), axis=(2))
            # self.integrator.accumulate(combined)
            for k, astatic in enumerate(self.computed_static):
                self.integrator.accumulate(astatic)
        self.integrator.static_list = []
        for k, astatic in enumerate(self.computed_static):
            self.integrator.static_list.append(diffuse[k].__name__)
            
        #mean, std = self.integrator.compute_stats()
        self.integrator.ft_phase = np.array(self.integrator.ft_phase).astype(dtype)
        self.integrator.inj_phase = np.array(self.integrator.inj_phase).astype(dtype)
        self.integrator.inj_amp = np.array(self.integrator.inj_amp).astype(dtype)
        return self.integrator
    
    def prepare_sequence(self, file=None, times=None, remove_daytime=False,
                        coordinates=None):
        """
        Prepare an observing sequence
        
        **Parameters:**
        * file : Parsed config file
        * times : deprecated
        * remove_daytimes : (*default* : False) Whether to remove from
          the sequence the daytime occurences
        * coordinates  : read from file by default
          if skycoord object provided, then use that.
        """
        if file is not None:
            logit.info("Building sequence from new config file")
            pass
        else:
            logit.info("Building sequence from main config file")
            file = self.config
        if coordinates is not None:
            file.set("target", "mode", "coords")
        
        self.seq_start_string = file.get("target", "seq_start")
        self.seq_end_string = file.get("target", "seq_end")
        n_points = file.getint("target", "n_points")
        self.sequence = self.obs.build_observing_sequence(times=[self.seq_start_string, self.seq_end_string],
                            npoints=n_points, remove_daytime=remove_daytime)
        condition = (coordinates is not None) and (not hasattr(self, "target"))
        if coordinates is None:
            self.target = sf.observatory.astroplan.FixedTarget.from_name(self.tarname)
        elif condition:
            logit.error("Forced to define target name and position. This should have been done by prepare_observatory.")
            if "name" in file.get("target", "mode"):
                self.target = sf.observatory.astroplan.FixedTarget.from_name(self.tarname)
            elif "coords" in file.get("target", "mode"):
                self.tarname = file.get("target", "target")
                self.target = sf.observatory.astroplan.FixedTarget(self.target)
            else :
                logit.error("Did not understand the mode for target")
                raise ValueError("Provide a valid mode for the target creation")
        else:
            if isinstance(coordinates, sf.observatory.SkyCoord):
                self.tarnmae = "from_skycoord"
                self.target = sf.observatory.astroplan.FixedTarget(coordinates)
            elif isinstance(coordinates, str):
                self.tarname = "from_string"
                self.target = sf.observatory.astroplan.FixedTarget(coordinates)
        
                
            
        pass
    def make_sequence(self):
        """
        Deprecated
        """
        pass
    
    
        
        
    
    def build_all_maps(self, mapres=100, mapcrop=1.,
                       dtype=np.float32, transmission="default"):
        """
        Builds the transmission maps for the combiner for all the pointings
        on self.target at the times of ``self.sequence``
        
        **Parameters:**
        
        * mapres        : The resolution of the map in pixels
        * mapcrop       : Adjusts the extent of the map
        
        **Returns** (also stored in self.map) a transmission map of shape:
        
            - [n_sequence,
            - n_wl_chanels,
            - n_outputs,
            - mapres,
            - mapres]
        
        To get final flux of a point source : ``Map/ds_sr * p.ss * DIT``
        
        ``ds_sr`` can be found in ``director.vigneting_map``
        """
        if transmission == "default":
            transmission = self.src.sky
        vigneting_map = sf.injection.injection_vigneting(self.injector, mapres, mapcrop)
        # Warning: this vigneting map for the maps are in the simulator.vigneting_map
        # not to be confused with simulator.injector.vigneting
        self.vigneting_map = vigneting_map
        self.mapsource = type('', (), {})()
        self.mapsource.xx_r = mas2rad(self.vigneting_map.xx)
        self.mapsource.yy_r = mas2rad(self.vigneting_map.yy)
        self.mapsource.rr_r = mas2rad(self.vigneting_map.rr)
        maps = []
        for i, time in enumerate(self.sequence):
            self.point(self.sequence[i], self.target)
            if transmission is None:
                local_transmission = 1.
            else:
                local_transmission = transmission.get_downstream_transmission(self.lambda_science_range)
            amap = self.make_map(i, self.vigneting_map, dtype=dtype,
                                         transmission=transmission)
            maps.append(amap)
        maps = np.array(maps)
        print(maps.shape)
        self.mapshape = (len(self.sequence),
                    self.lambda_science_range.shape[0],
                    maps.shape[2],
                    mapres,
                    mapres)
        self.maps = maps.reshape(self.mapshape)
        extent = [np.min(self.vigneting_map.xx),
                  np.max(self.vigneting_map.xx),
                  np.min(self.vigneting_map.yy),
                  np.max(self.vigneting_map.yy)]
        self.map_extent = extent

    def get_loc_map(self, loc):
        """
        Get the nearest pixel location of for a relative coordinate
        **Arguments:**
        * loc   : the relative coordinate in [mas]
         defined in (y, x) as per numpy convention

        Returns: (y, x ) coordinate index in pixel as per numpy conv.
        """
        xindex_raw = np.argmin(np.abs(self.vigneting_map.xx - loc[1]))
        yindex_raw = np.argmin(np.abs(self.vigneting_map.yy - loc[0]))
        xindex = np.unravel_index(xindex_raw,
                shape=(self.vigneting_map.resol, self.vigneting_map.resol))[1]
        yindex = np.unravel_index(yindex_raw,
                shape=(self.vigneting_map.resol, self.vigneting_map.resol))[0]
        return (yindex, xindex)
        
        
    def build_all_maps_dask(self, mapres=100, mapcrop=1.,
                       dtype="dask", transmission="default",
                        no_compute=False):
        """
        Builds the transmission maps for the combiner for all the pointings
        on self.target at the times of self.sequence.
        Returns an uncomputed dask array referencing one input map per pointing
        that are each stored to disk.
        call self.maps[element indices].compute() to compute specific elements
        without computing the whole map.
        
        **Parameters:**
        * mapres        : The resolution of the map in pixels
        * mapcrop       : Adjusts the extent of the map
        * transmission  : The first item in the transmission-emission objects
          if "default": will use self.src.sky
        
        **Returns** (also stored in self.map) a transmission map of shape:
        * [n_sequence,
        * n_wl_chanels,
        * n_outputs,
        * mapres,
        * mapres]
        
        To get final flux of a point source : ``Map/ds_sr * p.ss * DIT``
        ``ds_sr`` is the scene surface element in steradian and can be found in ``director.vigneting_map``
        """
        if transmission == "default":
            transmission = self.src.sky
        vigneting_map = sf.injection.injection_vigneting(self.injector, mapres, mapcrop)
        # Warning: this vigneting map for the maps are in the simulator.vigneting_map
        # not to be confused with simulator.injector.vigneting
        #set_trace()
        self.vigneting_map = vigneting_map
        self.mapsource = type('', (), {})()
        self.mapsource.xx_r = mas2rad(self.vigneting_map.xx)
        self.mapsource.yy_r = mas2rad(self.vigneting_map.yy)
        self.mapsource.rr_r = mas2rad(self.vigneting_map.rr)
        maps = []
        for i, time in enumerate(self.sequence):
            self.point(self.sequence[i], self.target)
            if transmission is None:
                local_transmission = 1.
            else:
                local_transmission = transmission.get_downstream_transmission(self.lambda_science_range)
            amap = self.make_map_dask(i, self.vigneting_map, dtype=dtype, map_index=i,
                                     transmission=transmission, no_compute=no_compute)
            maps.append(amap)
        maps = da.array(maps)
        print(maps.shape)
        self.mapshape = (len(self.sequence),
                    self.lambda_science_range.shape[0],
                    maps.shape[2],
                    mapres,
                    mapres)
        self.maps = maps.reshape(self.mapshape)
        extent = [np.min(self.vigneting_map.xx),
                  np.max(self.vigneting_map.xx),
                  np.min(self.vigneting_map.yy),
                  np.max(self.vigneting_map.yy)]
        self.map_extent = extent
        
    def persist_maps_to_disk(self, fname="/tmp/full_maps.zarr"):
        """
        Computes and stores the map to disk. The file is then loaded
        *out of core* and is still accessible in the same way (but without
        the computing times).
        """
        print("Computing and writing full map", flush=True)
        da.to_zarr(self.maps , fname, overwrite=True)
        print("Done", flush=True)
        print("Loading", flush=True)
        self.maps =  da.from_zarr(fname)
        print("Done", flush=True)
        fname = f"/tmp/full_geometric_phasor_*.zarr"
        print(f"The files {fname} can be removed manually")
        
        
        
    def make_map_dask(self, blockindex, vigneting_map, dtype="dask", map_index=0,
                             transmission=None, no_compute=False):
        """
        Create sensitivity map in m^2.sr per spectral channel.
        To get final flux of a point source :
        
        ``Map/ds_sr * p.ss * DIT``
        
        **Parameters:**
        
        * blockindex : The index in the observing sequence to create the map
        * vigneting_map : The vigneting map drawn used for resolution
        
        """
        self.point(self.sequence[blockindex], self.target)
        array = self.obs.get_projected_array()
        #injected = self.injector.best_injection(self.lambda_science_range)
        vigneted_spectrum = \
                vigneting_map.vigneted_spectrum(np.ones_like(self.lambda_science_range),
                                                  self.lambda_science_range, 1.,
                                               transmission=transmission)
        # Caution: line is not idempotent!
        #vigneted_spectrum = np.swapaxes(vigneted_spectrum, 0,1)
        self.mapsource.ss = da.from_array(vigneted_spectrum.T)
        #self.mapsource.ss = vigneted_spectrum
        dummy_collected = da.ones(self.lambda_science_range.shape[0])
        perfect_injection = da.ones((self.lambda_science_range.shape[0], self.ntelescopes))\
            * self.corrector.get_phasor(self.lambda_science_range)\
            * self.phasor_disp
        if no_compute:
            static_output = self.combine_light_dask2(self.mapsource, perfect_injection,
                                           array, dummy_collected,
                                           dosum=False, map_index=map_index)
        else:
            static_output = self.combine_light_dask(self.mapsource, perfect_injection,
                                           array, dummy_collected,
                                           dosum=False, map_index=map_index)
        static_output = static_output.swapaxes(0, -1)
        static_output = static_output.swapaxes(0, 1)
        #static_output = static_output# * np.sqrt(vigneted_spectrum[:,None,:])
        # lambdified argument order matters! This should remain synchronous
        # with the lambdify call
        # incoherently combining over sources
        # Warning: modifying the array
        #combined = np.abs(static_output*np.conjugate(static_output)).astype(np.float32)
        return static_output
    
    def make_map_dask2(self, blockindex, vigneting_map, dtype="dask", map_index=0):
        """
        Create sensitivity map in m^2.sr per spectral channel.
        To get final flux of a point source :
        
        ``Map/ds_sr * p.ss * DIT``
        
        **Parameters:**
        
        * blockindex : The index in the observing sequence to create the map
        * vigneting_map : The vigneting map drawn used for resolution
        
        """
        self.point(self.sequence[blockindex], self.target)
        array = self.obs.get_projected_array()
        #injected = self.injector.best_injection(self.lambda_science_range)
        vigneted_spectrum = \
                vigneting_map.vigneted_spectrum(np.ones_like(self.lambda_science_range),
                                                  self.lambda_science_range, 1.)
        # Caution: line is not idempotent!
        #vigneted_spectrum = np.swapaxes(vigneted_spectrum, 0,1)
        self.mapsource.ss = da.from_array(vigneted_spectrum.T)
        #self.mapsource.ss = vigneted_spectrum
        dummy_collected = da.ones(self.lambda_science_range.shape[0])
        perfect_injection = da.ones((self.lambda_science_range.shape[0], self.ntelescopes))\
            * self.corrector.get_phasor(self.lambda_science_range)
        static_output = self.combine_light_dask(self.mapsource, perfect_injection,
                                           array, dummy_collected,
                                           dosum=False,map_index=map_index)
        static_output = static_output.swapaxes(0, -1)
        static_output = static_output.swapaxes(0, 1)
        #static_output = static_output# * np.sqrt(vigneted_spectrum[:,None,:])
        # lambdified argument order matters! This should remain synchronous
        # with the lambdify call
        # incoherently combining over sources
        # Warning: modifying the array
        #combined = np.abs(static_output*np.conjugate(static_output)).astype(np.float32)
        return static_output
        
        
    def make_map(self, blockindex, vigneting_map, dtype=np.float32,
                transmission=None):
        """
        Create sensitivity map in m^2.sr per spectral channel.
        To get final flux of a point source :
        ``Map/ds_sr * p.ss * DIT``
        
        **Parameters:**
        
        * blockindex : The index in the observing sequence to create the map
        * vigneting_map : The vigneting map drawn used for resolution
        
        **Returns** the ``static_output``: the map
        """
        self.point(self.sequence[blockindex], self.target)
        array = self.obs.get_projected_array()
        #injected = self.injector.best_injection(self.lambda_science_range)
        vigneted_spectrum = \
                vigneting_map.vigneted_spectrum(np.ones_like(self.lambda_science_range),
                                                  self.lambda_science_range, 1.,
                                               transmission=transmission)
        # Caution: line is not idempotent!
        #vigneted_spectrum = np.swapaxes(vigneted_spectrum, 0,1)
        self.mapsource.ss = vigneted_spectrum.T
        #self.mapsource.ss = vigneted_spectrum
        dummy_collected = np.ones(self.lambda_science_range.shape[0])
        perfect_injection = np.ones((self.lambda_science_range.shape[0], self.ntelescopes))\
            * self.corrector.get_phasor(self.lambda_science_range)\
            * self.phasor_disp
        static_output = self.combine_light(self.mapsource, perfect_injection,
                                           array, dummy_collected,
                                           dtype=dtype,
                                           dosum=False)
        static_output = static_output.swapaxes(0, -1)
        static_output = static_output.swapaxes(0, 1)
        #static_output = static_output# * np.sqrt(vigneted_spectrum[:,None,:])
        # lambdifiked argument order matters! This should remain synchronous
        # with the lambdify call
        # incoherently combining over sources
        # Warning: modifying the array
        #combined = np.abs(static_output*np.conjugate(static_output)).astype(np.float32)
        return static_output

    @property
    def gain_map(self):
        ds_sr = self.vigneting_map.ds_sr
        # Since maps contain eta, their unti is in electron/photon * m**2 * sr
        mapunits = units.electron / units.photon * units.m**2 * units.sr
        throughput_map = 1/(ds_sr*units.sr) * self.maps*mapunits 
        return throughput_map

        
    def __call__(self):
        pass
    def reset_static(self):
        if hasattr(self, "computed_static"):
            del self.computed_static
    
    @property
    def spectral_res(self):
        """
        The spectral resolution of the instrument. Computed as a mean over all spectral channels.
        """
        res = np.mean(sf.utilities.range2R(self.lambda_science_range, mode="bins"))
        return res
    
def test_director():
    logit.warning("hard path in the test")
    asim = simulator.from_config(fpath="/home/rlaugier/Documents/hi5/SCIFYsim/scifysim/config/default_new_4T.ini")
    asim.prepare_injector(asim.config)
    asim.prepare_combiner(asim.config)