import sympy as sp
import numpy as np
from kernuller import mas2rad
from kernuller import rad2mas
from kernuller import fprint

import logging

logit = logging.getLogger(__name__)

def vec2diag(vec):
    """
    Replictes in sympy the np.diag functionnality to create a
    diagonal matrix from a 1D vector
    """
    thelen = vec.shape[0]
    A = sp.eye(thelen)
    for i in range(thelen):
        A[i,i]= vec[i]
    return A

def range2R(spectral_range, mode="bins"):
    """
    Convenience function to convert a sequence of wavelengths to the
    target spectral resolution.
    
    **Arguments:**
    
    * spectral_range : (array-like) The series of wavelength channels [m]
    * mode : Decides what is considered a step in the series.
    
        - "bins" : The simulator wl bins as found in
          ``simulator.lambda_science_range``:
          2 pixels per bin in Hi-5
        - "chans" : The individual wavelength step
        - "pixs" : The individual pixels (2.5 pixel per spectral channel)
    
    **Returns** an array of the individual spectral resolutions
    """
    if mode == "chans":
        deltalamb = np.gradient(spectral_range)
    elif mode == "bins":
        deltalamb = np.gradient(spectral_range)/2*2.5
    elif mode == "pixs":
        deltalamb = np.gradient(spectral_range)*2.5
    R = spectral_range/deltalamb
    return R


def lambdifyz(symbols, expr, modules="numpy"):
    """
    Circumvents a bug in lambdify where silent 
    variables will be simplified and therefore
    aren't broadcasted. `<https://github.com/sympy/sympy/issues/5642>`_
    
    Use an extra argument = 0 when calling
    
    Please, keep the function synchronous with kernuller
    """
    assert isinstance(expr, sp.Matrix)
    z = sp.symbols("z")
    thesymbols = list(symbols)
    thesymbols.append(z)
    exprz = expr + z*sp.prod(symbols)*sp.ones(expr.shape[0], expr.shape[1])
    fz = sp.lambdify(thesymbols, exprz, modules=modules)
    return fz



class ee(object):
    """
    Additional functionnality needed:
    see comments starting 2021/02/24 `<https://github.com/sympy/sympy/issues/5642>`_
    """
    def __init__(self, expression):
        """
        ee is for Executable expression
        Encapsulates an expression for flexible lambda evaluation.
        expression  : a sympy expression to lambdify
        """
        self.expr = expression
        if self.expr.is_Matrix:
            self.outshape = self.expr.shape
            self.outlen = np.prod(self.outshape)
            #self.callfunction = self.setup_matrixfunction()
        else :
            self.outlen = 1
            self.outshape = (1)
        
    def lambdify(self, args, modules="numexpr"):
        """
        Creates the lambda function. Currently, lambdification to numexpr
        does not support multiple outputs, so we create a list of lambda functions.
        
        **Parameters:**
        
        * args    : a tuple of sympy for symbols to use as inputs
        * modules : The module to use for computation.
        """
        #Here, we have to decide if we need the function to all 
        if ("numexpr" in modules) and (self.outlen != 1):
            thefuncs = []
            for i in range(self.outlen):
                thefuncs.append(sp.lambdify(args, sp.flatten(self.expr)[i], modules=modules))
            self.funcs = thefuncs
            self.callfunction = self.numexpr_call
            
        else :
            self.funcs = sp.lambdify(args, self.expr, modules=modules)
            self.callfunction = self.numpy_call
            
            
    def numexpr_call(self,*args):
        """
        The evaluation call for funcions for the numexpr case
        Evaluating the list of functions and returning it as an array
        """
        return np.stack([np.asarray(self.funcs[i](*args)) for i in range(self.outlen)])
    
    def numpy_call(self, *args):
        """
        Just the normal call of the lambda function
        """
        return self.funcs(*args) # Here, we flatten for consitency?
    
    def __call__(self,*args):
        return self.callfunction(*args)
    def fprint(self):
        fprint(self.expr)
    

def prepare_all(afile, thetarget=None, update_params=False,
               instrumental_errors=True, seed=None,
               crop=1., target_coords=None,
               compensate_chromatic=True,
               modificators=None):
    """
    A shortcut to prepare a simulator object
    **Parameters:**
    
    - afile      : Config file 
    - thetarget  : target name to replace in config file
    - instrumental_errors : Bool. Whether to include instrumental
      errors
    - seed       : seed to pass when generating the 
      time series of instrumental errors
    - crop       : factor to pass to the injector
    - target_coords : Coordinates of the target 
      default:None : finds the target from catalog
    - compensate_chromatic : Bool Whether to run the optimization
      of the actuators to compensate for chromaticity
      in the combiner
    - modificators : a dict of parameters to supersede the config file
    
    **Returns:**
    
    - asim        : A simulator object
    """
    from scifysim import director
    if isinstance(afile, str):
        asim = director.simulator(fpath=afile)
    else:
        asim = director.simulator(file=afile)
    #asim.config = loaded_config
    if thetarget is not None:
        asim.config.set("target", "target", value=thetarget)
    if update_params:
        update_star_params(config=asim.config)
    update_observing_night(config=asim.config, target_coords=target_coords)
    asim.prepare_observatory(file=asim.config)
    if modificators is not None:
        for amod in modificators:
            asim.config.set(amod["section"], amod["key"], amod["value"])
    if not instrumental_errors:
        asim.config.set("fringe tracker", "dry_scaling", "0.0001")
        asim.config.set("fringe tracker", "wet_scaling", "0.0001")
        asim.config.set("atmo", "correc", "300.")
    asim.prepare_injector(file=asim.config, seed=seed, crop=crop)
    asim.prepare_combiner(asim.config)
    asim.prepare_sequence(asim.config)
    asim.prepare_fringe_tracker(asim.config, seed=seed)
    asim.fringe_tracker.prepare_time_series(asim.lambda_science_range, duration=10, replace=True)
    asim.prepare_integrator(config=asim.config, keepall=False, infinite_well=True)
    asim.prepare_spectrograph(config=asim.config)
    asim.prepare_sources()
    highest = len(asim.sequence)//2
    asim.obs.point(asim.sequence[highest], asim.target)
    asim.reset_static()
    asim.combiner.chromatic_matrix(asim.lambda_science_range)
    asim.prepare_corrector(optimize=compensate_chromatic)
    return asim


import itertools
def get_uv(puparray):
    """
    Computes the uv baselines for a given pupil configuration.
    
    **Parameters:**
    
    * puparray    : An array representing the location of each pupil
    """
    uv = []
    for pair in itertools.combinations(puparray, 2):
        #print(pair)
        uv.append(pair[0]-pair[1])
    uv = np.array(uv)
    indices = []
    for pair in itertools.combinations(np.arange(puparray.shape[0]), 2):
        #print(pair)
        indices.append((pair[0],pair[1]))
    indices = np.array(indices)
    return uv, indices
    

def test_ex():
    x, y = sp.symbols("x y")
    f1 = x**2 + y**2
    objf1 = ee(f1)
    objf1.fprint()
    
    objf1.lambdify((x, y), modules="numpy")
    b = objf1(10.,11.)
    print("Result with numpy :\n", b)
    
    objf1.lambdify((x, y), modules="numexpr")
    b = objf1(10.,11.)
    print("Result with numexpr :\n", b)
    
    
    f2 = sp.Matrix([[x**2 + y**2],
                    [x**2 + y**2]])
    objf2 = ee(f2)
    objf2.fprint()
    
    objf2.lambdify((x, y), modules="numpy")
    b = objf2(10.,11.)
    print("Result with numpy :\n", b)
    
    objf2.lambdify((x, y), modules="numexpr")
    b = objf2(10.,11.)
    print("Result with numexpr :\n", b)
    
    
    f3 = sp.Matrix([[x**2 + y**2, x**2 - y**2],
                    [x**2 + 2*y**2, x**2 - 2*y**2]])
    objf3 = ee(f3)
    objf3.fprint()
    
    objf3.lambdify((x, y), modules="numpy")
    b = objf3(10.,11.)
    print("Result with numpy :\n", b)
    
    objf3.lambdify((x, y), modules="numexpr")
    print(objf3.outlen)
    b = objf3(10.,11.)
    print("Result with numexpr :\n", b)
    
    print("")
    print("Broadcasting example")
    print("====================")
    
    objf2.lambdify((x, y), modules="numpy")
    b = objf2(10.*np.arange(3)[:,None], 11.*np.arange(4)[None,:])
    print("broadcas with numpy :\n", b)
    print("shape:", b.shape)
    
    objf2.lambdify((x, y), modules="numexpr")
    b = objf2(10.*np.arange(3)[:,None], 11.*np.arange(4)[None,:])
    print("broadcas with numexpr :\n", b)
    print("shape:", b.shape)
    
    return b





import cProfile, pstats, io

def profileit(func, highres=False):
    
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile" # Name the data file sensibly
        if highres:
            from time import perf_counter_ns
            prof = cProfile.Profile(perf_counter_ns, timeunit=0.000001)
        else:
            prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open(datafn, 'w') as perf_file:
            perf_file.write(s.getvalue())
        return retval

    return wrapper

def random_series_fft(ps, matchto=None, matchlength=10, keepall=False, seed=None):
    """
    Returns a new series of data based on the provided power spectrum.
    If matchto is provided, the new series is adjusted for continuity 
    to that data.
    
    **Parameters:**
    
    * ps       : A power spectrum
    * matchto  : np.ndarray of preceding series. The mean of the first
      n (matchlength) values of the new series are matched
      to the mean of the last n values of this array.
    * matchlength : integer. The number of values to match 
    * keepall  : Boolean. If true, will return the concatenation of the
      new array with the old. If False, will return only the
      new array.
    
    
    **Returns:**
    
    * a numpy ndarray of the time series
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    newphi = rng.uniform(low=-np.pi, high=np.pi, size=ps.shape[0])
    #newphi = np.random.uniform(low=-np.pi, high=+np.pi, size=ps.shape[0])
    newspectrum = np.sqrt(ps)*np.exp(1j*newphi)
    series = np.fft.ifft(newspectrum).real
    if matchto is not None:
        adj0 = matchto[-matchlength:].mean()
        adj1 = series[:matchlength].mean()
        series = series - adj1 + adj0
        if keepall:
            return np.concatenate((matchto, series))
        else:
            return series
    else:
        return series
    
def matchseries(a, b, nmatch=10):
    """
    Matches series b with series a with the nmatch last samples of a and
    the nmatch first samples of b.
    
    
    * a        : 1D array the root of the series
    * b        : 1D array the tail of the series 
    * nmatch   : integer  The number of samples used to match the mean 
    
    **Returns** the concatenation.
    """
    adj0 = a[-nmatch:].mean()
    adj1 = b[:nmatch].mean()
    b2 = b - adj1 + adj0
    return np.concatenate((a,b2))
        



def test_random_series(seed=None):
    
    import matplotlib.pyplot as plt
    # Creating an initial time series with an interesting spectrum.
    s = 20
    a = np.random.uniform(low=0, high=1., size=200)
    b = np.convolve(a, np.ones(s)/s,mode="full")
    # Extracting the power spectrum
    
    psb = np.abs(np.fft.fft(b))**2
    
    # Looping to concatenate time series
    bnew = None
    for i in range(3):
        if seed is not None:
            seed = seed+1
        bnew = random_series_fft(psb, matchto=bnew, keepall=True, seed=seed)

    plt.figure()
    plt.plot(b, label="Original")
    plt.plot(bnew, label="Random")
    plt.legend()
    plt.title("Original and random time series")
    plt.xlabel("Time [arbitrary]")
    plt.show()
    #print(np.fft.fftfreq(b.shape[0]))
    freqs = np.fft.fftfreq(b.shape[0])
    freqsnew = np.fft.fftfreq(bnew.shape[0],)
    plt.figure()
    plt.plot(freqs, psb, label="Original")
    plt.plot(freqsnew, np.abs(np.fft.fft(bnew))**2, label="New")
    plt.yscale("log")
    plt.xlim(-0.2, 0.2)
    plt.legend()
    plt.title("Power spectrum")
    plt.xlabel("Frequency")
    plt.show()

    
# Function courtesy of Philippe Berio (OCA)
# this function returns a vector containing the timestaps of the GRAVITY measurements,
# an array containing the group delay and an array containing the phase delay [nFrame,nBase]
def loadGRA4MAT(filename):
    from astropy.time import Time
    from astropy.io import fits
    hdu = fits.open(filename)
    acqstart = hdu[0].header['HIERARCH ESO PCR ACQ START']
    t = Time(acqstart)
    time = t.mjd + hdu['IMAGING_DATA_FT'].data['TIME'] * 1E-6 / 86400.
    time = time *24*3600
    #gd = hdu['IMAGING_DATA_FT'].data['GD'] * 2.15 * 25 / (2. * np.pi)
    #pd = hdu['IMAGING_DATA_FT'].data['PD'] * 2.15 / (2. * np.pi)
    coherence = hdu['IMAGING_DATA_FT'].data['COHERENCE']
    header = hdu[0].header
    hdu.close
    array_properties = get_header_bl_data(header)
    nframeGRA4MAT=np.shape(coherence)[0]
    nChanGRA4MAT=4
    nBaseGRA4MAT=6
    gdc=np.zeros((nframeGRA4MAT,nBaseGRA4MAT),dtype=np.float)
    pdc=np.zeros((nframeGRA4MAT,nBaseGRA4MAT),dtype=np.float)
    foo1=np.zeros(nframeGRA4MAT,dtype=np.complex)
    foo2=np.zeros(nframeGRA4MAT,dtype=np.complex)
    lambGRA4MAT=[(2.07767+2.06029)/2.,(2.17257+2.15652)/2.,(2.27180+2.26070)/2.,(2.35531+2.35021)/2.]
    factor1=1./(2*np.pi*((1/lambGRA4MAT[0])-(1/lambGRA4MAT[3]))/3.)
    factor2=np.mean(lambGRA4MAT)/(2.*np.pi)
    for iBase in range(nBaseGRA4MAT):
        foo1[:]=0.
        foo2[:]=0.
        for iChan in range(nChanGRA4MAT):
            if (iChan < nChanGRA4MAT-1):
                foo1[:,]+=(coherence[:,4+iBase+iChan*16]+1J*coherence[:,10+iBase+iChan*16])*\
                         (coherence[:,4+iBase+(iChan+1)*16]-1J*coherence[:,10+iBase+(iChan+1)*16])
            foo2[:] += (coherence[:, 4 + iBase + iChan*16] + 1J * coherence[:, 10 + iBase + iChan*16])
        
        pdc[:, iBase]=np.angle(foo2[:]*np.conjugate(np.median(foo2)))*factor2 #2.15 / (2. * np.pi)
        gdc[:, iBase]=np.angle(foo1[:])*factor1 #2.15 * 25./ (2. * np.pi)
    #return time,gd,pd, array_properties 
    return time, gdc, pdc, array_properties
def get_header_bl_data(header):
    """
    Pulls the baseline infomation from the header 
    """
    idx = np.arange(1,4+1)
    print("Extracting array information")
    tnames = []
    tstas = []
    stalocs = []
    for i in idx:
        tnames.append(header["HIERARCH ESO ISS CONF T%dNAME"%i])
        tstas.append(header["HIERARCH ESO ISS CONF STATION%d"%i])
        stalocs.append(np.array((header["HIERARCH ESO ISS CONF T%dX"%i], header["HIERARCH ESO ISS CONF T%dY"%i])))
    stalocs = np.array(stalocs)
    A = A_GRAVITY
    blengths = []
    for i in range(A.shape[0]):
        print("Baseline %d"%i)
        t1 = np.squeeze(np.argwhere(A[i]==1))
        t2 = np.squeeze(np.argwhere(A[i]==-1))
        sta1, sta2 = (tstas[t1], tstas[t2])
        locA = stalocs
        print("From %s at %s (%.3f, %.3f) to %s at %s (%.3f, %.3f)"%(tnames[t1], tstas[t1], stalocs[t1][0], stalocs[t1][1], tnames[t2], tstas[t2], stalocs[t2][0], stalocs[t2][1]))
        blength = np.sqrt((stalocs[t2][0]-stalocs[t1][0])**2 + (stalocs[t2][1]-stalocs[t1][1])**2)
        blengths.append(blength)
        print("Length %.1f m"%(blength))
        print()
    blengths = np.array(blengths)
    array_properties = {"tnames":tnames, "tstas":tstas, "stalocs":stalocs, "blengths":blengths}
    return array_properties
# The baseline matrix for GRAVITY data
A_GRAVITY = np.array([[1.0, -1.0, 0.0, 0.0],
                     [1.0, 0.0, -1.0, 0.0],
                     [1.0, 0.0, 0.0, -1.0],
                     [0.0, 1.0, -1.0, 0.0],
                     [0.0, 1.0, 0.0, -1.0],
                     [0.0, 0.0, 1.0, -1.0]])


def get_star_params(star, verbose=True, showtable=False):
    """
    Queries GAIA DR2 catalog for distance, radius and 
    temperature of star:
    
    **Parameters:**
    
    * star   : str   the name of the star
    * verbose: Whether to print the details found
    * showtable: Whether to show the table in list form
    
    ** returns:**
    * dist [pc]   : Distance
    * T    [K]    : Effective temperature
    * R    [Rsun] : Radius
    """
    from astroquery.vizier import Vizier
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    
    v = Vizier.query_object(star, catalog=["GAIA"])
    v["I/345/gaia2"].sort(keys="Gmag")
    obj = v["I/345/gaia2"][0]
    dist = (obj["Plx"] * u.mas).to(u.parsec, equivalencies=u.parallax())
    T = obj["Teff"]
    Rad = obj["Rad"]
    if showtable:
        from IPython.display import display, HTML
        display(v["I/345/gaia2"])
    if verbose:
        print("Dist = ", dist, "[pc]")
        print("T = ", T, "[K]")
        print("R = ", Rad, "[R_sun]")
    return dist, T, Rad

def get_star_params_GAIA_JMMC(star, verbose=True, showtable=False):
    """
    Queries GAIA DR2 catalog for distance, radius and 
    temperature of star:
    
    **Parameters:**
    
    * star   : str   the name of the star
    * verbose: Whether to print the details found
    * showtable: Whether to show the table in list form
    
    **returns:**
    
    * dist [pc]   : Distance
    * T    [K]    : Effective temperature
    * R    [Rsun] : Radius
    """
    from astroquery.vizier import Vizier
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    
    v = Vizier.query_object(star, catalog=["GAIA"])
    j = Vizier.query_object(star, catalog=["JMMC"])
    catav = v["I/345/gaia2"]
    catav.sort(keys="Gmag")
    obj = catav[0]
    dist = (obj["Plx"] * u.mas).to(u.parsec, equivalencies=u.parallax()).value
    try: #See if it is available in JMMC
        cataj = j["II/346/jsdc2"]
        cataj.sort(keys="Vmag")
        objj = cataj[0]
        T = objj["Teff"]
        Rad = objj["UDDK"]*u.mas.to(u.rad)/2*dist*u.pc.to(u.Rsun)
    except: # if it is not in JMMC
        try:
            cataj = j["II/300/jsdc"]
            cataj.sort(keys="Vmag")
            objj = cataj[0]
            T = objj["Teff"]
            Rad = objj["UDDK"]*u.mas.to(u.rad)/2*dist*u.pc.to(u.Rsun)
        except:
            T = obj["Teff"]
            Rad = obj["Rad"]
            logit.error("Couldn't find the entry in JSDC catalog")
    if showtable:
        from IPython.display import display, HTML
        display(catav)
        display(cataj)
    if verbose:
        print("Dist = ", dist, "[pc]")
        print("T = ", T, "[K]")
        print("R = ", Rad, "[R_sun]")
    return dist, T, Rad

def update_star_params(config, verbose=True):
    
    name = config.get("target", "target")
    dist, T, Rad = get_star_params_GAIA_JMMC(name, verbose=False)
    if verbose:
        print("Dist set to ", dist, "[pc]")
        print("T set to ", T, "[K]")
        print("R set to ", Rad, "[R_sun]")
    
    config.set("target", "star_distance", value="%.2f"%dist)
    config.set("target", "star_temperature", value="%.2f"%T)
    config.set("target", "star_radius", value="%.2f"%Rad)
    
def find_best_night(obs, target, showplot=True, duration=None):
    """
    **Parameters:**
    
    * obs : Observatory object
    * showplot : Whether to show a plot of the max elevation 
    * duration   [h] : duration of the observing run
    
    **Returns:**
    
    * if duration is None (default) : ``besttime`` an ``astropy.Time`` object
    * else : ``(Tstart, Tend)`` Start and end of the observing run
    
    """
    import astropy.units
    from astropy.time import Time, TimeDelta
    dt = TimeDelta(1*astropy.units.day)
    t0 = Time("2021-01-01T03:00:00")
    times = [t0 + k*dt for k in range(365)]
    poslist = obs.get_positions(target, times)
    poslist[0]
    elev_vec = np.array(poslist.altaz.alt.value[0])
    highest_index = np.argmax(elev_vec)
    besttime = times[highest_index]
    if showplot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(poslist.altaz.alt.value[0])
        plt.xlabel("Day of the year")
        plt.ylabel("Elevation at midnight [deg]")
        plt.show()
        print("Best time:", besttime)
    if duration is None:
        return besttime
    else:
        Tstart = Time(besttime) - TimeDelta((duration/2)*astropy.units.h)
        Tend = Time(besttime) + TimeDelta((duration/2)*astropy.units.h)
        return Tstart, Tend
def update_observing_night(config, time=None, duration=6.,  verbose=True,
                          target_coords=None):
    """
    **Parameters**
    
    * config  : The parsed config file to modify
    * time : [fits format] (if None, will optimize for elevation)
    * duration : [h] *Default*: 6.h
    * verbose : Whether to print some of the intermediate results
    * target_coords : Coordinates of the object of interst.
    
        - **None** *Default*: Uses the target in the config file
        - ``astropy.SkyCoords`` object.
    
    """
    from scifysim.observatory import observatory
    from astroplan import FixedTarget
    from astropy.coordinates import SkyCoord
    obs = observatory(config=config)
    if target_coords is None:
        tarname = config.get("target", "target")
        target = FixedTarget.from_name(tarname)
    else:
        tarname = "from coords"
        target = FixedTarget(target_coords)
    if time is None:
        tstart, tend = find_best_night(obs, target, showplot=verbose, duration=duration)
        tstart = tstart.value
        tend = tend.value
    else : 
        tstart = time[0]
        tend = time[1]
    name = config.get("target", "target")
    dist, T, Rad = get_star_params(name, verbose=False)
    if verbose:
        print("Sequence start set to ", tstart)
        print("Sequence end set to ", tend)
    config.set("target", "seq_start", value=tstart)
    config.set("target", "seq_end", value=tend)
    
def get_location(simple_map, map_extent=None,
                 search_function=np.argmax, mode="polar"):
    frac = np.array(np.unravel_index(np.argmax(simple_map), simple_map.shape))/np.array(simple_map.shape) 
    if map_extent is not None:
        gain = np.array(map_extent[1]-map_extent[0], map_extent[3]-map_extent[2])
        offset = np.array(map_extent[0], map_extent[2])
    else :
        gain = np.array([simple_map.shape[-2] , simple_map.shape[-1]])
        offset = np.array([-simple_map.shape[-2]/2, -simple_map.shape[-1]/2])# order?
    pos = frac*gain+offset
    print("cartesian:", pos)
    cpform = pos[0] - 1j*pos[1]
    r = np.abs(cpform)
    theta = np.angle(cpform)
    print("r, theta", r, theta)
    if "polar" in mode:
        return r, theta
    else :
        return pos
    
def mag2sig(mag, dit, lead_transmission,context, eta=1., spectrum="flat", ds_sr=1.):
    """
    Obtains the coefficients including object magnitude, detector integration time,
    transmission, spectral channels, pixel solid angle, and quantum efficiency (default: 1.).
    
    **Parameters:**
    
    * mag     : Magnitude of object (Vega)
    * dit     : detector integration time [s]
    * lead_transmission : The transmission of the leading element (often sky) of the 
      transmission-emission chain
    * context : The spectral context of the instrument (defining the vega magnitudes)
    * eta     : (default=1.) The quantum efficiency of the detector
    * spectrum : The type of spectrum of the object defaults to flat as opposed to the 
      temperature of Vega.
    * ds_sr   : When working with a map the solid angle of a pixel [sr]
      Can be found in `simulator.vigneting_map.ds_sr` after maps have been computed
    """
    
    aspectrum = context.sflux_from_vegamag(mag) #asim.src.planet.ss.flatten()
    if spectrum == "flat":
        aspectrum = np.mean(aspectrum)*np.ones_like(aspectrum)
    acoeff = coeff(dit, lead_transmission, context,
                    eta=eta, spectrum=spectrum, ds_sr=ds_sr)
    full_coeff = aspectrum*acoeff
    return full_coeff

def coeff(dit, lead_transmission,context, eta=1., spectrum="flat", ds_sr=1.):
    """
    Obtains the coefficients including detector integration time,
    transmission, spectral channels, pixel solid angle, and quantum efficiency (default: 1.).
    
    **Parameters:**
    
    * dit     : detector integration time [s]
    * lead_transmission : The transmission of the leading element (often sky) of the 
      transmission-emission chain
    * context : The spectral context of the instrument (defining the vega magnitudes)
    * eta     : (default=1.) The quantum efficiency of the detector
    * spectrum : The type of spectrum of the object defaults to flat as opposed to the 
      temperature of Vega.
    * ds_sr   : When working with a map the solid angle of a pixel [sr]
      Can be found in `simulator.vigneting_map.ds_sr` after maps have been computed
    """
    acoeff = (1/(ds_sr) *\
                  dit * eta *\
                  lead_transmission.get_downstream_transmission(context.avega.lambda_science_range))
    return acoeff


def extract_diffobs_map_old(maps, simulator, dit=1., mag=None, postprod=None, eta=1., K=None):
    """
    
    Here in photons per dit (default=1s)
    
    **Parameters:**
    
    * maps     : Transmission maps for 1 ph/s/m2 
                        at the entrance of atmo
    * simulator : a sumulator object (see comments below)
    * dit      : Detector integration time [s]
    * mag      : The magnitude of a source on the map
    * postprod : A post-processing (whitening matrix)
    * eta      : The quantum efficiency of the detector
    * K        : A matrix to extract the differential
                observable quantities (kernel matrix).
                By default, the matrix is obtained from 
                simulator.combiner.K
    
    Simulator object is used for:
    
    * The head transmission (simulator.src.sky)
    * The solid angle of the pisel
    * The spectral context (for the spectral channels)
    """
    if len(maps.shape) == 5:
        nt, nwl, nout, ny, nx = maps.shape
    elif len(maps.shape) == 4:
        nt, nwl, nout, no = maps.shape
        raise NotImplementedError("single_datacube")
    else : 
        raise NotImplementedError("Shape not expected")
    if K is None:
        K = simulator.combiner.K
        
    acoeff = coeff(dit, simulator.src.sky,
                               simulator.context,ds_sr=simulator.vigneting_map.ds_sr,
                               eta=eta, spectrum="flat")
    
    
#    coeff = (1/(asim.vigneting_map.ds_sr) *\
#                  dit * eta *\
#                  diffuse[0].get_downstream_transmission(asim.lambda_science_range))
    ymap = []
    for at in range(nt):
        for awl in range(nwl):
            ay = K.dot(maps[at,awl,:,:,:].reshape(nout,ny*nx))
            ay = ay.reshape(1,ny,nx)
            ymap.append(ay)
    ymap = np.array(ymap)
    ymap = ymap.reshape(nt, nwl, 1, ny, nx)
    ymap = ymap*acoeff[None,:,None,None, None]
    
    if postprod is not None:
        print(ymap.shape)
        for (k, i, j), a in np.ndenumerate(ymap[:,0,0,:,:]):
            ymap[k,:,:,i,j] =  postprod[k,:,:].dot(ymap[k,:,:,i,j])
    return ymap
def extract_diffobs_map(maps, simulator, mod=np, dit=1., mag=None, postprod=None, eta=1., K=None):
    """
    Here in photons per dit (default=1s)
    
    **Parameters:**
    
    * maps     : Transmission maps for 1 ph/s/m2 
                        at the entrance of atmo
    * simulator : a sumulator object (see comments below)
    * dit      : Detector integration time [s]
    * mag      : The magnitude of a source on the map
    * postprod : A post-processing (whitening matrix)
    * eta      : The quantum efficiency of the detector
    * K        : A matrix to extract the differential
                observable quantities (kernel matrix).
                By default, the matrix is obtained from 
                simulator.combiner.K
    
    Simulator object is used for:
    
    * The head transmission (simulator.src.sky)
    * The solid angle of the pisel
    * The spectral context (for the spectral channels)
    """
    if len(maps.shape) == 5:
        nt, nwl, nout, ny, nx = maps.shape
    elif len(maps.shape) == 4:
        nt, nwl, nout, no = maps.shape
        raise NotImplementedError("single_datacube")
    else : 
        raise NotImplementedError("Shape not expected")
    if K is None:
        K = simulator.combiner.K
        
    acoeff = coeff(dit, simulator.src.sky,
                               simulator.context,ds_sr=simulator.vigneting_map.ds_sr,
                               eta=eta, spectrum="flat")
    
    
#    coeff = (1/(asim.vigneting_map.ds_sr) *\
#                  dit * eta *\
#                  diffuse[0].get_downstream_transmission(asim.lambda_science_range))
    ymap = []
    ymap = mod.einsum("k o , s w o x y -> s w k x y", K, maps)
    #for at in range(nt):
    #    for awl in range(nwl):
    #        ay = K.dot(maps[at,awl,:,:,:].reshape(nout,ny*nx))
    #        ay = ay.reshape(1,ny,nx)
    #        ymap.append(ay)
    #ymap = np.array(ymap)
    #ymap = ymap.reshape(nt, nwl, 1, ny, nx)
    ymap = ymap*acoeff[None,:,None,None, None]
    
    if postprod is not None:
        print(ymap.shape)
        for (k, i, j), a in np.ndenumerate(ymap[:,0,0,:,:]):
            ymap[k,:,:,i,j] =  postprod[k,:,:].dot(ymap[k,:,:,i,j])
    return ymap

def test_maps(fname="local_config/default_R200", target="GJ 86 A"):
    """
    Testing and comparing the 2 approaches to build transmission maps.
    """
    asim.build_all_maps_dask(mapcrop=0.2, mapres=10, )
    asim.persist_maps_to_disk()
    mapdask = asim.maps.compute()
    asim.build_all_maps(mapcrop=0.2, mapres=10, )
    mapnp = asim.maps.copy()
    print("They are the same map: ", np.allclose(mapnp, mapdask))
    return mapnp, mapdask


def trois(x, xmin, xmax, ymin=0., ymax=1.):
    xrange = xmax-xmin
    yrange = ymax-ymin
    normalized = (x - xmin)/xrange
    y = normalized*yrange + ymin
    return y
"""
WARNING-matplotlib.axes._axes- *c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2-D array with a single row if you intend to specify the same RGB or RGBA value for all points."""

def eval_perf(afile,t_exp,n_frames=100,
               thetarget=None, update_params=False,
               instrumental_errors=True, seed=None,
               crop=1., target_coords=None,
               compensate_chromatic=True,
               modificators=None,
               plotall=False,
               use_tqdm=False):
    """
    Evaluate the noise profile for a given parameter file.
    
    **Parameters:**
    
    - afile      : Config file 
    - t_exp      : Exposure time
    - n_frames   : The number of frames for MC simulation
      to evaluate the instrumental noise
    - instrumental_errors : Bool. Whether to include instrumental
      errors
    - seed       : seed to pass when generating the 
      time series of instrumental errors
    - crop       : factor to pass to the injector
    - target_coords : Coordinates of the target 
      default:None : finds the target from catalog
    - compensate_chromatic : Bool Whether to run the optimization
      of the actuators to compensate for chromaticity
      in the combiner
    - plotall    : Whether to plot a bunch of charts
    - use_tqdm   : Whether tqdm should be used at the step of the
      MC simulations. (use false for parameter exploration)
                
    **Returns:**
    
    - asim       : The simulator object
    - prof       : The noise profile object
    - characteristics : A dictionary containing a collection of
      performance statistics
    """
    import scifysim as sf
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import sqrtm
    from tqdm import tqdm
    
#    asim = prepare_all(afile, thetarget=thetarget, update_params=update_params,
#               instrumental_errors=instrumental_errors, seed=seed,
#               crop=crop, target_coords=target_coords, modificators=modificators,
#               compensate_chromatic=compensate_chromatic)
#    diffuse = [asim.src.sky, asim.src.UT, asim.src.warm_optics, asim.src.combiner, asim.src.cold_optics]
#    #asim.point(asim.sequence[3], asim.target)
#    #asim.combiner.chromatic_matrix(asim.lambda_science_range)
#    integ = asim.make_metrologic_exposure(asim.src.planet, asim.src.star, diffuse,
#                                          texp=t_exp)
#    integ.prepare_t_exp_base()
#    #consolidate_metrologic(integ)
#    integ.consolidate_metrologic()
    
    
    ###############
    #skipping
    ###############
    asim = sf.utilities.prepare_all(afile, thetarget=thetarget, update_params=True,
                  instrumental_errors=True, modificators=modificators)
    diffuse = [asim.src.sky, asim.src.UT, asim.src.warm_optics, asim.src.combiner, asim.src.cold_optics]

    asim.combiner.chromatic_matrix(asim.lambda_science_range)
    #asim.integrator = integ
    asim.context = sf.analysis.spectral_context("local_config/vega_R200.ini")
    integ = asim.make_metrologic_exposure(asim.src.planet, asim.src.star, diffuse,
                                          texp=t_exp)
    asim.integrator = integ
    
    print("Starting loop", flush=True)
    dit = 1.
    mynpix = 8
    reveta = 1/integ.eta
    full_record = True
    diffuse = [asim.src.sky, asim.src.UT, asim.src.warm_optics, asim.src.combiner, asim.src.cold_optics]
    screen_age = 0.
    datacube = []
    dit_intensity = []
    starlights = []
    planetlights = []
    coupling_means = []
    compling_stds = []
    phase_means = []
    phase_stds = []
    injphase_means = []
    injphase_stds = []
    if use_tqdm:
        theit = tqdm(range(n_frames))
    else:
        theit = range(n_frames)
    for i in theit:
        if screen_age>=20. :
            print("generating screen")
            asim.injector.update_screens()
            screen_age = 0.
        integ = asim.make_exposure(asim.src.planet, asim.src.star, diffuse,
                                    texp=dit,
                                    monitor_phase=True,
                                   spectro=None)
        datacube.append(integ.get_total(spectrograph=None,
                                        t_exp=dit,
                                        n_pixsplit=mynpix))
        dit_intensity.append(reveta * integ.forensics["Expectancy"].sum(axis=0))
        if full_record:
            starlights.append(integ.starlight.astype(np.float32))
            planetlights.append(integ.planetlight.astype(np.float32))
            coupling_means.append(np.mean(integ.inj_amp, axis=0))
            compling_stds.append(np.mean(np.std(integ.inj_amp, axis=0)))
            phase_means.append(np.mean(integ.ft_phase, axis=0))
            phase_stds.append(np.mean(np.std(integ.ft_phase, axis=0)))
            injphase_means.append(np.mean(integ.inj_phase, axis=0))
            injphase_stds.append(np.mean(np.std(integ.inj_phase, axis=0)))
            
        integ.reset() # This can be removed after new kernel start
        screen_age += dit
    datacube = np.array(datacube)
    dit_intensity = np.array(dit_intensity)
    starlights = np.array(starlights)
    planetlights = np.array(planetlights)
    statics = np.array(integ.static)
    
    coupling_means = np.array(coupling_means)
    coupling_stds = np.array(compling_stds)
    phase_means = np.array(phase_means)
    phase_stds = np.array(phase_stds)
    injphase_means = np.array(injphase_means)
    injphase_stds = np.array(injphase_stds)
    
    ###################################
    integ.reset()
    integ = asim.make_exposure(asim.src.planet, asim.src.star, diffuse,
                                    texp=dit,
                                    monitor_phase=False,
                                   spectro=None)
    block = integ.get_total(spectrograph=None,t_exp=dit, n_pixsplit=mynpix)
    print(f"datacube shape: {datacube.shape}")
    print(f"dit = {dit} s")
    brigh_max = np.max(np.mean(integ.forensics["Expectancy"][:,:,asim.combiner.bright], axis=0))
    dark_max = np.max(np.mean(integ.forensics["Expectancy"][:,:,asim.combiner.dark], axis=0))
    longest_exp_bright = 65000 / (brigh_max/dit)
    longest_exp_dark = 65000 / (dark_max/dit)
    print(f"Bright limit: {longest_exp_bright:.2f} s\n Dark limit: {longest_exp_dark:.2f} s")
    data_std = np.std(datacube, axis=0)
    diff_std = np.std(datacube[:,:,3]-datacube[:,:,4], axis=0)
    
    integ.static = asim.computed_static
    integ.mean_starlight = np.mean(starlights, axis=0)
    integ.mean_planetlight = np.mean(planetlights, axis=0)
    integ.mean_intensity = np.mean(dit_intensity, axis=0)
    
    ################ # Diff null distribution
    diffdata = datacube[:,:,4] - datacube[:,:,3]
    shapirops = np.array([shapiro(diffdata[:,i]).pvalue \
              for i in range(asim.lambda_science_range.shape[0])])
    
    nsamples = diffdata.shape[0]
    medians = np.median(diffdata, axis=0)
    histograms = []
    edgess = []
    gauss_params = []
    p0s = []
    for i, awl in enumerate(asim.lambda_science_range[:]):
        print(i)
        print(f"Shapiro p = {shapirops[i]:.3f}")
        print(f"Shapiro p-value: {np.format_float_scientific(shapirops[i], precision=3)}")
        p0 = (np.mean(diffdata[:,i]), np.std(diffdata[:,i]))
        myhist, edges = np.histogram(diffdata[:,i], bins=30, density=True)
        bincenters = get_bincenters(edges)
        parameters, histerror, covariance = fit_distribution(myhist, bincenters, Gauss,
                                           p0=p0, nsamples=nsamples)
        parameters_cauchy, histerror, covariance_cauchy = fit_distribution(myhist, bincenters, Cauchy,
                                           p0=p0, nsamples=nsamples)
        histograms.append(myhist)
        edgess.append(edges)
        p0s.append(p0)
        if plotall:
            print(p0)
            print(parameters)
            print(parameters_cauchy)
            plt.figure(figsize=(8,4), dpi=100)
            plt.subplot(121)
            plt.plot(bincenters, myhist, label="diff-null (MC)")
            plt.plot(bincenters, Gauss(bincenters, parameters[0], parameters[1]), label="Gaussian fit")
            plt.plot(bincenters, Cauchy(bincenters, parameters_cauchy[0], parameters_cauchy[1]), label="Cauchy fit")

            #plt.yscale("log")
            plt.title(f"Wavelength: $\lambda = {awl:.2e}$ m")
            plt.xlabel(f"Differential observable value [e-]")
            plt.axvline(parameters[0],color="k")
            plt.axvline(np.mean(diffdata[:,i]),color="k", linestyle="--")
            plt.legend()
            plt.subplot(122)
            plt.plot(bincenters, myhist, label="MC_sim (diff-null)")
            plt.plot(bincenters, Gauss(bincenters, parameters[0], parameters[1]), label="Gaussian fit")
            plt.plot(bincenters, Cauchy(bincenters, parameters_cauchy[0], parameters_cauchy[1]), label="Cauchy fit")

            plt.yscale("log")
            plt.title(f"Wavelength: $\lambda = {awl:.2e}$ m")
            plt.xlabel(f"Differential observable value [e-]")
            plt.axvline(parameters[0],color="k")
            plt.axvline(np.mean(diffdata[:,i]),color="k", linestyle="--")
            #plt.savefig(f"/tmp/plots/gaussian_fit_{1e6*awl:.2f}microns.pdf")
            plt.show()
    ####################
    # More elipsis
    #####################
    histograms = np.array(histograms)
    edgess = np.array(edgess)
    p0s = np.array(p0s)
    from kernuller import pairwise_kernel
    ak = pairwise_kernel(2)
    myk = np.hstack((np.zeros((1,3)), ak, np.zeros((1,3))))
    asim.combiner.K = myk


    diffobs = np.einsum("ij, mkj->mk",asim.combiner.K, dit_intensity)
    diff_std = np.std(diffobs, axis=0)
    prof = sf.analysis.noiseprofile(integ, asim, diffobs, n_pixsplit=mynpix)
    if plotall:
        fig = prof.plot_noise_sources(asim.lambda_science_range, dit=1., show=False,
                                     ymin=0.2, ymax=1.)
        plt.legend(loc="upper right", fontsize="xx-small")
        plt.savefig("/tmp/plots/noises.pdf", bbox_inches='tight', dpi=200)
        plt.show()
    
    characteristics = {"mean_coupling":np.mean(coupling_means),
                  "inter-frame_coupling_std":np.mean(np.std(coupling_means, axis=0)),
                  "intra-frame_coupling_std":np.mean(coupling_stds),
                  "inter-frame_ft_phase":np.mean(np.std(phase_means, axis=0)),
                  "intra-frame_ft_phase":np.mean(phase_stds),
                  "inter-frame_inj_phase":np.mean(np.std(injphase_means, axis=0)),
                  "shapirops":shapirops,
                  "fit_params":parameters,
                  "histograms":histograms,
                  "hist_edges":edgess,
                  "nsamples":nsamples,
                  "p0":p0,
                  "medians":medians
                  }
    
    return asim, prof, characteristics

def produce_maps(asim, prof, adit=1.,
                mypfa=0.01,mypdet=0.9,
                base_name="/tmp/R200_twinmaps_",
                total_time=3*3600,
                starmags=np.linspace(2.,10., 20),
                decs=np.linspace(-90, 30, 10),
                mapcrop=0.5, mapres=50):
    """
    Utility macro that packages the production of multiple sensitivity
    maps for a given director object and noise profile
    
    **Parameters:**
    
    - asim     : Simulator director object
    - prof     : Noise profile
    
    **Returns:**
    
    - metamap   : T
    - themap    : The energy detector map in magnitude
    - theNPmap  : The Neyman-Pearson detector map in magnitude
    
    """
    
    import copy
    import gc
    from tqdm import tqdm
    import scifysim as sf
    
    from scipy.linalg import sqrtm
    old_target = copy.copy(asim.target)
    
    from einops import rearrange
    rfx = 1.
    rfx_N = 22. # For NP test, the ref mag needs to be relatively high
    nslots = len(asim.sequence)
    ndits = total_time/adit/nslots
    ra = old_target.ra.deg #asim.target.ra.adeg

    metadata = {"DIT": (adit, "Detector integration time [s]"),
               "PFA": (mypfa, "False alarm rate"),
               "PDET": (mypdet, "Detection probability"),
               "INTEGRATION": (total_time, "Total integration time on target [s]"),
               "CALIBRATION": ("IDEAL", "Currently only IDEAL. Biases are assumed perfectly corrected")}
    for i, adec in enumerate(tqdm(decs[:])):
        print(f"======= Declination {adec}")
        metadata["DEC"] = (adec, "The declination for which the object map was computed")
        asim.config.set("target", "mode", value="coords")
        asim.config.set("target", "target", value="%.2f %.2f"%(ra, adec))
        asim.prepare_observatory(file=asim.config)
        asim.point(asim.sequence[3], asim.target)
        print(asim.target)
        asim.build_all_maps(mapres=mapres, mapcrop=mapcrop)

        themaps = []
        theNPmaps = []
        for starmag in starmags:
            amat = 1/ndits * prof.diff_noise_floor_dit(starmag, adit, matrix=True)
            wmat = sqrtm(np.linalg.inv(amat))
            whitenings = np.ones(len(asim.sequence))[:,None,None]*wmat[None,:,:]
            aymap = extract_diffobs_map(asim.maps, asim,
                                        postprod=None, eta=asim.integrator.eta) *\
                                    asim.context.sflux_from_vegamag(rfx_N)[None,:,None,None,None]
            #rearrange(aymap, "a b c d e -> a (b c) d e")
            mgs,fx, Tes = sf.analysis.get_sensitivity_Te(aymap, postproc_mat=whitenings,
                                                         ref_mag=rfx_N, pdet=mypdet,
                                                         pfa=mypfa, verbose=False, )

            mags_NP = sf.analysis.get_sensitivity_Tnp(aymap, postproc_mat=whitenings,
                                                      pfa=0.1, pdet=0.9, ref_mag=rfx_N,
                                                     verbose=False)
            themaps.append(mgs)
            theNPmaps.append(mags_NP)
        themaps = np.array(themaps)
        theNPmaps = np.array(theNPmaps)
        ameta = sf.map_manager.map_meta(asim, starmags, 
                                        mgs=themaps.astype(np.float64),
                                        mgs_TNP=theNPmaps,
                                        metadata=metadata)
        ameta.to_fits(base_name+"%s_dec_%.1f.fits"%(ameta.input_order, adec))
        aplot = ameta.plot_map(4.,dpi=50, show=True, detector="E")
        aplot = ameta.plot_map(4.,dpi=50, show=True, detector="N")
        #plt.title("Dec: %.0f"%adec)
        #plt.show()
        del asim.maps
        gc.collect()
        
    return ameta, themaps, theNPmaps
    

from scipy.optimize import curve_fit
from scipy.stats import shapiro, norm, cauchy

Gauss = norm.pdf
Cauchy = cauchy.pdf

def Gauss(x, loc, scale):
    """
    Shortcut to ``scipy.stats.norm.pdf``
    """
    y = norm.pdf(x, loc=loc, scale=scale)
    return y
def Cauchy(x, loc, scale):
    """
    Shortcut to ``scipy.stats.cauchy.pdf``
    """
    y = cauchy.pdf(x, loc=loc, scale=scale)
    return y

def get_bincenters(edges):
    """
    Quickly converts bin edges to bin centers:
    
    **Arguments:**
    
    * edges: *Array* of size n
      
    **Returns:**
    
    * bincenters: Centers of the intervals (size n-1)
    """
    bincenters = np.mean([edges[:-1], edges[1:]], axis=0)
    return bincenters
def fit_distribution(hist, bincenters, mypdf, nsamples=None, p0=(0.,0.)):
    """
    Fits the histogram to a given pdf
    
    **Arguments:**
    
    * hist : *array-like*  The empirical histogram given in density
      so that it has unity integral
    * bincenters : *array-like* The center of bins. Same shape as hist.
      tip: use ``get_bincenters`` to convert edges to bin centers
    * mypdf : *function* A function of the histogram to fit
    * p0 : *array-like* Starting parameters for the fit. Typically
      a couple (loc, scale). Recommended is (mean, std).
      
    **Returns:**
    
    * parameters: The best fit parameters (typically (loc, scale).
    * histerror : An estimation of the errors on the bins, based on 
      Poissonian statistics (and set to the max for empty bins.
    * covariance: The covariance of the parameters.
    """

    if nsamples is None:
        histerror=None
    else:
        histerror_base = np.sqrt(nsamples*hist)/nsamples
        # We further replace the 0 values (empty bins) with the maximum standard deviation.
        histerror = np.where(histerror_base>0., histerror_base, np.max(histerror_base))
    parameters, covariance = curve_fit(mypdf, bincenters, hist,
                                       p0=p0, sigma=histerror)
    return parameters, histerror, covariance 
