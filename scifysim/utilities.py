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



class ee(object):
    """
    Additional functionnality needed:
    see comments starting 2021/02/24 https://github.com/sympy/sympy/issues/5642 .
    """
    def __init__(self, expression):
        """
        ee is for Executable expression
        Encapsulates an expression for flexible lambda evaluation.
        expression  : a sympy expression to 
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
        args    : a tuple of sympy for symbols to use as inputs
        modules : The module to use for computation.
        """
        #Here, we have to decide if we need the function to all 
        if ("numexpr" in modules) and (self.outlen is not 1):
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
               crop=1., target_coords=None):
    """
    A shortcut to prepare a simulator object
    """
    from scifysim import director
    asim = director.simulator(fpath=afile)
    #asim.config = loaded_config
    if thetarget is not None:
        asim.config.set("target", "target", value=thetarget)
    if update_params:
        update_star_params(config=asim.config)
    update_observing_night(config=asim.config, target_coords=target_coords)
    asim.prepare_observatory(file=asim.config)
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
    return asim
    

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

def profileit(func):
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile" # Name the data file sensibly
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
    Parameters:
    ps       : A power spectrum
    matchto  : np.ndarray of preceding series. The mean of the first
                n (matchlength) values of the new series are matched
                to the mean of the last n values of this array.
    matchlength : integer. The number of values to match 
    keepall  : Boolean. If true, will return the concatenation of the
                new array with the old. If False, will return only the
                new array.
    
    
    output:
    a numpy ndarray of the time series
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
    Returns the concatenation.
    
    a        : 1D array the root of the series
    b        : 1D array the tail of the series 
    nmatch   : integer  The number of samples used to match the mean 
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
    star   : str   the name of the star
    verbose: Whether to print the details found
    
    returns:
    dist [pc]   : Distance
    T    [K]    : Effective temperature
    R    [Rsun] : Radius
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
    star   : str   the name of the star
    verbose: Whether to print the details found
    
    returns:
    dist [pc]   : Distance
    T    [K]    : Effective temperature
    R    [Rsun] : Radius
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
    try:
        cataj = j["II/346/jsdc2"]
        cataj.sort(keys="Vmag")
        objj = cataj[0]
        T = objj["Teff"]
        Rad = objj["UDDK"]*u.mas.to(u.rad)/2*dist*u.pc.to(u.Rsun)
    except:
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
    duration   [h] : duration of the observing run
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
    config               The parsed config file to modify
    time      [fits format] (if None, will optimize for elevation)
    duration  [h] 
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