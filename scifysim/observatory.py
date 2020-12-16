
import numpy as np
import sympy as sp

import kernuller


from astropy.time import Time
import astropy.units as u

import astroplan
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun

import logging

logit = logging.getLogger(__name__)


"""
Basic usage:

import kernuller.observatory
myobs = kernuller.observatory.observatory(kernuller.VLTI)
tarnames = "Spica"
targets = [kernuller.observatory.astroplan.FixedTarget.from_name(tar) for tar in tarnames]
obstimes = myobs.build_observing_sequence()
target_positions = myobs.get_positions(targets[0], obstimes)
newarray = myobs.get_projected_array(myobs.get_positions(targets, obstimes)[0,0])
"""


class observatory(object):
    """
    This class help define the properties of the observatory infrastructure, especially the uv coverage.
    """
    def __init__(self, statlocs=None, location=None, verbose=False, multi_dish=True, config=None):
        """
        statlocs : The station locations 
                    (east, north) for each aperture shape is (Na, 2)
        location : An astropy.coordinatEarthLocation (default = Paranal)
                    example: myloc = astroplan.Observer.at_site("Paranal", timezone="UTC")
        """
        self.verbose = verbose
        self.config = config
        if location is None:
            location = self.config.get("configuration", "location")
            self.observatory_location = astroplan.Observer.at_site(location, timezone="UTC")
        else :
            self.observatory_location = astroplan.Observer.at_site(location, timezone="UTC")
        
        
        self.array_config = self.config.get("configuration", "config")
        raw_array = eval("kernuller.%s"%(self.array_config))
        self.order = self.config.getarray("configuration", "order").astype(np.int16)
        self.statlocs = raw_array[self.order]
        
        self.theta = sp.symbols("self.theta")
        #R handles the azimuthal rotation
        self.Rs = sp.Matrix([[sp.cos(self.theta), sp.sin(self.theta)],
                       [-sp.sin(self.theta), sp.cos(self.theta)]])
        self.R = sp.lambdify(self.theta, self.Rs, modules="numpy")
        #P handles the projection due to elevation rotation
        self.Ps = sp.Matrix([[1, 0],
                       [0, sp.sin(self.theta)]])
        self.P = sp.lambdify(self.theta, self.Ps, modules="numpy")
        #C handles the piston due to elevation.
        self.Cs = sp.Matrix([[0, sp.cos(self.theta)]])
        self.C = sp.lambdify(self.theta, self.Cs, modules="numpy")
        
    def point(self, time, target):
        self.altaz = self.observatory_location.altaz(target=target,
                                                   time=obstimes)
        self.PA = self.observatory_location.parallactic_angle(time, target=target)
        

        
    def build_observing_sequence(self, times=None,
                            npoints=20, remove_daytime=False):
        """
        Returns the series of obstimes needed to compute the altaz positions
        times : a list of UTC time strings ("2020-04-13T00:00:00")
                that define an interval (if npoints is not None,
                or the complete list of times (if npoints is None)
        npoints : The number of samples to take on the interval
                 None means that the times is the whole list
                 
        remove_daytime : Whether to remove the points that fall during the day
        """
        if times is None:
            times = ["2020-04-13T00:00:00","2020-04-13T10:30:00"]
        #npoints is defined which means we work from define the sampling from an interval
        if npoints is not None:
            obs2502 = Time(times)
            dt = obs2502[1] - obs2502[0]
            obstimes = obs2502[0] + dt * np.linspace(0.,1., npoints)
        #npoints is None means the times represent a list of times 
        else: 
            obstimes = np.array([Time(times[i]) for i in range(len(obstimes))])
            
        totaltime = (obstimes[-1]-obstimes[0]).to(u.s).value
        if remove_daytime:
            logit.info("Removing daytime observations")
            totaltime = (obstimes[-1]-obstimes[0]).to(u.s).value
            halfhourpoints = int(npoints / (totaltime / 900))
            totaltime = (obstimes[-1]-obstimes[0]).to(u.s)
            mask = self.observatory_location.sun_altaz(obstimes).alt<0
            sunelev = self.observatory_location.sun_altaz(obstimes).alt
            rawsunelev = np.array([el.value for el in sunelev])
            midnight = np.argmin(rawsunelev)
            sunrise = np.argmin(np.abs(rawsunelev))
            logit.error("Removing daytime obs is not finallized")
            raise NotImplementedError
        #Record parallactic angles
        
        
        return obstimes
            
        
    def get_positions(self, targets, obstimes):
        """
        Deprecated
        
        Returns the astropy.coordinates.AltAz for a given target
        targets: A list of SkyCoord objects 
        obstimes: A list of astropy.Times to make the observations
        """
        taraltaz = self.observatory_location.altaz(target=targets,
                                                   time=obstimes,
                                                   grid_times_targets=True)
        tarPA = self.observatory_location.parallactic_angle(obstimes, target=targets)
        return taraltaz#, tarPA
    
    def get_position(self, target, time, grid_times_targets=False):
        """
        Returns the astropy.coordinates.AltAz for a given target
        target:   one or a list of of targets
        obstimes: one or a list of astropy.Times to make the observations
        """
        taraltaz = self.observatory_location.altaz(time, target=target,
                                                      grid_times_targets=grid_times_targets)
        tarPA = self.observatory_location.parallactic_angle(time, target=target,
                                          grid_times_targets=grid_times_targets)
        logit.debug("target altaz")
        logit.debug(str(taraltaz.alt)+str(taraltaz.az))
        logit.debug("target PA")
        logit.debug(tarPA)
        return taraltaz, tarPA
        
    def get_projected_array(self, taraltaz, PA=None, loc_array=None):
        """
        Returns the new coordinates for the projected array
        taraltaz : the astropy.coordinates.AltAz of the target
        PA       : parallactic angle to derotate 
        loc_array: the array of points to use (None: use self.statlocs)
        """
        if loc_array is None:
            loc_array = self.statlocs
        arrayaz = self.R((180 - taraltaz.az.value)*np.pi/180).dot(loc_array.T).T
        altazarray = self.P(taraltaz.alt.value * np.pi/180).dot(arrayaz.T).T
        
        if PA is not None:
            radecarray = self.R(PA.rad).dot(altazarray.T).T
            newarray = radecarray
        else: 
            newarray = altazarray
            
        if self.verbose:
            logit.debug("=== AltAz position:")
            logit.debug("az "+ str(taraltaz.az.value -180))
            logit.debug("alt "+ str(taraltaz.alt.value))
            logit.debug("old array "+ str(loc_array))
            logit.debug("new array "+ str(newarray))
        return newarray
    def get_projected_geometric_pistons(self, taraltaz):
        """
        Returns the geomtric piston resutling from the pointing
        of the array.
        taraltaz : the astropy.coordinates.AltAz of the target
        """
        arrayaz = self.R((180 - taraltaz.az.value)*np.pi/180).dot(self.statlocs.T).T
        pistons = self.C(taraltaz.alt.value * np.pi/180).dot(arrayaz.T).T
        if self.verbose:
            logit.debug("=== pistons:")
            logit.debug("az "+ str(taraltaz.az.value -180))
            logit.debug("alt "+ str(taraltaz.alt.value))
            logit.debug("old array "+ str(self.statlocs))
            logit.debug("new array "+ str(pistons))
        return pistons
    

def test_observatory(tarname="Tau cet",
                     startend=["2020-10-20T00:00:00", "2020-10-20T10:00:00"],
                     nblocs=20):
    from kernuller import VLTI
    from . import plot_tools as pt
    
    testobs = observatory(VLTI)
    tarnames = [tarname]
    targets = [astroplan.FixedTarget.from_name(tar) for tar in tarnames]
    mytimes = startend
    obstimes = testobs.build_observing_sequence(times=mytimes, remove_daytime=False, npoints=nblocs)
    target_positions, PAs = testobs.get_position(targets[0], obstimes, grid_times_targets=True)
    print(target_positions)
    testobs.verbose=True
    print("PAs",PAs)
    
    #thearrays = [testobs.get_projected_array(aposition) for aposition in target_positions[0,:]]
    thearrays = []
    for aposition, aPA in zip(target_positions[0,:], PAs[0,:]):
        logit.warning("aPA")
        logit.warning(aPA)
        projarray = testobs.get_projected_array(aposition, PA=aPA) 
        thearrays.append(projarray)
    thearrays = np.array(thearrays)
    thepistons = [testobs.get_projected_geometric_pistons(aposition) for aposition in target_positions[0,:]]
    thepistons = np.array(thepistons)
    
    perspective = True

    for seqindex in range(20):
        fig = pt.plot_pupil(thearrays[seqindex], thepistons=thepistons[seqindex])
        