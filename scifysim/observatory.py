
import numpy as np
import sympy as sp
from scipy.interpolate import interp1d

import kernuller

from scifysim import utilities


from astropy.time import Time
import astropy.units as u

import astroplan
from astroplan import plots
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun

import logging

logit = logging.getLogger(__name__)


"""
Basic usage:

.. code-block::

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
        
        Parameters:
        
        * statlocs : The station locations (optional)
          (east, north) for each aperture shape is (Na, 2)
        * location : An astropy.coordinatEarthLocation (default = Paranal)
          example: myloc = astroplan.Observer.at_site("Paranal", timezone="UTC")
        * multi_dish : When True, the geometry of the pupil varies depending on the relative position
          of the target, expecially in terms of projection of the pupil on the plane
          orthogonal to the line of sight.
          When False, (not implemented yet) the array is expected to be always facing
          the line of sight, as is the case for example with a systme like GLINT. 
        * config    : A parsed config object.
        * verbose  : Activate verbosity in the log
        
        """
        self.verbose = verbose
        self.config = config
        if location is None:
            location = self.config.get("configuration", "location")
            self.observatory_location = astroplan.Observer.at_site(location, timezone="UTC")
        else :
            self.observatory_location = astroplan.Observer.at_site(location, timezone="UTC")
        
        
        # self.array_config = self.config.get("configuration", "config")
        raw_array, array_config = utilities.get_raw_array(self.config)
        self.array_config = array_config; del array_config
        # raw_array = eval("kernuller.%s"%(self.array_config))
        self.order = self.config.getarray("configuration", "order").astype(np.int16)
        if statlocs is None:
            self.statlocs = raw_array[self.order]
        else:
            self.statlocs = statlocs
        
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
        
    def point(self, obstime, target):
        """
        Points the array towards the target, updating its position angle (PA) and altaz (used for airmass).
        These are later used by other methods to compute the projection of the array.
        
        **Parameters:**
        
        * obstime   : The astropy.time.Time object corresponding to the moment of observation
          (usually picked from the sequence provided by self.build_observing_sequence())
        * target    : The astroplan.FixedTarget object of interest. Usually resolved in the 
          self.build_observign_sequence() routine with astroplan.FixedTarget.from_name()
        """
        self.altaz = self.observatory_location.altaz(target=target,
                                                   time=obstime)
        self.PA = self.observatory_location.parallactic_angle(obstime, target=target)
        

        
    def build_observing_sequence(self, times=None,
                            npoints=None, remove_daytime=False):
        """
        
        **Parameters:**
        
        * times : a list of UTC time strings ("2020-04-13T00:00:00")
          that define an interval (if npoints is not None),
          or the complete list of times (if npoints is None)
        * npoints : The number of samples to take on the interval
          None means that the times is the whole list
        * remove_daytime : Whether to remove the points that fall during the day
        
        **Returns** the series of obstimes needed to compute the altaz positions
        
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
            obstimes = np.array([Time(times[i]) for i in range(len(times))])
            
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
        
        **Parameters:**
        
        * targets: A list of SkyCoord objects 
        * obstimes: A list of astropy.Times to make the observations
        
        **Returns** the astropy.coordinates.AltAz for a given target
        """
        taraltaz = self.observatory_location.altaz(target=targets,
                                                   time=obstimes,
                                                   grid_times_targets=True)
        tarPA = self.observatory_location.parallactic_angle(obstimes, target=targets)
        return taraltaz#, tarPA
    
    def get_position(self, target, time, grid_times_targets=False):
        """        
        **Parameters:**
        
        * target:   one or a list of of targets
        * obstimes: one or a list of astropy.Times to make the observations
        
        **Returns** the ``astropy.coordinates.AltAz`` for a given target
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
        
    def get_projected_array(self, taraltaz=None, PA=True, loc_array=None):
        """
        **Parameters:**
        
        * taraltaz : the astropy.coordinates.AltAz of the target
        * PA       : parallactic angle to derotate 
        * loc_array: the array of points to use (None: use self.statlocs)
        
        **Returns** the new coordinates for the projected array
        """
        if taraltaz is None:
            taraltaz = self.altaz
        if loc_array is None:
            loc_array = self.statlocs
        arrayaz = self.R((180 - taraltaz.az.value)*np.pi/180).dot(loc_array.T).T
        altazarray = self.P(taraltaz.alt.value * np.pi/180).dot(arrayaz.T).T
        # if PA is True:
        #     PA = self.PA
        #     radecarray = self.R(PA.rad).dot(altazarray.T).T
        #     newarray = radecarray
            
        if PA is False: 
            newarray = altazarray
        elif PA is True:
            radecarray = self.R(self.PA.rad).dot(altazarray.T).T
            newarray = radecarray
            
        elif isinstance(PA, u.quantity.Quantity):
            radecarray = self.R(PA.rad).dot(altazarray.T).T
            newarray = radecarray
        elif isinstance(PA, float):
            # raise TypeError("Provide PA as a quantity in [rad]")
            radecarray = self.R(u.rad * PA).dot(altazarray.T).T
            newarray = radecarray
        if PA is None: 
            raise AttributeError("This path is deprecated: ")
            newarray = altazarray
            
        if self.verbose:
            logit.debug("=== AltAz position:")
            logit.debug("az "+ str(taraltaz.az.value -180))
            logit.debug("alt "+ str(taraltaz.alt.value))
            logit.debug("old array "+ str(loc_array))
            logit.debug("new array "+ str(newarray))
        return newarray
    def get_projected_geometric_pistons(self, taraltaz=None):
        """
        **Parameters:**
        
        * taraltaz : the astropy.coordinates.AltAz of the target
        
        **Returns** the geomtric piston resutling from the pointing
        of the array.
        """
        if taraltaz is None:
            taraltaz = self.altaz
        arrayaz = self.R((180 - taraltaz.az.value)*np.pi/180).dot(self.statlocs.T).T
        pistons = self.C(taraltaz.alt.value * np.pi/180).dot(arrayaz.T).T
        if self.verbose:
            logit.debug("=== pistons:")
            logit.debug("az "+ str(taraltaz.az.value -180))
            logit.debug("alt "+ str(taraltaz.alt.value))
            logit.debug("old array "+ str(self.statlocs))
            logit.debug("new array "+ str(pistons))
        return pistons

class SpaceObservatory(observatory):
    def __init__(self, statlocs=None, location=None,
            verbose=False, multi_dish=True, config=None):
        """
        
        Parameters:
        
        * statlocs : The station locations (optional)
          (east, north) for each aperture shape is (Na, 2)
        * location : An astropy.coordinatEarthLocation (default = Paranal)
          example: myloc = astroplan.Observer.at_site("Paranal", timezone="UTC")
        * multi_dish : When True, the geometry of the pupil varies depending on the relative position
          of the target, expecially in terms of projection of the pupil on the plane
          orthogonal to the line of sight.
          When False, (not implemented yet) the array is expected to be always facing
          the line of sight, as is the case for example with a systme like GLINT. 
        * config    : A parsed config object.
        * verbose  : Activate verbosity in the log
        
        """
        self.verbose = verbose
        self.config = config
        if location is None:
            location = self.config.get("configuration", "location")
        self.observatory_location = location
        self.dummy_location = astroplan.Observer.at_site("paranal", timezone="UTC")
        self.n_tel = self.config.getint("configuration", "n_dish")
        
        # self.array_config = self.config.get("configuration", "config")
        raw_array, array_config = utilities.get_raw_array(self.config)
        self.array_config = array_config; del array_config
        # raw_array = eval("kernuller.%s"%(self.array_config))
        self.order = self.config.getarray("configuration", "order").astype(np.int16)
        if statlocs is None:
            self.statlocs = raw_array[0][self.order]
        else:
            self.statlocs = statlocs
        
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
        self.x_M = np.zeros(3)
        self.P_M = np.diag([1, 1, 0])
        self.rotation_rate = self.config.getfloat("configuration", "rotation_rate")
        self.time_0 = Time(self.config.get("target", "seq_start"))
        self.t_0 = self.time_0.to_value("unix")
        self.motion_type = self.config.get("configuration", "motion_type")
        if self.motion_type == "rotation":
            self.motion = self.rotation
        elif self.motion_type == "interpolation":
            self.motion = self.interpolation
            self.interp_t = self.config.getarray("configuration", "array_time_steps")
            self.interp_y = raw_array
            self.interpolation_function = interp1d(self.interp_t,
                                self.interp_y, axis=0,
                                bounds_error=False,
                                fill_value="extrapolate")
            self.motion = self.interpolation
        # Initializing by pointing to instant 0
        self.point(self.time_0)

    def time2t(self, time):
        """
            Convenience conversion from astropy.Time object
        to seconds since the start of observations used internally
        """
        return time.to_value("unix") - self.t_0

    def interpolation(self, t, loc_array=None, full_output=False):
        """
            **Arguments:**
        * t    : time [s]
        * loc_array : irrelevant hereo
        * full_output : irrelevant here
        
        **returns:**
        * x_A_t  [m] The 3D location of the array of apertures

        **Computes:**
        * x_A_t   [m] The 3D location of the array of apertures
        """
        if loc_array is not None:
            logit.warning("Passed loc_array irrelevant argument")
        if full_output is not None:
            logit.warning("Passed full_output: irrelevant argument")
        return self.interpolation_function(t)

    def rotation(self, t, loc_array=None, full_output=False):
        """
            Einsum:
        * i : Input space
        * o : Output space
        * a : Aperture
        """
        if loc_array is None:
            loc_array = self.statlocs
        theta = self.rotation_rate * t
        R_rotation = basic_z_rotation(theta)
        assert R_rotation.shape == (3,3), f"Got shape R_rotation {R_rotation.shape}"
        assert loc_array.shape == (self.n_tel, 3,), f"Got shape x_M {loc_array.shape}"
        x_A_t = np.einsum("o i, a i -> a o", R_rotation, loc_array)
        if full_output:
            return x_A_t, theta, R_rotation
        else:
            return x_A_t
        
    def point(self, time, target=None):
        """
            Refreshes the parameters related to pointing and motion
        in particular:
        * `self.x_A_t`
        * `self.R_rotation` when relevant
        """
        t = self.time2t(time)
        motion_result = self.motion(t, full_output=True)
        if self.motion_type == "rotation":
            self.x_A_t = motion_result[0]
            self.theta = motion_result[1]
            self.R_rotation = motion_result[2]
        else:
            self.x_A_t = motion_result

    # def build_observing_sequence(self):
    #     pass

    def get_position(self, target, time, grid_times_targets=False):
        dummy_taraltaz = self.dummy_location.altaz(time, target=target,
                            grid_times_targets=grid_times_targets)
        t = time.to_value("unix") - self.t_0
        x_A_t = self.motion(t)
        aPA = np.arctan2(x_A_t[0,1], x_A_t[0,0])
        return dummy_taraltaz, aPA

    def get_projected_array(self, taraltaz=None, time=None, PA=None, loc_array=None):
        if time is not None:
            self.point(time)
        if loc_array is None:
            loc_array = self.x_A_t
        x_P = np.einsum("o i , a i -> a o", self.P_M, loc_array)
        return x_P

    def get_projected_geometric_pistons(self):
        """
            Computes the distance traveled past the reference plane:
        $P_A - A + AM$
        """
        P_A = self.get_projected_array()
        P_A_A = self.x_A_t  - P_A
        norm_PAA = np.linalg.norm(P_A_A, axis=1)
        M_A = self.x_M - self.x_A_t
        norm_M_A = np.linalg.norm(M_A, axis=1)
        optical_path = (norm_M_A + norm_PAA)[:,None]
        assert optical_path.shape == (self.n_tel, 1), f"shape of optical path {norm_M_A.shape}"
        return optical_path
        
def basic_interpolation(t,):
    pass


    
def basic_z_rotation(theta):
    """
        basic rotation matrix around z
    """
    M_R = np.array([[np.cos(theta), np.sin(theta), 0,],
                    [-np.sin(theta), np.cos(theta),0,],
                    [0, 0, 1]])
    return M_R

def test_observatory(tarname="Tau cet",
                     startend=["2020-10-20T00:00:00", "2020-10-20T10:00:00"],
                     nblocs=20):
    
    """
    Runs some test for the core functions of observatory
    
    **Parameters:**
    
    * tarname    : The name of the target to use
    * startend   : A list or tuple of two time strings inspressed in UTC matching the format:
      "2020-10-20T00:00:00"
    * nblocs     : The number of observing blocs to compute
    """
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
        