

import numpy as np

import unittest
import observatory
import astropy
import astroplan
from copy import copy
# import astroplan
from scifysim.parsefile import parse_file

from dummy_space import makesim as makespacesim
from dummy_space import fname
from dummy_space import fname_interp

aspacesim = makespacesim(fname)
ainterpsim = makespacesim(fname_interp)


class TestObservatory(unittest.TestCase):
    def setUp(self):
        statlocs = np.array([[0., 0.],
                            [1., 2.],
                            [2., 2.]])
        aconfig = parse_file("config/test_default.ini")
        self.config = aconfig
        self.statlocs = statlocs
        self.anobs = observatory.observatory(statlocs=statlocs, config=aconfig)
        self.atarget = astroplan.FixedTarget.from_name("Tau Cet")
        self.thetime = astropy.time.Time("2020-10-20T03:00:00")
        self.theothertime = astropy.time.Time("2020-10-20T05:00:00")
        self.times_ends = (aconfig.get("target", "seq_start"), aconfig.get("target", "seq_end"))
        self.times = ["2020-10-20T00:00:00", "2020-10-20T03:00:00",
                         "2020-10-20T06:00:00", "2020-10-20T09:00:00"]
        
        
    def tearDown(self):
        del self.anobs

    def test_retrocompatible(self):
        statlocs = np.array([[0., 0.],
                            [1., 2.],
                            [2., 2.]])
        aconfig = parse_file("config/default_R400.ini")
        anobs = observatory.observatory(statlocs=statlocs, config=aconfig)
        self.assertIsNotNone(anobs)

    def test_retro_notstatlocs(self):
        aconfig = parse_file("config/default_R400.ini")
        anobs = observatory.observatory(config=aconfig)
        self.assertIsNotNone(anobs)
        
    def test_retro_point(self):
        """
        Checking that `obs.point` generates correct pointing in
        `astropy.coordinates.SkyCoord` format. Also checking that the values of
        `obs.altaz` and `obs.PA` change when pointing
        """
        self.anobs.point(self.thetime, self.atarget)
        altaz_1 = self.anobs.altaz
        PA_1 = self.anobs.PA
        self.anobs.point(self.theothertime, self.atarget)
        altaz_2 = self.anobs.altaz
        PA_2 = self.anobs.PA
        self.assertIsInstance(altaz_1, astropy.coordinates.SkyCoord)
        self.assertIsInstance(altaz_1.alt, astropy.coordinates.Latitude)
        self.assertIsInstance(altaz_1.az, astropy.coordinates.Longitude)
        self.assertIsInstance(PA_1, astropy.coordinates.Angle)
        self.assertFalse(np.allclose(altaz_1.alt, altaz_2.alt))
        self.assertFalse(np.allclose(altaz_1.az, altaz_2.az))
        self.assertFalse(np.allclose(PA_1, PA_2))
        
    def test_retro_build_observing_sequence_ends(self):
        anobs = observatory.observatory(statlocs=self.statlocs, config=self.config)
        obstimes = anobs.build_observing_sequence(self.times_ends, npoints=10, remove_daytime=False)
        print("#######################")
        print(isinstance([], list))
        self.assertTrue(hasattr(obstimes, "__len__"))
        self.assertTrue(hasattr(obstimes, "__getitem__"))
        self.assertIsInstance(obstimes[0], astropy.time.Time)
        self.assertEqual(len(obstimes), 10)

    def test_retro_build_observing_sequence_list(self):
        anobs = observatory.observatory(statlocs=self.statlocs, config=self.config)
        obstimes = anobs.build_observing_sequence(self.times, remove_daytime=False)
        self.assertTrue(hasattr(obstimes, "__len__"))
        self.assertTrue(hasattr(obstimes, "__getitem__"))
        self.assertIsInstance(obstimes[0], astropy.time.Time)
        self.assertEqual(len(obstimes), len(self.times))

    def test_behavior_PA_projected_array(self):
        # aPA = self.anobs.PA
        # Prefered way:
        anobs = copy(self.anobs)
        anobs.point(self.thetime, self.atarget)
        aproj = anobs.get_projected_array()
        # Legacy way:
        aproj_leg = anobs.get_projected_array(taraltaz=anobs.altaz,
                                            PA=anobs.PA)
        # Alternate way:
        aproj_alt = anobs.get_projected_array(taraltaz=anobs.altaz,
                                            PA=anobs.PA.rad)
        self.assertTrue(np.allclose(aproj, aproj_leg))
        self.assertTrue(np.allclose(aproj, aproj_alt))
        self.assertTrue(np.allclose(aproj_alt, aproj_leg))

    def test_config_number_of_points_times(self):
        self.assertEqual(self.config.getint("target", "n_points"), 7)
        
    def test_retro_positions(self):
        analtaz = self.anobs.get_positions(self.atarget, self.thetime)
        print("""
            ##########
            #TODO#####
            ##########
            Update the docstring: with return
            a SkyCoord object (which contains AltAz.)""")
        self.assertTrue(isinstance(analtaz, astropy.coordinates.SkyCoord))

    def test_retro_get_projected_array(self):
        """
        simplifying this would be interesting
        """
        # self.anobs.point(self.thetime, self.atarget) ## could use point instead
        altaz = self.anobs.observatory_location.altaz(target=self.atarget,
                                                   time=self.thetime)
        proj_array = self.anobs.get_projected_array(altaz, PA=False)
        ref_proj = np.array([[ 0.        ,  0.        ],
                             [ 1.58675864, -1.45079393],
                             [ 1.27391078, -2.32541767]])
        self.assertTrue(np.allclose(proj_array, ref_proj))

    def test_retro_get_projected_geometric_pistons(self):
        # self.anobs.point(self.thetime, self.atarget) ## Could use point instead
        altaz = self.anobs.observatory_location.altaz(target=self.atarget,
                                                   time=self.thetime)
        pistons = self.anobs.get_projected_geometric_pistons(altaz,)
        ref_pistons = np.array([[ 0.        ],
                                 [-0.614324  ],
                                 [-0.98467457]])
        self.assertTrue(np.allclose(pistons, ref_pistons))

    def test_getUV(self):
        anobs = copy(self.anobs)
        anobs.point(self.times[0], self.atarget)
        self.assertTrue(hasattr(anobs, "bl_mat"))
        n = anobs.statlocs.shape[0]
        self.assertEqual(anobs.bl_mat.shape[0], (n*(n-1))//2)
        self.assertEqual(anobs.bl_mat.shape[1], n)
        self.assertTrue(hasattr(anobs, "uv"))
        self.assertEqual(anobs.uv.shape[0], (n*(n-1))//2)
        self.assertEqual(anobs.uv.shape[1], 2)
        self.assertTrue(np.max(anobs.uv)<= np.max(2*np.abs(anobs.statlocs)))
        



#    def test_init(self):
#        """
#        Currently set to fail:
#        We want possibility to setup from:
#        * Basic array
#        * Basic name for location
#        * 
#            
#        """
#        statlocs = np.arrAlmostEqualay([[0., 0.],
#                            [1., 2.],
#                            [2., 2.]])
#        #aloc = astroplan.Observer.at_site("Paranal")
#        anobs = observatory.observatory(statlocs=statlocs, location="Paranal")
#        self.assertIsNotNone(anobs)
        

class TestSpaceObservatory(unittest.TestCase):
    def setUp(self):
        # self.aspaceobs = observatory.SpaceObservatory()
        statlocs = np.array([[-20., -10., 100.],
                            [-20., +10., 100.],
                            [+20., -10., 100.],
                            [+20., +10., 100.]])
        aconfig = parse_file("config/test_space_default.ini")
        self.config = aconfig
        self.statlocs = statlocs
        self.aspaceobs = observatory.SpaceObservatory(statlocs=statlocs, config=aconfig)
        self.asim = copy(aspacesim)
        self.interpsim = copy(ainterpsim)

    def tearDown(self):
        del self.aspaceobs
        del self.config
        del self.statlocs

    def test_getUV(self):
        aspaceobs = copy(self.asim.obs)
        aspaceobs.point(self.asim.sequence[0])
        self.assertTrue(hasattr(aspaceobs, "bl_mat"))
        self.assertEqual(aspaceobs.bl_mat.shape[0], 6)
        self.assertEqual(aspaceobs.bl_mat.shape[1], 4)
        self.assertTrue(hasattr(aspaceobs, "uv"))
        self.assertEqual(aspaceobs.uv.shape[0], 6)
        self.assertEqual(aspaceobs.uv.shape[1], 2)
        self.assertTrue(np.max(aspaceobs.uv)<= np.max(2*np.abs(aspaceobs.statlocs)))


    def test_spaceobs_has_methods(self):
        self.assertTrue(hasattr(self.aspaceobs, "point"))
        self.assertTrue(hasattr(self.aspaceobs, "build_observing_sequence"))
        self.assertTrue(hasattr(self.aspaceobs, "get_position"))
        self.assertTrue(hasattr(self.aspaceobs, "get_projected_array"))
        self.assertTrue(hasattr(self.aspaceobs, "get_projected_geometric_pistons"))

    def test_spaceobs_point(self):
        aspaceobs = copy(self.asim.obs)
        aspaceobs.point(self.asim.sequence[0])
        x_A_t_0 = aspaceobs.x_A_t
        aspaceobs.point(self.asim.sequence[1])
        x_A_t_1 = aspaceobs.x_A_t
        self.assertFalse(np.allclose(x_A_t_0, x_A_t_1))
        self.assertIsInstance(x_A_t_0, np.ndarray)
        self.assertEqual(x_A_t_0.shape, (aspaceobs.n_tel, 3))

    def test_time2t(self):
        aspaceobs = copy(self.asim.obs)
        ref_t = 1200.0
        a_t = aspaceobs.time2t(self.asim.sequence[2])
        self.assertAlmostEqual(ref_t, a_t)
        self.assertIsInstance(a_t, float)

    def test_spaceobs_motion_rotation(self):
        aspaceobs = copy(self.asim.obs)
        atime = self.asim.sequence[2]
        self.assertEqual(aspaceobs.motion_type, "rotation")
        self.assertIsInstance(atime, observatory.Time)
        at = (atime.to_value("unix") - self.asim.sequence[0].to_value("unix"))
        self.assertIsInstance(at, float)
        axat_default = aspaceobs.motion(at)
        axat_same = aspaceobs.motion(at, loc_array=aspaceobs.statlocs)
        axat_other = aspaceobs.motion(at, loc_array=self.statlocs)
        self.assertIsInstance(axat_default, np.ndarray)
        self.assertTrue(np.allclose(axat_default, axat_same))
        self.assertFalse(np.allclose(axat_default, axat_other))
        self.assertFalse(np.allclose(axat_same, axat_other))
        # Testing the full_output:
        axat, theta, aR = aspaceobs.motion(at, full_output=True)
        self.assertIsInstance(axat, np.ndarray)
        self.assertIsInstance(theta, float)
        self.assertIsInstance(aR, np.ndarray)
        self.assertEqual(aR.shape, (3,3))
        self.assertEqual(axat.shape, (aspaceobs.n_tel, 3))

    def test_spaceobs_motion_interp(self):
        myinterpsim = copy(self.interpsim.obs)
        atime = self.asim.sequence[2]
        self.assertEqual(myinterpsim.motion_type, "interpolation")
        self.assertIsInstance(atime, observatory.Time)
        at = (atime.to_value("unix") - self.asim.sequence[0].to_value("unix"))
        self.assertIsInstance(at, float)
        axat_default = myinterpsim.motion(at)
        axat_same = myinterpsim.motion(at, loc_array=myinterpsim.statlocs)
        axat_other = myinterpsim.motion(at, loc_array=self.statlocs)
        self.assertIsInstance(axat_default, np.ndarray)
        self.assertTrue(np.allclose(axat_default, axat_same))
        self.assertTrue(np.allclose(axat_default, axat_other))
        self.assertTrue(np.allclose(axat_same, axat_other))

        





if __name__ == "__main__":
    unittest.main()