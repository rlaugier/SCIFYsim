

import numpy as np

import unittest
import observatory
import astropy
import astroplan
# import astroplan
from scifysim.parsefile import parse_file


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
        proj_array = self.anobs.get_projected_array(altaz,)
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
        



if __name__ == "__main__":
    unittest.main()