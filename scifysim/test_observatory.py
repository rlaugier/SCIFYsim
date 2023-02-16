

import numpy as np

import unittest
import observatory
import astropy
import astroplan
#import astroplan
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
        self.thetime = astropy.time.Time("2020-10-20T3:00:00")
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
        self.anobs.point(self.thetime, self.atarget)
        
    def test_retro_build_observing_sequence_ends(self):
        anobs = observatory.observatory(statlocs=self.statlocs, config=self.config)
        obstimes = anobs.build_observing_sequence(self.times_ends, npoints=10, remove_daytime=False)
        print("#######################")
        print(isinstance([], list))
        self.assertTrue(hasattr(obstimes, "__len__"))
        self.assertTrue(hasattr(obstimes, "__getitem__"))
        self.assertEqual(len(obstimes), 10)
    def test_retro_build_observing_sequence_list(self):
        anobs = observatory.observatory(statlocs=self.statlocs, config=self.config)
        obstimes = anobs.build_observing_sequence(self.times, remove_daytime=False)
        self.assertTrue(hasattr(obstimes, "__len__"))
        self.assertTrue(hasattr(obstimes, "__getitem__"))
        self.assertEqual(len(obstimes), 20)

        
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
        print("warning: get_position untested")
        pass
    def test_retro_get_projected_geometric_pistons(self):
        print("warning get_projected_piston untested")
        pass


#    def test_init(self):
#        """
#        Currently set to fail:
#        We want possibility to setup from:
#        * Basic array
#        * Basic name for location
#        * 
#            
#        """
#        statlocs = np.array([[0., 0.],
#                            [1., 2.],
#                            [2., 2.]])
#        #aloc = astroplan.Observer.at_site("Paranal")
#        anobs = observatory.observatory(statlocs=statlocs, location="Paranal")
#        self.assertIsNotNone(anobs)
        



if __name__ == "__main__":
    unittest.main()