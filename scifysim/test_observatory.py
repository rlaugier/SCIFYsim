

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
        aconfig = parse_file("config/default_R400.ini")
        self.anobs = observatory.observatory(statlocs=statlocs, config=aconfig)
        self.atarget = astroplan.FixedTarget.from_name("Tau Cet")
        self.thetime = astropy.time.Time("2022-10-03T03:00:00")
        
    def tearDown(self):
        del self.anobs

    def test_retrocompatible(self):
        statlocs = np.array([[0., 0.],
                            [1., 2.],
                            [2., 2.]])
        aconfig = parse_file("config/default_R400.ini")
        anobs = observatory.observatory(statlocs=statlocs, config=aconfig)
        self.assertIsNotNone(anobs)

    def test_retroc_notstatlocs(self):
        aconfig = parse_file("config/default_R400.ini")
        anobs = observatory.observatory(config=aconfig)
        self.assertIsNotNone(anobs)
        
    def test_retro_point(self):
        self.anobs.point(self.thetime, self.atarget)
        
    def test_retro_build_observing_sequence(self):
        print("Warning build_observing_sequence untested")
        pass
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