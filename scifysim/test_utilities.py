


import unittest
import analysis
import scifysim as sf
import numpy as np


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.config_file = "config/default_R400.ini"
        self.config = sf.parsefile.parse_file(self.config_file)

    def tearDown(self):
        del self.config
        del self.config_file

    def test_get(self):
        theloc = self.config.get("configuration", "location")
        self.assertEqual(theloc, "Paranal")

    def test_getfloat(self):
        thefloat = self.config.getfloat("configuration", "beam_size")
        self.assertAlmostEqual(0.018, thefloat) 

    def test_getarray(self):
        thearray_float = self.config.getarray("configuration", "diam")
        self.assertTrue(isinstance(thearray_float, np.ndarray))
        refarray = np.array([8.0, 8.0, 8.0, 8.0])
        self.assertTrue(np.allclose(refarray, thearray_float))


class map_tester(unittest.TestCase):
    def setUp(self):
        fname = "config/default_R400.ini"
        target = "GJ 86 A"
        self.asim = sf.utilities.prepare_all(afile=fname, thetarget=target, update_params=True,
                                             seed=1, compensate_chromatic=True, verbose=False)

    def tearDown(self):
        del self.asim

    def test_maps(self,):
        """
        Testing and comparing the 2 approaches to build transmission maps.
        """
        self.asim.build_all_maps_dask(mapcrop=0.2, mapres=10, )
        # self.asim.persist_maps_to_disk()
        mapdask = self.asim.maps.compute()
        self.asim.build_all_maps(mapcrop=0.2, mapres=10, )
        mapnp = self.asim.maps.copy()
        self.assertTrue(np.allclose(mapnp, mapdask)) 


if __name__ == "__main__":
    unittest.main()
