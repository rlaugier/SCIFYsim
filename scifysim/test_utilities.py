


import unittest
import analysis
import scifysim as sf
import numpy as np

class map_tester(unittest.TestCase):
    def setUp(self):
        fname = "local_config/default_R200"
        target = "GJ 86 A"
        self.asim = sf.utilities.prepare_all(afile=fname, thetarget=target, update_params=True,
        seed=1, compnesate_chromatic=True)
    def tearDown(self):
        del self.asim
    def test_maps(self,):
        """
        Testing and comparing the 2 approaches to build transmission maps.
        """
        self.asim.build_all_maps_dask(mapcrop=0.2, mapres=10, )
        self.asim.persist_maps_to_disk()
        mapdask = self.asim.maps.compute()
        self.asim.build_all_maps(mapcrop=0.2, mapres=10, )
        mapnp = self.asim.maps.copy()
        print("They are the same map: ", np.allclose(mapnp, mapdask))
        return mapnp, mapdask 
