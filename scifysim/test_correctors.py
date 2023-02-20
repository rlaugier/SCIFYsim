

import unittest
from scifysim import correctors, config
import numpy as np

class test_legacy_integration_corrector(unittest.TestCase):

    def setUp(self):
        # fname = "config/test_default.ini"
        # target = "GJ 86 A"
        # self.asim = sf.utilities.prepare_all(afile=fname,
        #                 thetarget=target, update_params=True,
        #                 seed=1, compensate_chromatic=True,
        #                 verbose=False)
        # self.asim.point(self.asim.sequence[0], self.asim.target,
        #                 refresh_array=False, disp_override=None)
        from dummy import asim
        self.asim = asim
        self.asim.point(self.asim.sequence[0], self.asim.target,
                        refresh_array=False, disp_override=None)
        ref_corrector_b = np.array([ 0.00000000e+00, -6.77618302e-08, -8.12038610e-08, -1.48966270e-07])
        ref_corrector_c = np.array([ 0.        , -0.00014417, -0.00021132, -0.00035549])
        ref_corrector_dcomp = np.array([-0.        ,  0.00020676,  0.00030305,  0.00050981])
        self.assertTrue(np.allclose(self.asim.corrector.b, ref_corrector_b))
        self.assertTrue(np.allclose(self.asim.corrector.c, ref_corrector_c))
        print(self.asim.corrector.dcomp)
        self.assertTrue(np.allclose(self.asim.corrector.dcomp, ref_corrector_dcomp))
        
    def tearDown(self):
        del self.asim


    def test_get_phasor(self):
        alpha = self.asim.corrector.get_phasor(self.asim.lambda_science_range)
        self.assertTrue(isinstance(alpha, np.ndarray))
        self.assertTrue(alpha.dtype is np.dtype(np.complex128))






class test_corrector_from_file(unittest.TestCase):
	def setUp(self):
        self.corrector = correctors.corrector(config="config/test_config.ini")
    
		pass
	def tearDown(self):
		pass
	def test_myFunction(self):
		self.assertAlmostEqual(1., 1.)




