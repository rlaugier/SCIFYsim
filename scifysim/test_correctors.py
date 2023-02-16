

import unittest
import scifysim as sf

#import correctors
import numpy as np






class test_legacy_integration_corrector(unittest.TestCase):

	def setUp(self):
		fname = "config/test_default.ini"
		target = "GJ 86 A"
		self.asim = sf.utilities.prepare_all(afile=fname,
						thetarget=target, update_params=True,
				        seed=1, compensate_chromatic=True,
						verbose=False)
		self.asim.point(self.asim.sequence[0], self.asim.target,
						refresh_array=False, disp_override=None)
	def tearDown(self):
		del self.asim


	def test_get_phasor(self):
		alpha = self.asim.corrector.get_phasor(self.asim.lambda_science_range)
		self.assertTrue(isinstance(alpha, np.ndarray))
		self.assertTrue(alpha.dtype is np.dtype(np.complex128))

