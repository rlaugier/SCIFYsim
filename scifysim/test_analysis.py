
import unittest
import analysis
import scifysim as sf
import numpy as np
from copy import copy

from dummy import asim

class test_simple_context(unittest.TestCase):
	def setUp(self):
		self.asim = copy(asim)
		self.a_simp_cont = analysis.simplified_context(self.asim.context)

	def tearDown(self):
		del self.asim

	def test_create(self):
		a = analysis.simplified_context(self.asim.context)
		del a

	def test_basic_flux(self):
		a = self.asim.context.sflux_from_vegamag(6.)
		b = self.a_simp_cont.sflux_from_vegamag(6.)
		self.assertTrue(np.allclose(a, b))

	def test_basic_vegamag(self):
		a = self.asim.context.vegamag_from_ss(6.)
		b = self.a_simp_cont.vegamag_from_ss(6.)
		self.assertTrue(np.allclose(a, b))

	def test_basic_mags_of_sim(self):
		a = self.asim.context.get_mags_of_sim(self.asim)
		b = self.a_simp_cont.get_mags_of_sim(self.asim)
		self.assertTrue(np.allclose(a, b))
		
	
