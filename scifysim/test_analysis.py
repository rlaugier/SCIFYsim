
import unittest
import analysis
import scifysim as sf
import numpy as np
from copy import copy
from astropy import units

from dummy import makesim, fname
asim = makesim(fname)

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
        

class test_BasicETC(unittest.TestCase):
    def setUp(self):
        self.asim = copy(asim)
        self.etc = analysis.BasicETC(self.asim)

    def tearDown(self):
        del self.asim
        del self.etc

    def test_types(self):
        self.assertIsInstance(self.etc.pdiam, units.quantity.Quantity)
        self.assertEqual(self.etc.pdiam.unit, units.m)
        self.assertIsInstance(self.etc.odiam, units.quantity.Quantity)
        self.assertEqual(self.etc.pdiam.unit, units.m)
        self.assertIsInstance(self.etc.S_collecting,
                        units.quantity.Quantity)
        self.assertEqual(self.etc.S_collecting.unit, units.m**2)
        self.assertIsInstance(self.etc.eta, units.quantity.Quantity)
        self.assertEqual(self.etc.eta.unit,
                        units.electron/units.photon)
        self.assertIsInstance(self.etc.dit_0, units.quantity.Quantity)
        self.assertEqual(self.etc.dit_0.unit,
                        units.s)
        self.assertIsInstance(self.etc.contribs, units.quantity.Quantity)
        self.assertEqual(self.etc.contribs.unit, units.ph/units.s)
        self.assertIsInstance(self.etc.contribs_current, units.quantity.Quantity)
        self.assertEqual(self.etc.contribs_current.unit, units.electron/units.s)
        self.assertIsInstance(self.etc.contribs_variance_rate, units.quantity.Quantity)
        self.assertEqual(self.etc.contribs_variance_rate.unit, units.electron**2/ units.s)


    def test_planet_photons(self):
        ref_dit = 10.
        ref_mag = 13.
        ref_T = 800.
        asig = self.etc.planet_photons(ref_mag, ref_dit, T=ref_T)
        self.assertIsInstance(asig, units.quantity.Quantity)
        self.assertEqual(asig.shape, self.asim.lambda_science_range.shape)
        asig = self.etc.planet_photons(ref_mag, ref_dit)
        self.assertIsInstance(asig, units.quantity.Quantity)
        self.assertEqual(asig.shape, self.asim.lambda_science_range.shape)
        
    def test_signal_noise(self):
        ref_dit = 10.
        ref_mag = 13.
        ref_T = 800.
        asig, anoise = self.etc.show_signal_noise(
            ref_mag, dit=ref_dit, T=ref_T,
            verbose=False, plot=False,
            show=False
        )
        
        othersig, othernoise = self.etc.show_signal_noise(
            ref_mag, dit=2*ref_dit, T=ref_T,
            verbose=False, plot=False,
            show=False
        )
        self.assertTrue(np.allclose(2 * asig, othersig))
        self.assertTrue(np.allclose(np.sqrt(2) * anoise, othernoise))
        # self.assertTrue(asig.unit, units.)
        
    def test_signal_noise_plot(self):
        ref_dit = 10.
        ref_mag = 13.
        ref_T = 800.
        asig, anoise, afig = self.etc.show_signal_noise(
            ref_mag, dit=ref_dit, T=ref_T,
            verbose=False, plot=True,
            show=False
        )
        self.assertIsInstance(asig, units.quantity.Quantity)

    def test_unit_gainmap(self):
        mysim = copy(asim)
        mysim.build_all_maps(mapcrop=0.01, mapres=20)
        againmap = mysim.gain_map
        self.assertIsInstance(againmap, units.quantity.Quantity)
        self.assertEqual(againmap.unit, units.m**2 * units.electron \
                                    /units.photon)
        ref_dit = 10.
        ref_mag = 13.
        ref_T = 800.
        asig = self.etc.planet_photons(ref_mag, ref_dit, T=ref_T)
        self.assertEqual((againmap * asig[None,:,None,None,None]).unit, units.electron)

