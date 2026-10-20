


import unittest
from scifysim import sources
import scifysim as sf
import numpy as np
from copy import copy
import astropy.units as u

from dummy import makesim, fname
asim = makesim(fname)


class Test_references(unittest.TestCase):
    def setUp(self):
        self.asim = copy(asim)
        self.avega = self.asim.context.avega
        self.wl = self.asim.lambda_science_range
        self.fd_unit = u.photon/u.s/u.m**2

    def tearDown(self):
        del self.asim
        del self.avega
        del self.wl

    def test_vega_flux_bin(self):
        """
            Check consistency of vega fluxes hard-coded separately
        
        * One in `sf.analysis` for the context
        * One in `sf.sources` for the simple calculation
        """
        vega_fd_direct = sources.vega_flux_bin(self.wl)
        vega_fd_context = self.fd_unit * self.asim.context.sflux_from_vegamag(0.)
        self.assertTrue(np.allclose(vega_fd_direct, vega_fd_context))

    def test_sflux_from_vegamag(self):
        refmag = 16.
        ref_sflux = self.fd_unit * np.array([28.04798598, 26.70469062, 25.44557988,
            24.26418052, 23.15460566, 22.11149396,
            21.12995582, 20.20552583, 19.33412059])
        flux_16 = self.fd_unit * self.asim.context.sflux_from_vegamag(refmag)
        flux_direct_16 = sources.mag2flux_bin(self.wl, refmag)
        self.assertTrue(np.allclose(flux_16, ref_sflux))
        self.assertTrue(np.allclose(ref_sflux, flux_direct_16))
        self.assertTrue(np.allclose(flux_16, flux_direct_16))

    def test_magT2flux_bin(self):
        refmag = 16.
        ref_sflux = sources.mag2flux_bin(self.wl, refmag)
        sflux_1000K = sources.magT2flux_bin(self.wl, refmag, 1000.)
        self.assertAlmostEqual(np.sum(ref_sflux), np.sum(sflux_1000K))
        self.assertIsInstance(sflux_1000K, u.quantity.Quantity)
        
    def test_flux_bin2mag(self):
        refmag = 16.
        ref_sflux = self.fd_unit * np.array([28.04798598, 26.70469062, 25.44557988,
            24.26418052, 23.15460566, 22.11149396,
            21.12995582, 20.20552583, 19.33412059])
        mymag = sources.flux_bin2mag(self.wl, ref_sflux)
        self.assertTrue(np.allclose(mymag, refmag))

    def test_ab_jansky(self):
        ref_ab = 16.
        ref_jy = u.jansky * 0.001445439770745928 # u.jansky *
        ajy = sources.ABmag2Jy(ref_ab)
        self.assertAlmostEqual(ajy, ajy)
        self.assertIsInstance(ajy, u.quantity.Quantity)
        self.assertIs(ajy.unit, u.jansky)
        self.assertIsInstance(ref_jy, u.quantity.Quantity)
        self.assertTrue(ref_jy.unit is u.jansky)
        aab = sources.Jy2ABmag(ref_jy)
        self.assertAlmostEqual(ref_ab, aab)



if __name__ == "__main__":
    unittest.main()
