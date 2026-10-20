

import unittest
from scifysim import correctors, parsefile
import numpy as np
from scifysim import wet_atmo
from copy import copy

config = parsefile.parse_file("config/test_default.ini")

from dummy import makesim, fname
asim = makesim(fname)

bconf = parsefile.parse_file(fname)
bconf.set("corrector", "mode", "znse")
bsim = makesim(bconf)



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
        self.asim = copy(asim)
        self.asim.point(self.asim.sequence[0], self.asim.target,
                        refresh_array=False, disp_override=None)
        ref_corrector_b = np.array([ 0.00000000e+00, -6.77618302e-08,
                                    -8.12038610e-08, -1.48966270e-07])
        ref_corrector_c = np.array([ 0.        , -0.00014417,
                                     -0.00021132, -0.00035549])
        ref_corrector_dcomp = np.array([-0.        ,  0.00020676,
                                          0.00030305,  0.00050981])
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


class test_corrector_creation_from_file(unittest.TestCase):
    def setUp(self):
        self.lambs = np.linspace(3.5e-6, 4.0e-6, 9)
        self.config = config

    def tearDown(self):
        del self.config

    def test_creation(self):
        self.config = config
        myair = wet_atmo(self.config)
        co2_for_compensation = wet_atmo(temp=myair.temp, pres=myair.pres,
                                           rhum=0., co2=1.0e6)
        corrector_single = correctors.corrector(self.config, self.lambs, 
                                model_comp=myair)
        corrector_co2 = correctors.corrector(self.config, self.lambs, 
                                model_comp=myair,
                                model_material2=co2_for_compensation)
        # At creation they should have the default parameters
        self.assertTrue(np.allclose(corrector_single.a, np.ones(4)))
        self.assertTrue(np.allclose(corrector_single.b, np.zeros(4)))
        self.assertTrue(np.allclose(corrector_single.c, np.zeros(4)))
        self.assertTrue(np.allclose(corrector_single.e, np.zeros(4)))
        self.assertTrue(np.allclose(corrector_co2.a, np.ones(4)))
        self.assertTrue(np.allclose(corrector_co2.b, np.zeros(4)))
        self.assertTrue(np.allclose(corrector_co2.c, np.zeros(4)))
        self.assertTrue(np.allclose(corrector_co2.e, np.zeros(4)))
        

class test_corrector_behavior_from_file(unittest.TestCase):
    def setUp(self):
        self.config = config
        myair = wet_atmo(self.config)
        co2_for_compensation = wet_atmo(temp=myair.temp, pres=myair.pres,
                                           rhum=0., co2=1.0e6)
        self.lambs = np.linspace(3.5e-6, 4.0e-6, 9)
        self.corrector_single = correctors.corrector(self.config, self.lambs, 
                                model_comp=myair)
        self.corrector_co2 = correctors.corrector(self.config, self.lambs, 
                                model_comp=myair,
                                model_material2=co2_for_compensation)

    def tearDown(self):
        del self.config
        del self.lambs
        del self.corrector_single
        del self.corrector_co2
        
    def test_n_values(self):
        ref_lambs = np.array([3.5000e-06, 3.5625e-06, 3.6250e-06,
                            3.6875e-06, 3.7500e-06, 3.8125e-06,
                            3.8750e-06, 3.9375e-06, 4.0000e-06])
        self.assertTrue(np.allclose(self.lambs, ref_lambs))
        ref_n_plate = np.array([2.43514826, 2.43488049, 2.43461819,
                             2.43436084, 2.43410796, 2.43385911,
                             2.4336139 , 2.43337194, 2.4331329 ])
        the_nplates_single = self.corrector_single.nplate(self.lambs)
        self.assertTrue(np.allclose(ref_n_plate, the_nplates_single))
        the_nplates_co2 = self.corrector_co2.nplate(self.lambs)
        self.assertTrue(np.allclose(ref_n_plate, the_nplates_co2))

    def test_dcomp(self):
        ref_dcomp = np.array([-0.        ,  0.00020676,  0.00030305,  0.00050981])
        the_dcomp_single = self.corrector_single.get_dcomp(np.array([ 0.        , -0.00014417, -0.00021132, -0.00035549]))
        self.assertTrue(np.allclose(the_dcomp_single, ref_dcomp))
        the_dcomp_co2 = self.corrector_single.get_dcomp(np.array([ 0.        , -0.00014417, -0.00021132, -0.00035549]))
        self.assertTrue(np.allclose(the_dcomp_co2, ref_dcomp))

    # def test_dcomp_from_vector(self):
    #     self.corrector_single.get_dcomp_from_vector()

class test_sellmeier_linb(unittest.TestCase):
    def setUp(self):
        self.lamb_ref = 3.758e-6
        # Values pulled from https://refractiveindex.info/?shelf=main&book=LiNbO3&page=Zelmon-o
        self.n_e_linb_ref = 2.06665321225504
        self.n_o_linb_ref = 2.12698094060923
        self.bs_e_test = np.array([2.9804,
                                  0.02047,
                                  0.5981,
                                  0.0666,
                                  8.9543,
                                  416.08])
        self.b_e_test = np.array([2.9804,
                                  0.5981,
                                  8.9543])
        self.c_e_test = np.array([0.02047,
                                  0.0666,
                                  416.08])


    def test_make_sellmeier_single(self):
        mylinb = correctors.material_sellmeier(self.bs_e_test)
        # Note: by default, add should be 0
        self.assertAlmostEqual(self.n_e_linb_ref, 1+mylinb.get_n(self.lamb_ref))
        self.assertAlmostEqual(self.n_e_linb_ref, mylinb.get_n(self.lamb_ref, add=1))
        
    def test_make_sellmeier_double(self):
        mylinb = correctors.material_sellmeier(bs=self.b_e_test, cs=self.c_e_test)
        # Note: by default, add should be 0
        self.assertAlmostEqual(self.n_e_linb_ref, 1+mylinb.get_n(self.lamb_ref))
        self.assertAlmostEqual(self.n_e_linb_ref, mylinb.get_n(self.lamb_ref, add=1))

    def test_builtin_linb(self):
        self.assertAlmostEqual(self.n_e_linb_ref, correctors.linb_e.get_n(self.lamb_ref, add=1))
        self.assertAlmostEqual(self.n_o_linb_ref, correctors.linb_o.get_n(self.lamb_ref, add=1))


class test_correcor_in_simulator(unittest.TestCase):
    def setUp(self):
        self.asim = copy(asim)

    def tearDown(self):
        del self.asim

    def test_tuning_convergence(self):
        ref_b = np.array([ 0.00000000e+00, -6.77618302e-08, -8.12038610e-08, -1.48966270e-07])
        ref_c = np.array([ 0.        , -0.00014417, -0.00021132, -0.00035549])
        self.assertTrue(np.allclose(ref_b, self.asim.corrector.b))
        self.assertTrue(np.allclose(ref_c, self.asim.corrector.c))

    def test_map(self):
        self.asim.build_all_maps(mapres=20, mapcrop=0.1)

        ref_extent = [-16.25, 16.25, -16.25, 16.25]
        the_extent = self.asim.map_extent
        self.assertTrue(np.allclose(ref_extent, the_extent))
        ref_maps = np.load("data/ref_maps_test.npy")
        self.assertTrue(np.allclose(ref_maps.shape, self.asim.maps.shape))
        self.assertTrue(np.allclose(ref_maps, self.asim.maps))

class test_offband_FT(unittest.TestCase):
    def setUp(self):
        self.asim = copy(asim)
        self.bsim = copy(bsim)
        altaz = self.asim.obs.observatory_location.altaz(target=self.asim.target,
                                                   time=self.asim.sequence[0])
        self.pistons = self.asim.obs.get_projected_geometric_pistons(altaz,)

    def tearDown(self):
        del self.asim

    def test_basic_simulation(self):
        integ = self.asim.make_exposure(asim.src.planet, asim.src.star, asim.diffuse, texp=1.)
        integ2 = self.bsim.make_exposure(bsim.src.planet, bsim.src.star, bsim.diffuse, texp=1.)
        
        
    def test_test_offband(self):
        self.assertIsInstance(self.pistons, np.ndarray)

