
import numpy as np
import sympy as sp
from sympy.functions.elementary.piecewise import Piecewise
from kernuller import mas2rad, rad2mas
from . import utilities

class blackbody(object):
    def __init__(self, modules="numexpr", mode=["nu","k","lambda"]):
        """
        Builds the spectral radiance as a function of T and \nu, k, and \lambda
        Call to get the function B = f(lambda , T)
        self.Bofnu is the function  B = f(nu , T)
        self.Bofk is the function  B = f(k , T)
        self.Boflambda is the function  B = f(lambda , T) (same as __call__)
        """
        
        self.mode = mode
        self.h, self.nu, self.c, self.k_b, self.T = sp.symbols("h, nu, c, k_b, T", real=True)
        self.B = 2*self.h*self.nu**3/self.c**2 * \
                    1/(sp.exp(self.h*self.nu/(self.k_b*self.T)) -1)
        self.thesubs = [(self.c, 299792458.),# speed of light
                       (self.k_b, 1.380649e-23),# Boltzmann constant (J.K^-1)
                       (self.h, 6.62607015e-34),# Planck constant (J.s)
                       ]
        if "nu" in self.mode:
            #fprint(self.B.subs(self.thesubs).simplify(), r"B(\nu, T) = ")
            self.Bofnu = utilities.ee(self.B.subs(self.thesubs))
            self.Bofnu.lambdify((self.nu, self.T), modules=modules)
            pass
        if "k" in self.mode:
            self.k, self.lamb = sp.symbols("k, lambda", real=True)
            self.ksubs = self.thesubs.copy()
            self.ksubs.insert(0, (self.lamb, 2*np.pi/self.k))
            self.ksubs.insert(0, (self.nu, self.c/self.lamb))
            #fprint(self.B.subs(self.ksubs), r"B(k, T) = ")
            self.Bofk = utilities.ee(self.B.subs(self.ksubs))
            self.Bofk.lambdify((self.k, self.T), modules=modules)
            
            
        if "lambda" in self.mode:
            self.k, self.lamb = sp.symbols("k, lambda", real=True)
            self.lambsubs = self.thesubs.copy()
            self.lambsubs.insert(0, (self.nu, self.c/self.lamb))
            #fprint(self.B.subs(self.lambsubs), r"B(\lambda, T) = ")
            self.Boflamb = utilities.ee(self.B.subs(self.lambsubs))
            self.Boflamb.lambdify((self.lamb, self.T), modules=modules)
        
        
    def __call__(self, nu, T):
        """
        Default call is for B(lambda)
        """
        result = self.Boflamb(nu, T)
        return result

class source(object):
    def __init__(self, xx, yy, ss):
        self.xx = xx
        self.yy = yy
        self.ss = ss
    def __add__(self,other):
        xx = np.concatenate((self.xx, other.xx))
        yy = np.concatenate((self.yy, other.yy))
        ss = np.concatenate((self.ss, other.ss), axis=1)
        return source(xx, yy, ss)
    def copy(self):
        return source(self.xx.copy(), self.yy.copy(), self.ss.copy())
    
    @classmethod
    def sky_bg(cls, injector, res,  T, lamb_range, crop=1.):
        """
        Builds the sky background for a given injector
        """
        # First build a grid of coordinates
        hskyextent = rad2mas(injector.focal_hrange/injector.focal_length)
        hskyextent = hskyextent*crop
        resol = res #injector.focal_res
        xx, yy = np.meshgrid(
                            np.linspace(-hskyextent, hskyextent, resol),
                            np.linspace(-hskyextent, hskyextent, resol))
        xx = xx.flatten()
        yy = yy.flatten()
        src = cls(xx, yy,
                  np.ones_like(lamb_range)[:,None]*np.ones_like(xx)[None,:])
        src.rr = np.sqrt(src.xx**2 + src.yy**2)
        thebb = blackbody()
        spectrum = thebb.Boflamb(lamb_range, T)
        src.ss = src.ss * spectrum[:,None]
        if injector is not None:
            
            src.mask = injector.LP01.numerical_evaluation(injector.focal_hrange*crop, resol, lamb_range)
            src.mask = src.mask.reshape((src.mask.shape[0], src.mask.shape[1]*src.mask.shape[2]))
            src.ss = src.ss * src.mask
        return src
        
        
        
class src_extended(object):
    def __init__(self, resol, extent, fiber_vigneting=False):
        """
        
        resol              : Number of elements across
        extent             : The extent of the source
        fiber_vigneting    : whether to include fiber vigneting in the luminosity distribution
                            Fiber vigneting should not be included when off-axis injection is simulated
        """
        self.xx, self.yy = np.meshgrid(
                            np.linspace(-extent/2, exten/2, resol),
                            np.linspace(-extent/2, exten/2, resol))
        self.rr = np.sqrt(self.xx**2 + self.yy**2)
        
        
    def uniform_disk(self, radius):
        self.ss = np.zeros_like(self.rr)
        self.ss[self.rr<radus] = 1.
        
        #self.f = Piecewise((1, self.r<=radius),
        #                   (0, self.r>radius))

        
        
class sky_bg(object):
    def __init__(self, temp, ):
        pass
    