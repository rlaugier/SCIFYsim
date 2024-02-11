import numpy as np
from . import mol_dens
from .mol_dens import mol_dens

import pathlib
import scipy.interpolate as interp
from astropy import constants, units

import logging

logit = logging.getLogger(__name__)
def set_logging_level(level=logging.WARNING):
    logit.setLevel(level=level)
    logit.info("Setting the logging level to %s"%(level))
    
    
class wet_atmo(object):
    def __init__(self, config=None,
                 temp=None, pres=None,
                 rhum=None, co2=None, ph2o=None, eso=False,
                 column=False, nws=None, name="True atmosphere"):
        """
        Creates a model for humid refractive index
        
        **Parameters:**
        
        * config : Give a config file that provides the parameters for
          the fields left at `None`.
        * temp   : air temperature [K]
        * pres   : air pressure [mbar]
        * rhum   : relative humidity of the air [%]
        * co2    : CO2 content in the air [ppm]
        * ph2o   : partial pressure of water vapour [in Pa = 0.01 mbar]
        
        """
        self.name = name
        self.eso = eso
        
        if temp is None:
            self.temp = config.getfloat("vlti", "T_vlti")
        else:
            self.temp = temp
        if pres is None:
            self.pres = config.getfloat("atmo", "pres")
        else:
            self.pres = pres
        if co2 is None:
            self.co2 = config.getfloat("atmo", "co2")
        else :
            self.co2 = co2
        if rhum is None:
            self.rhum = config.getfloat("atmo", "rhum")
        else:
            self.rhum = rhum
            
        self.Nair = None
        

    
    def get_Nair(self, lambs, add=0):
        """
        Returns the refractive index (or reduced refractive index)
        for humid air at the given wavelengths.

        **Parameters:**

        * lambs : An array of wavelengths [m]
        * add   : Gets added to the reduced refracive index:
          default: 0. Use 1 to obtain the actual refractive index.

        **Returns:**  add + n-1 
        """

        # Computing fraction of components
        P_tot = self.pres*units.mbar.to(units.Pa)
        F_co2 = 1e-6 * self.co2
        F_h2o = self.rhum/100 * get_pvsat_buck(self.temp)/P_tot
        F_da = 1-(F_co2 + F_h2o)
        #print("H2O", F_h2o)
        #print("CO2", F_co2)
        #print("DA", F_da)
        n_ref = ((r_index_DA(lambs))*F_da + (r_index_co2(lambs))*F_co2 + (r_index_h2o(lambs))*F_h2o)
        n_compound_air = constants.c.value*P_tot/(constants.R.value * self.temp) * n_ref
        self.Nair = n_compound_air + add
        return self.Nair

    def get_Nair_wn(self, sig, add=1):
        """
        Add=1 by default
        
        Returns the refractive index (or reduced refractive index)
        for humid air at the given wavelengths.

        **Parameters:**

        * sig : An array of wavenumber [m^-1]
        * add   : Gets added to the reduced refracive index:
          default: 1. Use 1 to obtain the actual refractive index.

        **Returns:**  add + n-1 
        """
        lambs = 1./sig
        return self.get_Nair(lambs, add=add)
                
    def get_Nair_old(self, lambs, add=0):
        """
        Returns the refractive index (or reduced refractive index)
        for humid air at the given wavelengths.
        
        **Parameters:**
        
        * lambs : An array of wavelengths [m]
        * add   : Gets added to the reduced refracive index:
          default: 0. Use 1 to obtain the actual refractive index.
        """
        logit.error("Using get_Nair_old() which is deprecated")
        logit.error("This uses mol_dens...")
        logit.error("This is deprecated")
        self.Nair = add + n_air(lambs, temp=self.temp,
                         pres=self.pres,
                         rhum=self.rhum,
                         co2=self.co2,
                         eso=self.eso)
        return self.Nair
    def report_setup(self):
        print("================================")
        print(f"{self.name}")
        print(f"Temperature : {self.temp:.3e} K")
        print(f"Pressure : {self.pres:.3e} mbar")
        print(f"CO2 content : {self.co2:.1f} ppm")
        print(f"Relative humidity : {self.rhum:.1f} %")
        print("================================")
        
    
    def report_setup(self):
        print("================================")
        print(f"{self.name}")
        print(f"Temperature : {self.temp:.3e} K")
        print(f"Pressure : {self.pres:.3e} mbar")
        print(f"CO2 content : {self.co2:.1f} ppm")
        print(f"Relative humidity : {self.rhum:.1f} %")
        print("================================")
######################################################################
# Update chunk
######################################################################
def n_hat2nm1(nhat, P_i, T, add=0):
    """
    Works for pure gazes:
    
    **Arguments:**
    * P_i  : Partial pressure [Pa]
    * T    : Temperature [K]
    
    **Return:** add + n-1
    """
    return add + nhat * constants.c.value * P_i /(constants.R.value * T)
    

def compile_table(file, ):
    """
    Compiles a Mathar table into an interpolation function
    
    **Arguments:**
    * file  : The url for a .dat file containing:
        - in column 1 the frequency in THz
        - in column 2 the reduced refracive index [fs/(mol/m^2)].
    
    **Returns:** A function that computes the reduced the reduced
    refractive index. Note that it switches from fs to s: [s/(mol/m^2)].
    """
    atab = np.loadtxt(file)
    freqM = atab[:, 0]
    ref_indexM = atab[:, 1]
    
    spline = interp.splrep(freqM, ref_indexM, k=1)
    def get_n_red(lam2,):
        """
        Interpolation of one of the Mathar tables. 
        Note that we switch to seconds here.
        
        **Arguments:**
        
        * lam2  : The wavelength [m]
        
        **Returns:** the reduced refractive index [s.mol^-1.m^2].
        """
        freq2 = m2thz(lam2) # THz
        red_index2 = interp.splev(freq2, spline, ext=3)
        return red_index2*1e-15
    return get_n_red

# Loading tables compiled by Richard Mathar & Jeff Meisner.
r = pathlib.Path(__file__).parent.absolute()
r_index_co2 = compile_table(r/"data/n_mathar_co2.dat")        # Pure CO2
r_index_air = compile_table(r/"data/n_mathar_air_370co2.dat") # Dry air with 370ppm of CO2 
r_index_h2o = compile_table(r/"data/n_mathar_h2o.dat")        # Purte H2O

def r_index_DA(lamb):
    """
    Subtracting the contribution fo CO2 from the reference air that 
    is given by Mathar with 370ppm of CO2.
    
    **Arguments:**
    * Wavelength [m]
    """
    F_co2 = 370.*1e-6
    return (r_index_air(lamb) - r_index_co2(lamb) * F_co2) / (1-F_co2)


def get_pvsat_buck(T):
    """
    Saturating vapor pressure of H2O
    
    **Parmameters:**
    
    * T  : Temperature [K]
    
    **Returns:** P_vsat [Pa]
    """
    T_C = T - 273.15
    P_kPA = 0.61121*np.exp( (18.678 - T_C/234.5) * (T_C/(257.14 + T_C)))
    return P_kPA * 1000
def get_Ph2o(Rhum, T):
    """
    **Parameters:**
    
    * Rhum  : Relative humidity [%]
    * T     : Temperature [K]
    
    **Returns:** The partial pressure of water vapor [Pa]
    """
    Ph2o = Rhum/(100)*get_pvsat_buck(T)
    return Ph2o

def get_Pco2(co2_ppm, T, ptot):
    """
    **Parameters:**
    
    * co2_ppm : Relative content of CO2 [ppm] (ppmv)
    * T       : Temperature [K]
    * ptot       : The total pressure of the mixture [Pa]
    
    **Returns:** The partial pressure of CO2 [Pa]
    """
    Fco2 = 1e-6 * co2_ppm
    Pco2 = Fco2*ptot
    return Pco2

def get_n_air(lamb, pres, g_co2, rhum, T, nref=False, add=0):
    """
    * pres  : [mbar]
    * g_co2 : [ppmv]
    * rhum  : [%]
    * T     : [K]
    """
    P_tot = pres*units.mbar.to(units.Pa)
    F_co2 = 1e-6 * g_co2
    F_h2o = rhum/100 * get_pvsat_buck(T)/P_tot
    F_da = 1-(F_co2 + F_h2o)
    print("H2O", F_h2o)
    print("CO2", F_co2)
    print("DA", F_da)
    c = constants.c.value
    n_ref = ((r_index_DA(lamb))*F_da + (r_index_co2(lamb))*F_co2 + (r_index_h2o(lamb))*F_h2o)
    if nref:
        return n_ref
    else:
        n_compound_air = add + constants.c.value*P_tot/(constants.R.value * T) * n_ref
        return n_compound_air





#######################################################################
# End of new block
#######################################################################
        
    
#class simulated_air(object):
#    def __init__(temp=273.15+15., pres=1000.,
#         rhum=0., co2=450., ph2o=None, eso=False,
#         column=False, nws=None, name="Simple model for atmosphere"):
#        """
#        An object to represent refractive and dispersive medium.
#        """
#        self.temp = temp
#        self.pres = pres
#        self.rhum = rhum
#        self.co2 = co2
#        self.ph2o = ph2o
#        self.eso = eso
#        
#        self.__doc__ = name
#        
#    def get_n(lambda_):
#        """
#        
#        """
#        then = n_air(lambda_, temp=self.temp, pres=self.pres,
#                    rhum=self.rhum, co2=self.co2, ph2o=self.ph2o,
#                    eso=self.eso, column=self.column) 
        
def get_n_CO2(wavelength, add=0):
    """
    **Parameters :**
    
    * wavelength: [m] 
    * add : Value to add to n-1
      (default: 0 returns n-1)
    
    Returns the refractive index of CO2 from
    dispersion formula found at
    https://refractiveindex.info/?shelf=main&book=CO2&page=Bideau-Mehu
    
    n_absolute: true
    wavelength_vacuum: true
    temperature: 0 °C
    pressure: 101325 Pa
    
    Ref:
    A. Bideau-Mehu, Y. Guern, R. Abjean and A. Johannin-Gilles. Interferometric
    determination of the refractive index of carbon dioxide in the ultraviolet
    region, Opt. Commun. 9, 432-434 (1973)
    """
    logit.error("Using get_n_CO2() which is deprecated")
    logit.error("This is deprecated. Use n_hat2nm1(r_index_co2(lambs), P, T) instead")
    lambs = wavelength*1e6
    Bs_CO2 = np.array([6.991e-2, 1.4472e-3, 6.42941e-5, 5.21306e-5, 1.46847e-6])
    Cs_CO2 = np.array([166.175, 79.609, 56.3064, 46.0196, 0.055178])#0.0584738
    terms = np.array([aB/(aC-lambs**-2) for aB, aC in zip(Bs_CO2, Cs_CO2)])
    n_CO2 = add + np.sum(terms, axis=0)
    return n_CO2
    

def n_air(lambda_ , temp=273.15+15., pres=1000.,
         rhum=0., co2=450., ph2o=None, eso=False,
         column=False, nws=None):
    """
    This function is a translation of a function of GENIEsim.
    
    **DESCRIPTION**
        Returns the phase refractive index (n-1) of air a function of (IR) wavelength (in m),
        and optionally, temperature (in K), pressure (in bar), relative humidity (%) and CO2 content (in ppm).
        Note that the used approximation for air applies in the range from 300 to 1690 nm,
        hence their use at much longer wavelengths should be done with caution.
        For wavelengths longer than 1.7 micron, use the Hill & Lawrence approximation for water vapor,
        which has been verified with experimental data up to 15 micron.
    
    **Argument**
    
    * lambda_ : wavelength vector in meters
    
    **Keyword arguments**
    
    * temp   : air temperature [K]
    * pres   : air pressure [mbar]
    * rhum   : relative humidity of the air [%]
    * co2    : CO2 content in the air [ppm]
    * ph2o   : partial pressure of water vapour [in Pa = 0.01 mbar]
      if set, overrules RHUM -- if not defined, partial pressure will be given on output
    * eso    : set this keyword to use E. Marchetti's moist air refractive index instead of Ciddor + Hill & Lawrance
    * column : set this keyword to convert to input into fs/(mopl/m²), instead of the standard unitless n-1 value
    * nws    : on output, returns the refractive index of pure water vapour (unless ESO keyword is set)
    
    **CALLS**
    
    * MOL_DENS
    * N_H2O
    
    **REFERENCE**
    
    * J.E. Decker et al. "Updates to the NRC gauge block interferometer", NRC document 42753, 8 August 2000
    * P.E. Ciddor, "The refractive index of air: new equations for the visible and near infrared", Appl. Opt. 35 (9), 1566-1573
    * J. Meisner & R. Le Poole, "Dispersion affecting the VLTI and 10 micron interferometry using MIDI", Proc. SPIE 4838
    * `<http://www.eso.org/gen-fac/pubs/astclim/lasilla/diffrefr.html>`_
    
    **MODIFICATION HISTORY**
    
    * Version 1.0, 17-SEP-2002, by Roland den Hartog, ESA / ESTEC /  Genie team, rdhartog@rssd.esa.int
    * Version 1.1, 09-OCT-2002, RdH: conversion to column densities (Meisner's n^hat) implemented
    * Version 1.2, 01-NOV-2002, RdH: water vapor index based on approximation also valid in the IR
    * Version 1.3, 03-JUL-2003, OA:  PostScript output of test harness modified
    * Version 1.4, 15-DEC-2009, OA:  Removed discontinuity at 1.7µm by using tabulated water vapour refraction index instead of models + improved header
    * SCIFYsim , Oct. 2020, : Translated to python by R. Laugier for SCIFYsim

    """
    
    cvac = 299792458.
    dax, dw = mol_dens(temp, pres, rhum, co2, ph2o=ph2o, wvdens=True)
    # Compute the refractive index of moist air...
    if eso:
        #
        PS = -10474.0 + 116.43*temp - 0.43284*temp**2 + 0.00053840*temp**3
        P2 = rhum/100.0 * PS
        P1 = pres - P2
        D1 = P1/temp * (1.0 + P1*(57.90*1.0e-8-(9.3250 * 1.0e-4/temp) \
                                  + (0.25844/temp**2)))
        D2 = P2/temp * (1.0 + P2*(1.0 + 3.7e-4*P2) * (-2.37321e-3 + (2.23366/temp) - (710.792/temp**2) + (7.75141e4/temp**3)))
        S = 1e-6 / lambda_
        nair = 1.0e-8*((2371.34 + 683939.7/(130-S**2) + 4547.3/(38.9 - S**2)) * D1 \
                       + (6487.31 + 58.058*S**2 - 0.71150*S**4 + 0.08851*S**6)*D2)
    else:
        #... or Ciddor
        # Refractive index of pure air...
        k0, k1, k2, k3 = 238.0185, 5792105., 57.362 , 167917.       # um^-2
        s2 = (1. /(lambda_ * 1e+6))**2                                  # um^-2
        #naxs = 1e-8*(k1/(k0 - s2) + k3/(k2 - s2))*(1. + 0.534e-6*(co2 - 450.)) # in reality this is n-1
        # Using the new n_CO2 definition instead
        naxs = 1e-8*(k1/(k0 - s2) + k3/(k2 - s2)) # in reality this is n-1
        naxs += get_n_CO2(lambda_)*co2/1e6
        

        # and pure water vapor, using the Hill & Lawrence approximation in IR, Ciddor in VIS
        #n=N_ELEMENTS(lambda) & nws=FLTARR(n)
        #w=WHERE(lambda GT 1.7D-6, c) & IF c GT 0 THEN nws[w]=N_H2O(lambda[w], APPROX=1, CO2=co2, PRES=pres, RHUM=rhum, TEMP=temp) # Hill & Lawrence
        #w=WHERE(lambda LE 1.7D-6, c) & IF c GT 0 THEN nws[w]=N_H2O(lambda[w], APPROX=2, CO2=co2, PRES=pres, RHUM=rhum, TEMP=temp) # Ciddor
        nws = n_h2o(lambda_, temp=temp, pres=pres, rhum=rhum, co2=co2) # returns the water vapour refractive index (n-1) as tabulated by Mathar 
        #nws = N_H2O(lambda, APPROX=2, CO2=co2, PRES=pres, RHUM=rhum, TEMP=temp) # Ciddor

        # Compute the densities of air, standard dry air, water vapor and standard water vapor
        daxs = mol_dens(288.15, 1013.25, 0., 450.)# !RL Added the CO2 value. Bug?
        #daxs *= 1.e6/450.
        dummy, dws = mol_dens(293.15, 13.33, 100., 450., wvdens=True)

        # Compute the phase index
        nair = (dax/daxs)*naxs + (dw/dws)*nws
    
    # Convert units to fs/(mol/m2)
    if column:
        rhoc = cvac*dax*1e-15
        nair = nair/rhoc
    
    return nair

# Test harness
def test_air():
    import matplotlib.pyplot as plt
    
    # Comparison with Meisner's figure
    freq = np.linspace(20., 166., 1460) # THz (very similar to "np.arange(1460)/10.+20.")
    c = 2.997925e+8
    lambda_ = c / freq / 1e+12

    temp = 273.15 + 15.
    co2 = 350.
    model = n_air(lambda_, temp=temp, co2=co2, column=True)
    
    nwv, freq = n_h2o(lambda_, freq=True, temp=temp, rhum=0., column=True)
    nwda, freq = n_h2o(lambda_, freq=True, temp=temp, rhum=0., column=True, wda=True)
    
    plt.figure()
    plt.plot(freq, model)
    plt.xlabel("Frequency [Thz]")
    plt.ylabel(r"$\frac{n-1}{c.\rho}$")
    plt.title("Refractive index of dry air")
    plt.show()
    plt.figure()
    plt.plot(lambda_*1e6, model, label="n_air")
    plt.plot(lambda_*1e6, nwv, label="n_h2o")
    plt.title("Refractive index of water vapor vs. dry air")
    plt.ylabel(r"$\frac{n-1}{c.\rho}$ [$fs/mol/m^2$]")
    plt.xlabel(r"wavelength [$\mu m$]")
    plt.show()
    
    nda = n_air(lambda_, temp=temp, co2=co2, rhum=0., column=True)
    print("Min and max: of nda")
    print(np.min(nda), np.max(nda))
    
    plt.figure()
    plt.plot(freq, nwv, label="Humid air")
    plt.plot(freq, nda, label="Dry air")
    plt.title("Refractive index of water vapor vs. dry air")
    plt.ylabel(r"$\frac{n-1}{c.\rho}$")
    plt.xlabel("Frequency [Thz]")
    plt.legend()
    plt.show()
    
    
    plt.figure()
    plt.plot(freq, nwda)
    plt.title("Refractive index of water vapor vs. dry air")
    plt.ylabel(r"$\frac{n-1}{c.\rho} $")
    plt.xlabel("Frequency [Thz]")
    plt.show()
    
    # The following figure shows the wavelength-dependence of OPD
    opd_air = 1e-15 * 1e6 * c * nda * 1.5  # differential column density = 1.5 mol/m², opd_air in µm
    opd_wda = -1e-15 * 1e6 * c * nwda * 1.5  # differential column density = -1.5 mol/m², opd_wda in µm
    opd_tot = opd_air + opd_wda
    plt.figure()
    plt.plot(lambda_*1e6, opd_air + opd_wda, label="n_air")
    plt.title("Differential OPD")
    plt.ylabel(r"OPD [$\mu m$]")
    plt.xlabel(r"wavelength [$\mu m$]")
    plt.show()
    #return model, freq
    
    # Air: comparison with tables in Ciddor paper:
    lambda_ = 633e-9
    print('Table 1 for dry air from P. Ciddor, Appl. Opt. 35, 1566 (1996)')
    print(' Temperature [C]    Pressure [Pa]   (n-1)*1E-8')
    temp, pres = 273.15+20., 80e+3/100.
    print(temp-273.15, pres*100., 1e8*(n_air(lambda_, temp=temp, pres=pres)))
    temp, pres = 273.15+20., 100e+3/100.
    print(temp-273.15, pres*100., 1e8*(n_air(lambda_, temp=temp, pres=pres)))
    temp, pres = 273.15+20., 120e+3/100.
    print(temp-273.15, pres*100., 1e8*(n_air(lambda_, temp=temp, pres=pres)))
    temp, pres = 273.15+10., 100e+3/100.
    print(temp-273.15, pres*100., 1e8*(n_air(lambda_, temp=temp, pres=pres)))
    temp, pres = 273.15+30., 100e+3/100.
    print(temp-273.15, pres*100., 1e8*(n_air(lambda_, temp=temp, pres=pres)))
    print("")

    print('Table 2 for moist air from P. Ciddor, Appl. Opt. 35, 1566 (1996)')
    print(' Temperature [C]   Pressure [Pa]   H2O pres.   CO2 [ppm]     (n-1)*1E-8')
    temp, pres, ph2o, co2 = 273.15+19.526,  102094.8/100.,  1065.,  510.
    print(temp-273.15, pres*100., ph2o, co2, 1e8*(n_air(lambda_, temp=temp, pres=pres, ph2o=ph2o, co2=co2)))
    temp, pres, ph2o, co2 = 273.15+19.517,  102096.8/100.,  1065.,  510.
    print(temp-273.15, pres*100., ph2o, co2, 1e8*(n_air(lambda_, temp=temp, pres=pres, ph2o=ph2o, co2=co2)))
    temp, pres, ph2o, co2 = 273.15+19.173,  102993.0/100.,  641.,  450.
    print(temp-273.15, pres*100., ph2o, co2, 1e8*(n_air(lambda_, temp=temp, pres=pres, ph2o=ph2o, co2=co2)))
    temp, pres, ph2o, co2 = 273.15+19.173,  103006.0/100.,  642.,  440.
    print(temp-273.15, pres*100., ph2o, co2, 1e8*(n_air(lambda_, temp=temp, pres=pres, ph2o=ph2o, co2=co2)))
    temp, pres, ph2o, co2 = 273.15+19.188,  102918.8/100.,  706.,  450.
    print(temp-273.15, pres*100., ph2o, co2, 1e8*(n_air(lambda_, temp=temp, pres=pres, ph2o=ph2o, co2=co2)))
    temp, pres, ph2o, co2 = 273.15+19.189,  102927.8/100.,  708.,  440.
    print(temp-273.15, pres*100., ph2o, co2, 1e8*(n_air(lambda_, temp=temp, pres=pres, ph2o=ph2o, co2=co2)))
    temp, pres, ph2o, co2 = 273.15+19.532,  103603.2/100.,  986.,  600.
    print(temp-273.15, pres*100., ph2o, co2, 1e8*(n_air(lambda_, temp=temp, pres=pres, ph2o=ph2o, co2=co2)))
    temp, pres, ph2o, co2 = 273.15+19.534,  103596.2/100.,  962.,  600.
    print(temp-273.15, pres*100., ph2o, co2, 1e8*(n_air(lambda_, temp=temp, pres=pres, ph2o=ph2o, co2=co2)))
    temp, pres, ph2o, co2 = 273.15+19.534,  103599.2/100.,  951.,  610.
    print(temp-273.15, pres*100., ph2o, co2, 1e8*(n_air(lambda_, temp=temp, pres=pres, ph2o=ph2o, co2=co2)))
    print("")

    print('Table 3 for moist air from P. Ciddor, Appl. Opt. 35, 1566 (1996)')
    print(' Temperature [C]  Pressure [Pa]  Humidity [%]    (n-1)*1E-8')
    temp, pres, rhum = 273.15+20.,  80e+3/100.,  75.
    print(temp-273.15, pres*100., rhum, 1e8*(n_air(lambda_, temp=temp, pres=pres, rhum=rhum)))
    temp, pres, rhum = 273.15+20.,  120e+3/100.,  75.
    print(temp-273.15, pres*100., rhum, 1e8*(n_air(lambda_, temp=temp, pres=pres, rhum=rhum)))
    temp, pres, rhum = 273.15+40.,  80e+3/100.,  75.
    print(temp-273.15, pres*100., rhum, 1e8*(n_air(lambda_, temp=temp, pres=pres, rhum=rhum)))
    temp, pres, rhum = 273.15+40.,  120e+3/100.,  75.
    print(temp-273.15, pres*100., rhum, 1e8*(n_air(lambda_, temp=temp, pres=pres, rhum=rhum)))
    temp, pres, rhum = 273.15+50.,  80e+3/100.,  100.
    print(temp-273.15, pres*100., rhum, 1e8*(n_air(lambda_, temp=temp, pres=pres, rhum=rhum)))
    temp, pres, rhum = 273.15+50.,  120e+3/100.,  100.
    print(temp-273.15, pres*100., rhum, 1e8*(n_air(lambda_, temp=temp, pres=pres, rhum=rhum)))
    print()


####################################################################################

    



def n_h2o(lambda_, approx=False, column=False,
            co2=450., pres=1013.25, rad=False, rhum=0.,
            table=False, temp=296.15, wda=False, freq=False):
    """
    **PURPOSE:**
    
    Returns the refractive index (n-1) of water vapor a function of (IR) wavelength (in m),
    in the range from 3.3 to 10.6 micron,
    and optionally, temperature (in K), and vapor density in (kg/m3)
        
    **Argument**
    
    * lambda_ : wavelength vector in meters
    
    **Keyword arguments:**
    
    * temp:   temperature in K
    * pres:   pressure in mbar
    * rhum:   relative humidity in %
    * co2:    CO2 fraction in ppm
    * approx: 
    
            - if not set, the table from Mathar will be interpolated at the input wavelengths
            - if set to 1, use approximate formula by Hill & Lawrence
            - if set to 2, use approximate formula by Ciddor
            
    * table:  set this keyword to use the full Mathar table -- warning, this modifies the lambda array on output (ONLY FOR TEST PURPOSE)
    * column: convert units to fs / (mol/m^2), such that t_delay = n_H20 * column density
    * rad:    convert units to radians
    * wda:    the refraction index is given for water vapour displacing air instead of bare water vapour, in units of fs/(mol/m^2)
    
    **RESTRICTIONS:**
       Does require a file 'n_mathar.dat' to be present in same directory
       The Mathar data table does not include wavelengths smaller than 1.819 µm.
       At wavelengths smaller than 1.819 µm, the Hill & Lawrence approximation is used instead.
       This produces a discontinuity of N_H2O at 1.819 µm.
    
    **CALLS:**
    
        * n_air
    
    **REFERENCE:**
    
    * R.J. Hill, R.S. Lawrence, "Refractive index of water vapor in infrared windows", Infrared Phys. 26, 371 - 376 (1986)
    * P.E. Ciddor, "The refractive index of air: new equations for the visible and near infrared", Appl. Opt. 35 (9), 1566-1573
    * F. Hase, R.J. Mathar, "Water vapor dispersion in the atmospheric window at 10 um", Preprint, 06-FEB-2002
    
    **MODIFICATION HISTORY:**
    
    * Version 1.0, 13-SEP-2002, by Roland den Hartog, ESA / ESTEC /  Genie team, rdhartog@rssd.esa.int
    * Version 2.0, 09-OCT-2002, RdH: included tabulated data by R.J. Mathar (obtained via J. Meisner)
    * Version 2.1, 29-OCT-2002, RdH: conversion to WDA implemented
    * Version 2.2, 01-NOV-2002, RdH: Ciddor's approximation implemented
    * Version 2.3, 15-DEC-2009, OA:  Improved header
    
    **TESTED**
    
    * 13-SEP-2002, RdH: comparison with measurements by Hase and Mathar
    * 09-OCT-2002, RdH: direct comparison between Hill & Lawrence's approximation and Mathar's data
    * 15-NOV-2004, RdH: implemented option to convert output directly into radians
    """
    cvac = 299792458.
    # !RL we need to do something for global variables
    global nh2o, freqM, nh2oM
    if rad :
        column = True
    
    airdens, wvdens = mol_dens(temp, pres, rhum, co2, wvdens=True) #mol/m3
    rhoc = wvdens * cvac * 1e-15 # converts n-1 into units of fs / mol / m2
    
    # The Mathar data table does not include wavelengths smaller than 1.819 µm.
    # At wavelengths smaller than 1.819 µm, use the Hill & Lawrence approximation instead
    separate = False
    if not approx:
        w1 = lambda_ < 1.819e-6 
        c1 = np.count_nonzero(w1)
        w2 = np.logical_not(w1)
        c2 = np.count_nonzero(w2)
        
        if (c1>0) and (c2>0):
            separate = True
            lam1 = lambda_[w1]
            # the Hill & Lawrence approximation will be used on this part of the spectrum
            lam2 = lambda_[w2]
        elif (c1>0) and (c2==0):
            approx = True
    if approx or separate:
        if not separate:
            lam1 = lambda_
        freq1 = m2thz(lam1) # Frequencies in THz
        if approx==2: # !RL need to check that call!
            s1 = 1/(lam1*1e6)
            s2 = s1**2#(lam1*1e6)**2 # um^-2
            cf, w0 = 1.022, 295.235
            w1 = 2.6422    # um^-2
            w2 = -0.032380 # um^-4
            w3 = 0.004028  # um^-6
            s4 = s1**4 # !RL How about that?
            s6 = s1**6
            nh2o1 = 1e-8 * cf * (w0 + w1*s2 + w2*s4 + w3*s6)  # in reality this is n-1
            #print("nh2o1 (n-1)", nh2o1)
        else :
            tt=temp/273.16
            x=1e-5/lam1
            Q=18.015*wvdens  # go from mol / m3 to g / m3
            nh2o1 = 1e-6 * Q * \
                    ( (0.957 - 0.928*(tt**0.4)*(x - 1.)) \
                    /(1.03*(tt**0.17) - 19.8*(x**2) + 8.1*(x**4) - 1.7*(x**8)) \
                        + 3747./(12449. - (x**2)) )
            #HELP, tt, x, q, nh2o1
            #PRINT, MIN(tt), MAX(tt), MIN(x), MAX(x), MIN(q), MAX(q), MIN(nh2o1), MAX(nh2o1)
        if column or wda :
            nh2o1 = nh2o1/rhoc
    
    if not approx:
        if not separate:
            lam2 = lambda_
        #if nh2oM.shape[0] < 2: # !RL Should probably cleanup this condition
        r = pathlib.Path(__file__).parent.absolute()
        n_mathar = np.loadtxt(r/"data/n_mathar.dat")
        freqM = n_mathar[:, 0]
        nh2oM = n_mathar[:, 1]
        # Interpolate
        if lam2.shape[0]<=0 or table:
            #lam2 = cvac/freqM/1e+12 
            lam2 = thz2m(freqM)
            freq2 = freqM
            nh2o2 = nh2oM
        else:
            freq2 = m2thz(lam2) # THz
            spline = interp.splrep(freqM, nh2oM)
            nh2o2 = interp.splev(freq2, spline, ext=3)
        # Conversion 
        nh2o2 = (9.05e-7 + nh2o2) * (6.022 * 3335000.) # fs /mol /m2
        if not (column or wda) :
            nh2o2 = nh2o2 * rhoc # convert to dimensionless n-1
    # Reform the arrays if they were separated
    if approx:
        freqs = freq1
        nh2o = nh2o1
    else :
        if separate :
            lambda_ = np.concatenate((lam1, lam2), axis=0)
            freqs = np.concatenate((freq1, freq2), axis=0)
            nh2o = np.concatenate((nh2o1, nh2o2), axis=0) 
        else :
            lambda_ = lam2
            freqs = freq2
            nh2o = nh2o2
    
    if wda :
        nh2o = nh2o - n_air(lambda_, column=True, pres=pres,
                            rhum=0., temp=temp)
    if rad :
        nh2o = nh2o * 1e15 * cvac * 2. * np.pi / lambda_
    if freq:
        return nh2o, freqs
    else:
        return nh2o
def m2thz(lambda_):
    return 299792458.* 1e-12/lambda_ 
def thz2m(f):
    return 299792458./ (f * 1e12)

# Test harness
def test_h2o():
    import matplotlib.pyplot as plt
    lambda_ = np.array([3.368, 3.392, 3.508, 10.246, 10.571,
                        10.591, 10.611, 10.632, 10.653])*1e-6
    print("Frequencies: ")
    print(299792458./lambda_ * 1e-12)
    temp = 273.15 + 20.
    pres = 743.
    rhum = 99.
    approx = n_h2o(lambda_, temp=temp, pres=pres, rhum=rhum, approx=True)
    mathar = n_h2o(lambda_, temp=temp, pres=pres, rhum=rhum)
    appcol = n_h2o(lambda_, temp=temp, pres=pres, rhum=rhum, approx=True, column=True)
    matcol = n_h2o(lambda_, temp=temp, pres=pres, rhum=rhum, column=True)
    print('                      (n-1) * 1E+6                    (n-1)/rho/c')
    print('  lambda      Hill & Lawrence     Mathar      Hill & Lawrence     Mathar')
    for i in range(8):
        print(lambda_[i]*1e6, approx[i]*1e6, mathar[i]*1e6, appcol[i], matcol[i])
    print("")
    temp=273.15+23.
    rhum=42.3
    
    mathar, freq = n_h2o(lambda_, temp=temp, pres=pres, rhum=rhum,
                         column=True, table=True, freq=True)
    # !RL Non-idempotent code ahead!
    #lambda_ = lambda_*1e6
    #n = mathar.shape[0]
    dr = 0.5*np.abs(mathar[0] - mathar[-1])
    ar = 0.5*(mathar[0] - mathar[-1])
    print( mathar[0], mathar[-1])
    plt.figure()
    plt.plot(freq, mathar, label="Mathar")
    plt.title("Refractive index of H2O vapor")
    plt.xlabel("Frequency [THz]")
    plt.ylabel(r"$n-1$")
    plt.show()
    
    # Hill and Lawrence approximation
    lambda_ = np.linspace(1.e-6, 10.6e-6, 1000)
    approx_hl, freq2 = n_h2o(lambda_, freq=True, temp=temp, rhum=rhum,
                   approx=True, column=True)
    # Ciddor approximation 
    approx_c, freq2 = n_h2o(lambda_, freq=True, temp=temp, rhum=rhum,
                   approx=2, column=True)
    
    plt.figure()
    plt.plot(freq, mathar, label="Mathar")
    plt.plot(freq2, approx_hl, label="Hill and Lawrence approx.")
    plt.plot(freq2, approx_c, label="Ciddor approx.")
    plt.xlabel("Frequency [THz]")
    plt.title("Refractive index of H2O vapor")
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(thz2m(freq)*1e6, mathar, label="Mathar")
    plt.plot(thz2m(freq2)*1e6, approx_hl, label="Hill and Lawrence approx.")
    plt.plot(thz2m(freq2)*1e6, approx_c, label="Ciddor approx.")
    plt.xlabel(r"Wavelength [$\mu m$]")
    plt.title("Refractive index of H2O vapor")
    plt.legend()
    plt.show()
    
    # WDA
    mathar, freq2 = n_h2o(lambda_*1e6, freq=True, temp=temp, rhum=rhum, wda=True)