import numpy as np

import logging

logit = logging.getLogger(__name__)
def set_logging_level(level=logging.WARNING):
    logit.setLevel(level=level)
    logit.info("Setting the logging level to %s"%(level))

def mol_dens( temp, pres, rhum, co2, ph2o=None,
            wvdens=False):
    """
    **PURPOSE:**
    Returns the molair density of air and watervapor as a function temperature (in K), pressure (in mbar), relative humidity (%) and CO2 content (in ppm).
    
    **Parameters:**
    
    * temp : Temperature
    * pres : Pressure
    * rhum : Relative humidity
    * co2  : 
    
    
    **REFERENCE:**
        P.E. Ciddor, "The refractive index of air: new equations for the visible and near infrared", Appl. Opt. 35 (9), 1566-1573
    
    **MODIFICATION HISTORY:**
        Version 1.0, 09-OCT-2002, by Roland den Hartog, ESA / ESTEC /  Genie team, rdhartog@rssd.esa.int
    
    """
    
    
    ptot=pres*100.
    # Compute the partial water vapour pressure
    if ph2o is None:
        A, B, C, D = 1.2378847e-5, -1.9121316e-2, 33.93711047e0 , -6.3431645e+3
        psvp = np.exp(A * temp**2 + B * temp + C + D / temp)
        ph2o = (rhum/100.) * psvp
    
    A, B, C = 1.00062, 3.14e-8, 5.6e-7
    f = A + B * ptot + C * temp**2
    # !RL this is a bit weird?
    xw = np.min([(f * ph2o/ptot), 1.])
    # Compute the densities of air, standard dry air, water vapor and standard water vapor
    R = 8.314510  # J/mol/K

    a0 = 1.58123e-6 # K/Pa
    a1 = -2.9331e-8 # 1/Pa
    a2 = 1.1043e-10 # 1/K/Pa
    b0 = 5.707e-6   # K/Pa
    b1 = -2.051e-8  # 1/Pa
    c0 = 1.9898e-4  # K/Pa
    c1 = -2.376e-6  # 1/Pa
    d0 = 1.83e-11   # K^2/Pa^2
    d1 = -0.765e-8  # K^2/Pa^2

    # Density of dry air and water vapor
    t = temp - 273.15
    Z = 1e0 - (ptot/temp) *\
            (a0 + a1*t + a2*t**2 + (b0 + b1*t)*xw + (c0 + c1*t)*xw**2 ) +\
            (ptot/temp)**2 * (d0 + d1*xw**2)

    dax = ptot / (Z * R * temp) * (1e0 - xw)
    if wvdens:
        wvdens = ptot / (Z * R * temp) * xw
        return dax, wvdens
    else:
        return dax

def test_mol():
    
    #wvdens = np.array([1.27])
    
    temp = 288.15
    pres = 1013.25
    rhum = 0.
    co2 = 450.
    airdens, wvdens = mol_dens(temp, pres, rhum, co2, wvdens=True)
    print(airdens, wvdens)
    
    
    temp = 288.5
    pres = 743
    rhum = 33.
    cO2 = 450.
    airdens, wvdens = mol_dens(temp, pres, rhum, co2, wvdens=True)
    print(airdens, wvdens)
    
    temp = 293.15
    pres = 13.33
    rhum = 100.
    cO2 = 450.
    airdens, wvdens = mol_dens(temp, pres, rhum, co2, wvdens=True)
    print(airdens, wvdens)
    # !RL This last one returns 0, is that normal?