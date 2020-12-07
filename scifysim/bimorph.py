import numpy as np
import pathlib
import scipy.interpolate as interp
import matplotlib.pyplot as plt

import logging

logit = logging.getLogger(__name__)
def set_logging_level(level=logging.WARNING):
    logit.setLevel(level=level)
    logit.info("Setting the logging level to %s"%(level))


r = pathlib.Path(__file__).parent.absolute()
table = np.loadtxt(r/"data/bimorph.dat")
freqBM = table[:,0]
psdBM = table[:,1]

f = interp.interp1d(freqBM, psdBM, kind="linear",
                    fill_value="extrapolate")

def bimorph(freq):
    """
    
     DESCRIPTION
       Returns the power in the residual piston, after the VLTI MACAO deformable bimorph mirror,
       as a function of frequency, in units of nm²/Hz, by interpolation of measured data,
       obtained by ESO over the capacitive sensor area of MACAO, for a 420 Hz control loop frequency.

     INPUT
       freq:  frequency array in Hz

     OUTPUT:
      psd:   power of bimorph piston in m²/Hz

     REFERENCE:
       Liviu Ivanescu, priv. comm. 28-JUN-2004, via Florence Puech.

     MODIFICATION HISTORY:
       Version 1.0, 30-JUN-2004, Roland den Hartog, ESA/ESTEC Genie team, rdhartog@rssd.esa.int
    """
    
    # !RL COMMO not implemented
    # COMMON BIM, freqBM, psdBM, nBM
    
    # Open the sky transmission file, if it has not been done so before
    
    psd = f(freq)
    return psd
    
def test_bim():
    frs = np.linspace(0., 2000., 2000)
    h = bimorph(frs)
    #temp = BODE(f, H, TITLE='MACAO bimorph mirror PSD')
    
    plt.figure()
    plt.loglog(freqBM, psdBM, label="Table")
    plt.loglog(frs, h, label="Interpolation")
    plt.title("Interpolation quality")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [??]")
    plt.legend()
    plt.show()