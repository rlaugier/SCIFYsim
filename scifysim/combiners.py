import kernuller
import sympy as sp
import numpy as np
from kernuller import fprint


import logging

logit = logging.getLogger(__name__)



def bracewell_ph():
    """
    in: 2
    out: 4 (ph0, bright, dark, ph1)
    symbols:
        sigma  : the photometric ratio (in intensity)
    Build a bracewell combiner with photometric outputs.
    
    
    Returns: the sympy.Matrix of the combiner M
    Free symbols can be retrieved in list(M.free_symbols)
    """
    sigma = sp.symbols("sigma", real=True)
    psplitter1 = sp.Matrix([[sp.sqrt(sigma)],
                            [sp.sqrt(1-sigma)]])
    psplitter2 = kernuller.crossover@psplitter1
    #fprint(psplitter1, "\mathbf{Y}_1 = ")
    #fprint(psplitter2, "\mathbf{Y}_2 = ")
    #fprint(kernuller.xcoupler, "\mathbf{X} = ")

    A = sp.diag(psplitter1, psplitter2)
    B = sp.diag(1,kernuller.xcoupler, 1)

    combiner = B@A
    #fprint(combiner1, "\mathbf{M}_1 = ")
    #print("Here with a 0.05 splitting ratio")
    #Mn = kernuller.sp2np(combiner1.subs([(sigma, 0.05)])).astype(np.complex128)
    #fig, axs = kernuller.cmp(Mn, nx=1, out_label=np.arange(4), mainlinewidth=0.05)
    
    return combiner



def angel_woolf_ph(ph_shifters=None):
    """
    optional :
    ph_shifters : a list of phase shifters in between the 2 stages
                (eg: for kernel-nuller ph_shifters=[0, sp.pi/2])
    in: 4
    out: 8 (ph0, ph1, bright0, dark0, dark1, bright1, ph2, ph3)
    symbols:
        sigma  : the photometric ratio (in intensity)
        phi_0   : the phase shifter1
        phi_1   : the phase shifter2
    Build a bracewell combiner with photometric outputs.
    
    
    Returns: the sympy.Matrix of the combiner
    Free symbols can be retrieved in list(M.free_symbols)
    """
    sigma = sp.symbols("sigma", real=True)
    phi = sp.Matrix(sp.symbols('phi0:{}'.format(2), real=True))
    psplitter1 = sp.Matrix([[sp.sqrt(sigma)],
                            [sp.sqrt(1-sigma)]])
    psplitter2 = kernuller.crossover@psplitter1
    #fprint(psplitter1, "\mathbf{Y}_1 = ")
    #fprint(psplitter2, "\mathbf{Y}_2 = ")
    #fprint(kernuller.xcoupler, "\mathbf{X} = ")

    A = sp.diag(psplitter1, psplitter1, psplitter2, psplitter2)
    B = sp.diag(1, kernuller.crossover, sp.eye(2), kernuller.crossover, 1)
    C = sp.diag(sp.eye(2),kernuller.xcoupler, kernuller.xcoupler, sp.eye(2))
    C2 = sp.diag(sp.eye(4), kernuller.crossover, sp.eye(2))
    D = sp.diag(sp.eye(3), kernuller.ph_shifter(phi[0]), kernuller.ph_shifter(phi[1]), sp.eye(3))
    E = sp.diag(sp.eye(3), kernuller.xcoupler, sp.eye(3))
    E1  = sp.diag(sp.eye(3), kernuller.ph_shifter(sp.pi/2), kernuller.ph_shifter(sp.pi), sp.eye(3))


    combiner = E1@E@D@C2@C@B@A
    #fprint(combiner2, "\mathbf{M}_2 = ")
    
    if ph_shifters is not None:
        thesubs = [(phi[0], ph_shifters[0]),
                  (phi[1], ph_shifters[1])]
        combiner = combiner.subs(thesubs)
    
    return combiner

    
def test_combiners():
    b = bracewell_ph()
    a = angel_woolf_ph(ph_shifters=[0, sp.pi/2])
    
    sigma = list(b.free_symbols)[0]
    thesubs = [(sigma, 0.05)]
    
    fprint(b)
    Mn = kernuller.sp2np(b.subs(thesubs)).astype(np.complex128)
    fig, axs = kernuller.cmp(Mn, nx=1, out_label=np.arange(4), mainlinewidth=0.05)
    
    fprint(a)
    Mn = kernuller.sp2np(a.subs(thesubs)).astype(np.complex128)
    fig, axs = kernuller.cmp(Mn, nx=2, out_label=np.arange(8), mainlinewidth=0.05)
    
