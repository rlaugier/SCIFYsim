import kernuller
import sympy as sp
import numpy as np
from kernuller import fprint


import logging

logit = logging.getLogger(__name__)




def bracewell_ph(include_masks=False):
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
    if include_masks:
        bright = np.array([False,True,  False, False])
        dark = np.array([False, False, True, False])
        photometric = np.array([True, False, False, True])
    
    if include_masks:
        return combiner, bright, dark, photometric
    else:
        return combiner



def angel_woolf_ph(ph_shifters=None, include_masks=False):
    """
    optional :
    ph_shifters : a list of phase shifters in between the 2 stages
                (eg: for kernel-nuller ph_shifters=[0, sp.pi/2])
    include_masks: If true, the output will include bright, dark and photometric masks
                selecting the relevant outputs
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
    
    if include_masks:
        bright = np.array([False, False, True, False, False, True, False, False])
        dark = np.array([False, False, False, True, True, False, False, False])
        photometric = np.array([True, True, False, False, False, False, True, True])
        return combiner, bright, dark, photometric
    else:
        return combiner

def VIKiNG(ph_shifters=None, include_masks=False):
    """
    optional :
    ph_shifters : a list of phase shifters in between the 2 stages
                (eg: for kernel-nuller ph_shifters=[0, sp.pi/2])
    include_masks: If true, the output will include bright, dark and photometric masks
                selecting the relevant outputs
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
    
    from kernuller.nullers import matrices_4T
    kernel_nuller_4T = matrices_4T[0]
    sigma = sp.symbols("sigma", real=True)
    phi = sp.Matrix(sp.symbols('phi0:{}'.format(2), real=True))
    psplitter1 = sp.Matrix([[sp.sqrt(sigma)],
                            [sp.sqrt(1-sigma)]])
    psplitter2 = kernuller.crossover@psplitter1

    A = sp.diag(psplitter1, psplitter1, psplitter2, psplitter2)
    B = sp.diag(1, kernuller.crossover, sp.eye(2), kernuller.crossover, 1)
    C = sp.diag(sp.eye(2), kernel_nuller_4T, sp.eye(2))
    VIKiNG = C@B@A
    if include_masks:
        bright = np.array([False, False, True, False, False, False, False, False, False, False, False])
        dark = np.array([False, False, False, True, True, True, True, True, True, False, False])
        photometric = np.array([True, True, False, False, False, False, False, False, False,  True, True])
        return VIKiNG, bright, dark, photometric
    else:
        return VIKiNG
    

    
def ABCD():
    """
    Build an ABCD combiner.
    
    
    Returns: the sympy.Matrix of the combiner
    Free symbols can be retrieved in list(M.free_symbols)
    """
    A = sp.diag(kernuller.splitter, kernuller.splitter)
    B = sp.diag(1, kernuller.crossover, 1)
    C = sp.diag(kernuller.ph_shifter(sp.pi/2), sp.eye(3))
    D = sp.diag(kernuller.xcoupler, kernuller.xcoupler)
    ABCD = D@C@B@A
    return ABCD

def GRAVITY():
    """
    Build a 4 input baseline-wise ABCD combiner
    similar in principle to the one used in GRAVITY.
    
    
    Returns: the sympy.Matrix of the combiner
    Free symbols can be retrieved in list(M.free_symbols)
    """
    tricoupler = sp.Matrix([1/sp.sqrt(3) for i in range(3)])
    A = sp.diag(tricoupler, tricoupler, tricoupler, tricoupler)
    B = sp.diag(sp.eye(5), ABCD(), sp.eye(5))
    C = sp.diag(sp.eye(2), kernuller.crossover, kernuller.crossover, sp.eye(2),
                kernuller.crossover,kernuller.crossover, sp.eye(2))
    D = sp.diag(1, kernuller.crossover,kernuller.crossover,
               kernuller.crossover,kernuller.crossover,
               kernuller.crossover,kernuller.crossover, 1)
    E = sp.diag(sp.eye(2), kernuller.crossover,kernuller.crossover,
               kernuller.crossover,kernuller.crossover,
               kernuller.crossover, sp.eye(2))
    F = sp.diag(sp.eye(3), kernuller.crossover,kernuller.crossover,
                kernuller.crossover,kernuller.crossover, sp.eye(3))
    G = sp.diag(ABCD(), sp.eye(2), ABCD(),ABCD(),ABCD(), sp.eye(2), ABCD())
    GRAVITY = G@F@E@D@C@B@A
    return GRAVITY


def GLINT(include_masks=False):
    """
    Build a 4 input baseline-wise Bracewell combiner
    similar in principle to the one used in GLINT.
    
    
    Returns: the sympy.Matrix of the combiner
    Free symbols can be retrieved in list(M.free_symbols)
    """
    b_nuller = bracewell_ph()
    tricoupler = sp.Matrix([1/sp.sqrt(3) for i in range(3)])
    A = sp.diag(tricoupler, tricoupler, tricoupler, tricoupler)
    B = sp.diag(sp.eye(5), b_nuller, sp.eye(5))
    C = sp.diag(sp.eye(2), kernuller.crossover, kernuller.crossover, sp.eye(2),
                kernuller.crossover,kernuller.crossover, sp.eye(2))
    D = sp.diag(1, kernuller.crossover,kernuller.crossover,
               kernuller.crossover,kernuller.crossover,
               kernuller.crossover,kernuller.crossover, 1)
    E = sp.diag(sp.eye(2), kernuller.crossover,kernuller.crossover,
               kernuller.crossover,kernuller.crossover,
               kernuller.crossover, sp.eye(2))
    F = sp.diag(sp.eye(3), kernuller.crossover,kernuller.crossover,
                kernuller.crossover,kernuller.crossover, sp.eye(3))
    G = sp.diag(b_nuller, sp.eye(2), b_nuller,b_nuller,b_nuller, sp.eye(2), b_nuller)
    GLINT = G@F@E@D@C@B@A
    if include_masks:
        bright = np.array([False, True, False, False, # Combination 0-1
                          False, True,                # Combination 1-2
                          False, True, False, False,  # Combination 0-2
                          False, True, False, False,  # Combination 0-3
                          False, True, False, False,  # Combination 1-3
                          False, False,               # Combination 1-2
                          False, True, False, False,  # Combination 2-3
                          ])
        dark = np.array([False, False, True, False, # Combination 0-1
                          False, False,                # Combination 1-2
                          False, False, True, False,  # Combination 0-2
                          False, False, True, False,  # Combination 0-3
                          False, False, True, False,  # Combination 1-3
                          True, False,               # Combination 1-2
                          False, False, True, False,  # Combination 2-3
                          ])
        photometric = np.array([True, False, False, True, # Combination 0-1
                          True, False,                # Combination 1-2
                          True, False, False, True,  # Combination 0-2
                          True, False, False, True,  # Combination 0-3
                          True, False, False, True,  # Combination 1-3
                          False, True,               # Combination 1-2
                          True, False, False, True,  # Combination 2-3
                          ])
        return GLINT, bright, dark, photometric
    else:
        return GLINT

    
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
    
