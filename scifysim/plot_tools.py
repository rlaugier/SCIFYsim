

##################################################
# Some utility functions for plotting
##################################################
import matplotlib.pyplot as plt
import numpy as np
import logging

logit = logging.getLogger(__name__)

# Some colormaps chosen to mirror the default (Cx) series of colors
colortraces = [plt.matplotlib.cm.Blues,
              plt.matplotlib.cm.YlOrBr, # This in order to use oranges for the brouwns
              plt.matplotlib.cm.Greens,
              plt.matplotlib.cm.Reds,
              plt.matplotlib.cm.Purples,
              plt.matplotlib.cm.Oranges,
              plt.matplotlib.cm.RdPu,
              plt.matplotlib.cm.Greys,
              plt.matplotlib.cm.YlOrRd,
              plt.matplotlib.cm.GnBu]

def getcolor(index, tracelength=None):
    """
    Converts an index to one of the basic colors of matplotlib
    """
    return "C"+str(index)
def getcolortrace(index, length):
    trace = colortraces[index]()
    return t
    
def piston2size(piston, dist=140., psz=8., usize=150.):
    """
    Pretty visualization of projected array: Emulates perspective
    dist  : The distance of the observer to the array
    psz   : The diameters of the pupils
    usize : A scaling parameter default = 150
    """
    d = dist + piston
    alpha = np.arctan(psz/d)
    # The size parameter actually points to its area.
    return (alpha * usize)**2



    
def plot_pupil(thearray, thepistons=None, psz=8., usize=150., dist=140., perspective=True):
    """
    Plots the projected 
    dist  : The distance of the observer to the array
    psz   : The diameters of the pupils
    usize : A scaling parameter default = 150
    perspective: whether to simulate an effect of perspective
                with the size of the markers
                
    Returns:
    fig   : The figure 
    """
    # Without piston information, impossible to plot the fake perspective
    if thepistons is None:
        perspective = False
    if perspective:
        projection_string = " (sizes emulate perspective)"
    fig = plt.figure()
    for i in range(thearray.shape[0]):
        if perspective:
            thesize = piston2size(thepistons[i])
        else :
            thesize = 100.*np.ones_like(thepistons[i])
        plt.scatter(thearray[i, 0],thearray[i, 1], c=getcolor(i),
                    s=thesize, alpha=0.5)
    logit.debug(str(thepistons[:]))
    plt.gca().set_aspect("equal")
    plt.title("Array as seen by target%s"%(projection_string))
    plt.xlabel("Az-position (m)")
    plt.ylabel("Alt-position (m)")
    #plt.xlim(np.min([0,:]), np.max([0,:]))
    #plt.ylim(np.min([1,:]), np.max([1,:]))
    plt.show()
    return fig