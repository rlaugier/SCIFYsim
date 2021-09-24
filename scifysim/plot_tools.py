

##################################################
# Some utility functions for plotting
##################################################
import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm

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
def getcolortrace(cmap, value, length, alpha=None):
    if not isinstance(cmap, plt.matplotlib.colors.LinearSegmentedColormap):
        thecmap = colortraces[cmap]
    else:
        thecmap = cmap
    trace = thecmap(value/length, alpha=alpha)
    return trace
    
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



    
def plot_pupil(thearray, thepistons=None, psz=8.,
               usize=150., dist=140., perspective=True,
               compass=None, grid=None):
    """
    Plots the projected 
    dist  : The distance of the observer to the array
    psz   : The diameters of the pupils
    usize : A scaling parameter default = 150
    perspective: whether to simulate an effect of perspective
                with the size of the markers
    compass: A pair of positions indicating the direction of North and East
             after transformation by the same projection as the array
             [North_vector[e,n], East_vector[e,n]]
    grid   : Similar to the compass but for a bunch of parallels and meridians
            [parallels[[e0,n0], [e1, n1], ... ],
             meridians[[e0,n0], [e1, n1], ...]]
    
    Returns:
    fig   : The figure 
    """
    # Without piston information, impossible to plot the fake perspective
    if thepistons is None:
        perspective = False
    if perspective:
        projection_string = " (sizes emulate perspective)"
    else:
        projection_string = ""
    fig = plt.figure()
    for i in range(thearray.shape[0]):
        if perspective:
            thesize = piston2size(thepistons[i])
        else :
            thesize = 100.*np.ones_like(thepistons[i])
        plt.scatter(thearray[i, 0],thearray[i, 1], c=getcolor(i),
                    s=thesize, alpha=0.5)
    logit.debug("Pistons: ")
    logit.debug(str(thepistons[:]))
    
    if compass is not None:
        plt.plot((0.,compass[0,0]),
                 (0.,compass[0,1]),
                 "r-", linewidth=5, label="North")
        plt.plot((0.,compass[1,0]),
                 (0.,compass[1,1]),
                 "k-", linewidth=5,  label="east")
    if grid is not None:
        # Plotting parallels:
        for line in grid[0]:
            plt.plot(line[:,0], line[:,1],"k-", linewidth=0.5)
        # Plotting meridians:
        for line in grid[1]:
            plt.plot(line[:,0], line[:,1],"k--", linewidth=0.5)
        
    
    plt.gca().set_aspect("equal")
    plt.title("Array as seen by target%s"%(projection_string))
    plt.xlabel("RA position (m)")
    plt.ylabel("Dec position (m)")
    if compass is not None:
        plt.legend()
    #plt.xlim(np.min([0,:]), np.max([0,:]))
    #plt.ylim(np.min([1,:]), np.max([1,:]))
    plt.show()
    return fig

def plot_projected_pupil(asim, seq_index,
                         grid=False, grid_res=5,
                         compass=True, compass_length=10.,
                         usize=150., dist=140., perspective=True):
    """
    Designed as a wrapper around plot_pupil that also handles
    additional illustration.
    As a contrast to plot_pupil, plot_projected_pupil takes in a
    simulator object.
    The plots are made of the array as seen from the target in meters 
    projected to RA-Dec coordinates.
    
    asim    : Simulator object
    seq_index: The index in the observing sequence (This feature may evolve)
    grid    : Whether to plot a grid of ground position
    grid_res: The number of lines in the grid for each direction
    compass : Whether to plot a little North and East symbol for direction
    compoass_length: In meters the length of the compass needles.
    """
    anarray = asim.obs.statlocs
    
    #Get the pointing of the array:
    altaz, PA = asim.obs.get_position(asim.target, asim.sequence[seq_index])
    #Building a grid
    if grid :
        xx, yy = np.meshgrid(np.linspace(np.min(anarray[:,0]), np.max(anarray[:,0]), grid_res),
                            np.linspace(np.min(anarray[:,1]), np.max(anarray[:,1]), grid_res))
        grid = np.array([xx,yy])
        parallels = np.array(list(zip(grid[0].flat, grid[1].flat))).reshape(grid_res,grid_res,2)
        meridians = np.array(list(zip(grid[0].T.flat, grid[1].T.flat))).reshape(grid_res,grid_res,2)
        formatted_grid = np.array([parallels,meridians])
        projected_grid = formatted_grid.copy()
        for i in range(formatted_grid.shape[0]):
            for j in range(formatted_grid.shape[1]):
                projected_grid[i,j] = asim.obs.get_projected_array(altaz,
                                                               PA=PA,
                                                               loc_array=formatted_grid[i,j])
    else:
        projected_grid = None
        
    if compass:
        mycompass = np.array([[0., 10.],
                             [10.,0.]])
        pcompass = asim.obs.get_projected_array(altaz, PA=PA, loc_array=mycompass)
    else:
        compass = None
        
    p_array = asim.obs.get_projected_array(altaz, PA=PA, loc_array=anarray)
    thepistons = asim.obs.get_projected_geometric_pistons(altaz)
    
    fig = plot_pupil(p_array, thepistons, compass=pcompass,
                    usize=usize, dist=dist, perspective=perspective,
                    grid=projected_grid)
    
    return fig

def plot_phasescreen(theinjector, show=True, noticks=True, screen_index=True):
    import matplotlib.pyplot as plt
    import scifysim as sf
    if not isinstance(theinjector, sf.injection.injector):
        raise ValueError("Expects an injector module")
    # Showing the wavefronts
    #tweaking the colormap showing the pupil cutout
    current_cmap = plt.matplotlib.cm.get_cmap("coolwarm")
    current_cmap.set_bad(color='black')
    fig = plt.figure(figsize=(8,4),dpi=100)
    for i in range(theinjector.ntelescopes):
        plt.subplot(1,theinjector.ntelescopes,i+1)
        plt.imshow((theinjector.focal_plane[i][0]._phs/theinjector.focal_plane[i][0].pupil),
                           cmap=current_cmap)
        if noticks:
            plt.xticks([])
            plt.yticks([])
        if screen_index:
            plt.title(f"Pupil {i}")
    if show:
        plt.show()
    return fig
    
    
    

def plot_injection(theinjector, show=True, noticks=True):
    """
    Provides a view of the injector status.
    Plots the phase screen for each pupil.
    """
    import matplotlib.pyplot as plt
    import scifysim as sf
    if not isinstance(theinjector, sf.injection.injector):
        raise ValueError("Expects an injector module")
    # Showing the wavefronts
    pscreen = plot_phasescreen(theinjector, show=False)
    # Showing the injection profiles
    if show:
        pscreen.show()
    focal_plane = plt.figure(figsize=(2*theinjector.ntelescopes,2*2),dpi=100)
    for i in range(theinjector.ntelescopes):
        plt.subplot(2,theinjector.ntelescopes,i+1)
        plt.imshow(np.abs(theinjector.focal_planes[i,0]), cmap="Blues")
        CS = plt.contour(theinjector.lpmap[0], levels=theinjector.map_quartiles[0], colors="black")
        plt.clabel(CS, inline=1, fontsize=6)
        if noticks:
            plt.xticks([])
            plt.yticks([])
        plt.subplot(2,theinjector.ntelescopes,i+1+theinjector.ntelescopes)
        plt.imshow(np.abs(theinjector.focal_planes[i,-1]), cmap="Reds")
        CS = plt.contour(theinjector.lpmap[-1], levels=theinjector.map_quartiles[-1], colors="black")
        plt.clabel(CS, inline=1, fontsize=6)
        if noticks:
            plt.xticks([])
            plt.yticks([])
    plt.suptitle("Injection focal plane (contours: LP01 mode quartiles)")
    if show:
        plt.show()

    tindexes = ["Telescope %d"%(i) for i in range(theinjector.injected.shape[0])]
    
    amplitudes = plt.figure(figsize=(6.,2.))
    width = 0.1
    for i in range(theinjector.injected.shape[1]):
        plt.bar(np.arange(4)+i*width,np.abs(theinjector.injected[:,i]), width, label="%.1f µm"%(theinjector.lambda_range[i]*1e6))
    plt.legend(loc="lower center",fontsize=7, title_fontsize=8)
    plt.ylabel("Injection amplitude")
    plt.title("Injection amplitude for each telescope by WL")
    if show:
        plt.show()

    phases = plt.figure(figsize=(6.,2.))
    width = 0.1
    for i in range(theinjector.injected.shape[1]):
        plt.bar(np.arange(4)+i*width,np.angle(theinjector.injected[:,i]), width, label="%.1f µm"%(theinjector.lambda_range[i]*1e6))
    plt.legend(loc="lower center",fontsize=7, title_fontsize=8)
    plt.ylabel("Injection phase [radians]")
    plt.title("Injection phase for each telescope by WL")
    if show:
        plt.show()
        
    return focal_plane, amplitudes, phases

    
def plot_response_map(asim, outputs=None,
                      wavelength=None,
                      sequence_index=None,
                      show=True,
                      save=False,
                      figsize=(12,3),
                      dpi=100,
                      central_marker=True,
                      **kwargs):
    """
    Plot the response map of the instrument for the target and sequence
    wavelength   : np.ndarray containing wl indices (if None: use all the wavelength channels)
    sequence_index : The indices of the sequence to plot
    show         : Whether to call plt.show() for each plot
    save         : either False or a string containing the root of a path like "maps/figures_"
    figsize      : 2 tuple to pass to plt.figure()
    dpi          : The dpi for the maps
    **kwargs     : Additional kwargs to pass to imshow
    add_central_marker: Add a marker at the 0,0 location
    central_marker_size: The size parameter to give the central marker
    central_marker_type: The type of marker to use
    """
    base_params = {'x':0, 'y':0, 's':10., 'c':"w", 'marker':"*"}
    if central_marker is True:
        central_marker = base_params
    elif isinstance(central_marker, dict):
        # Update the params with the passed dict
        for akey in central_marker:
            base_params[akey] = central_marker[akey]
        central_marker = base_params
    if sequence_index is None:
        sequence_index = range(len(asim.sequence))
    if wavelength is None:
        sumall = True
    else:
        sumall = False
    n_outputs = asim.maps.shape[2]
    if outputs is None:
        outputs = np.arange(n_outputs)
    figs = []
    for i in sequence_index:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        for oi, o in enumerate(outputs):
            seqmap = asim.maps[i,:,o,:,:]
            plt.subplot(1,outputs.shape[0], oi+1)
            #outmap = seqmap[:,:,:]
            if sumall:
                themap = seqmap.sum(axis=0)
            else :
                themap = seqmap[wavelength,:,:]
            plt.imshow(themap, extent=asim.map_extent, **kwargs)
            if central_marker is not False:
                plt.scatter(**central_marker)
            plt.title("Output %d"%(o))
            plt.xlabel("Position [mas]")
        if sumall:
            plt.suptitle("The compound maps for all wavelengths %.1f to %.1f $\mu m$ for block %d"%\
                        (np.min(asim.lambda_science_range)*1e6,
                         np.max(asim.lambda_science_range)*1e6,
                         i))
        else:
            plt.suptitle(r"The maps at %.2f $\mu m$"%(asim.lambda_science_range[wavelength]*1e6))

        plt.tight_layout()
        if save is not False:
            plt.savefig(save+"Compound_maps_%04d.png"%(i), dpi=dpi)
        if show:
            plt.show()
            
def plot_opds(integ, step_time):
    """
    Plots the phase error information collected by 
    the integrator object.
    """
    # I should profide an easier access to this time step
    integration_step = step_time
    t = np.arange(integ.summed_signal.shape[0])*integration_step
    fig_phase = plt.figure(dpi=150)
    pup = 1
    plt.plot(t, integ.ft_phase[:,pup], label="Fringe tracker phase")
    plt.plot(t, integ.inj_phase[:,:], label="Injection phase")
    #plt.plot(asim.fringe_tracker.ref_sample_times[:1000],
    #         2*np.pi/3.5e-6*asim.fringe_tracker.dry_piston_series[:1000,pup],
    #        label= "Sample", alpha=0.3)
    plt.title("Residual phase for pupil %d"%(pup))
    plt.xlabel("Time [s]")
    plt.ylabel("Phase [rad]")
    plt.legend()
    plt.show()

    fig_amp = plt.figure(dpi=150)
    plt.plot(t, integ.inj_amp[:]**2, label="Injection rate")
    #plt.plot(asim.fringe_tracker.ref_sample_times[:1000],
    #         2*np.pi/3.5e-6*asim.fringe_tracker.dry_piston_series[:1000,pup],
    #        label= "Sample", alpha=0.3)
    plt.title("Residual coupling rate")
    plt.xlabel("Time [s]")
    plt.ylabel("Coupling ")
    plt.ylim(0,0.8)
    plt.legend()
    plt.show()

    fig_outputs = plt.figure()
    pup = 1
    plt.plot(t, integ.summed_signal.sum(axis=1)[:,3:5], label="Dark output signal")
    plt.plot(t, integ.summed_signal.sum(axis=1)[:,3] - integ.summed_signal.sum(axis=1)[:,4], label="Kernel signal")
    #plt.plot(asim.fringe_tracker.ref_sample_times[:1000],
    #         2*np.pi/3.5e-6*asim.fringe_tracker.dry_piston_series[:1000,pup],
    #        label= "Sample", alpha=0.3)
    plt.title("Individual and differential outputs")
    plt.xlabel("Time [s]")
    plt.ylabel("Photons")
    plt.legend()
    plt.show()
    
    return fig_phase, fig_amp, fig_outputs


def plot_corrector_tuning_angel_woolf(corrector,lambs,
                                      combiner,show=True):
    """
    Plots some 
    Currently works only with lambda_science_range
    """
    
    from kernuller.diagrams import plot_chromatic_matrix as cmpc
    import matplotlib.pyplot as plt
    from scifysim.correctors import get_Is
    
    darkout_indices = np.arange(combiner.M.shape[0])[combiner.dark]
    # Normalisation factor is the peak intensity
    normalization = np.sum(np.abs(combiner.Mcn[:,3,:]), axis=-1)**2
    
    orig_vecs = (np.ones_like(corrector.b), np.ones_like(corrector.c))
    current_vecs = (corrector.b, corrector.c)
    origIs = get_Is(orig_vecs, combiner, corrector, lambs)
    bestIs = get_Is(current_vecs, combiner, corrector, lambs)

    nul_plot = plt.figure(dpi=150)
    for i, adarkout in enumerate(darkout_indices):
        plt.plot(lambs, 1/normalization * origIs[:,adarkout], label=f"Original {adarkout}", color=f"C{i}", linestyle=":")
        plt.plot(lambs, 1/normalization * bestIs[:,adarkout], label=f"Adjusted {adarkout}", color=f"C{i}")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Wavelength [m]")
    plt.ylabel("Output contrast (I+/I-)")
    plt.title("On-axis chromatic response")
    if show : plt.show()
    
    corphasor = corrector.get_phasor(lambs)
    
    original_plt_params = plt.rcParams
    plt.style.use("default")
    cmp_plot = cmpc(combiner.M[2:6,:], combiner.lamb, lambs,
                plotout=corphasor, minfrac=0.9, show=show)
    plt.rcParams = original_plt_params
    
    bar_plot = plt.figure()
    plt.bar(np.arange(corrector.b.shape[0]),corrector.b, width=0.2, label="Geometric piston")
    plt.bar(np.arange(corrector.b.shape[0])+0.2,corrector.c, width=0.2, label="ZnSe length")
    plt.bar(np.arange(corrector.b.shape[0])+0.4,-(corrector.nmean-1)*corrector.c, width=0.2, label="Geometric compensation")
    plt.legend()
    plt.xlabel("Input index")
    plt.ylabel("Path length [m]")
    plt.title("Path lengths of corrector")
    if show : plt.show()
        
        
        
    #  The morphology residual (enantiomorph excursion)
    
    thetas = np.linspace(-np.pi, np.pi, 10000)
    comphasor = np.ones(4)[None,:]*np.exp(1j*thetas[:,None])
    
    amat = combiner.Mcn
    aphasor = corrector.get_phasor(lambs)
    amatcomp = np.einsum("ijk, ik -> ijk", amat, aphasor)

    allcor = np.einsum("ik, mk -> mik", amat[:,3,:], comphasor) - np.conjugate(amat[:, 4,:])[None,:,:]
    morph_error_orig = 1/normalization * np.min(np.linalg.norm(allcor, axis=2), axis=0)

    allcor = np.einsum("ik, mk -> mik", amatcomp[:,3,:], comphasor) - np.conjugate(amatcomp[:, 4,:])[None,:,:]
    morph_error_cor = 1/normalization * np.min(np.linalg.norm(allcor, axis=2), axis=0)
    
    morph_plot = plt.figure(dpi=150)
    plt.plot(lambs, morph_error_orig**2,color="C0", linestyle=":",
             label="Initial")
    plt.plot(lambs, morph_error_cor**2,color="C0",
             label="Corrected")
    plt.yscale("log")
    plt.legend(fontsize="x-small")
    plt.xlabel("Wavelength [m]")
    plt.ylabel(f"Shape error $\Lambda$")
    plt.title("Enantiomorph excursion")
    if show : plt.show()

    
    return nul_plot, cmp_plot, bar_plot, morph_plot