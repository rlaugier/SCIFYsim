

##################################################
# Some utility functions for plotting
##################################################
import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm
from scifysim import utilities as util
from pdb import set_trace

logit = logging.getLogger(__name__)

# Some colormaps chosen to mirror to avoid deuteranomalyl Reds is swapped with RdBu
# This avoids confusion between 2 and 3 
colortraces = [plt.matplotlib.cm.Blues,
              plt.matplotlib.cm.YlOrBr, # This in order to use oranges for the brouwns
              plt.matplotlib.cm.Greens,
              plt.matplotlib.cm.RdPu,
              plt.matplotlib.cm.Purples,
              plt.matplotlib.cm.Oranges,
              plt.matplotlib.cm.Reds,
              plt.matplotlib.cm.Greys,
              plt.matplotlib.cm.YlOrRd,
              plt.matplotlib.cm.GnBu]

# Some colormaps chosen to mirror the default (Cx) series of colors
colortraces_0 = [plt.matplotlib.cm.Blues,
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
    
    **Arguments:**
    
    * dist  : The distance of the observer to the array
    * psz   : The diameters of the pupils
    * usize : A scaling parameter default = 150
    """
    d = dist + piston
    alpha = np.arctan(psz/d)
    # The size parameter actually points to its area.
    return (alpha * usize)**2



    
def plot_pupil(thearray, thepistons=None, psz=8.,
               usize=150., dist=140., perspective=True,
               compass=None, grid=None, show=True):
    """
    Plots the projected 
    
    **Arguments:**
    
    * dist  : The distance of the observer to the array
    * psz   : The diameters of the pupils
    * usize : A scaling parameter default = 150
    * perspective: whether to simulate an effect of perspective
      with the size of the markers
    * compass: A pair of positions indicating the direction of North and East
      after transformation by the same projection as the array
      [North_vector[e,n], East_vector[e,n]]
    * grid   : Similar to the compass but for a bunch of parallels and meridians
      [parallels[[e0,n0], [e1, n1], ... ],
      meridians[[e0,n0], [e1, n1], ...]]
    * show   : whether to call ``plt.show`` before returning
    
    **Returns:**
    
    * fig   : The figure 
    
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
    if show:
        plt.show()
    return fig

def plot_projected_pupil(asim, seq_index,
                         grid=False, grid_res=5,
                         compass=True, compass_length=10.,
                         usize=150., dist=140., perspective=True,
                        show=True):
    """
    Designed as a wrapper around plot_pupil that also handles
    additional illustration.
    As a contrast to plot_pupil, plot_projected_pupil takes in a
    simulator object.
    The plots are made of the array as seen from the target in meters 
    projected to RA-Dec coordinates.
    
    **Arguments:**
    
    * asim    : Simulator object
    * seq_index: The index in the observing sequence (This feature may evolve)
    * grid    : Whether to plot a grid of ground position
    * grid_res: The number of lines in the grid for each direction
    * compass : Whether to plot a little North and East symbol for direction
    * compoass_length: In meters the length of the compass needles.
    """
    if asim.space:
        grid = False
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
        mycompass = compass_length*np.array([[0., 10.],
                             [10.,0.]])
        if asim.space:
            pcompass = mycompass
        else:
            pcompass = asim.obs.get_projected_array(altaz, PA=PA, loc_array=mycompass)
    else:
        compass = None
        
    p_array = asim.obs.get_projected_array(altaz, PA=PA)
    if not asim.space:
        thepistons = asim.obs.get_projected_geometric_pistons(altaz)
    else:
        thepistons = np.ones_like(p_array[:,0])
    
    fig = plot_pupil(p_array, thepistons, compass=pcompass,
                    usize=usize, dist=dist, perspective=perspective,
                    grid=projected_grid,
                    show=show)
    
    return fig

def plot_projected_uv(asim, seq_indices=None,
                         grid=False, grid_res=5,
                         compass=True, compass_length=10.,
                         usize=150., dist=140., perspective=True,
                         show=True, dpi=100, legend=True,
                         **kwargs):
    """
    Designed as a wrapper around plot_pupil that also handles
    additional illustration.
    As a contrast to plot_pupil, plot_projected_pupil takes in a
    simulator object.
    The plots are made of the array as seen from the target in meters 
    projected to RA-Dec coordinates.
    
    **Arguments:**
    
    * asim    : Simulator object
    * seq_indices: The index in the observing sequence (This feature may evolve)
      if None: the whole sequence is mapped
    * show    : Whether to call ``plt.show`` before returning
    """
    anarray = asim.obs.statlocs
    if legend is not False:
        all_legends = legend
        legend = True
    if seq_indices is None:
        thesequence = asim.sequence
    else:
        thesequence = asim.sequence[seq_indices]
    alluvs = []
    for atime in thesequence:
        if not asim.space:
            #Get the pointing of the array:
            altaz, PA = asim.obs.get_position(asim.target, atime)
            p_array = asim.obs.get_projected_array(altaz, PA=PA, loc_array=anarray)
        else:
            p_array = asim.obs.get_projected_array(time=atime)
        uvs, indices = util.get_uv(p_array)
        alluvs.append(uvs)
    alluvs = np.array(alluvs)
    #thepistons = asim.obs.get_projected_geometric_pistons(altaz)
    
    fig = plt.figure(dpi=dpi)
    for at in alluvs:
        for i, abl in enumerate(at):
            plt.scatter(abl[0], abl[1], color=f"C{i}", label=i,**kwargs)
            plt.scatter(-abl[0], -abl[1], color=f"C{i}", label=i,**kwargs)
    plt.gca().set_aspect("equal")
    plt.xlabel("Baseline U (RA) [m]")
    plt.ylabel("Baseline V (dec) [m]")
    if legend:
        plt.legend()
    if show:
        plt.show()
    
    return fig, alluvs


def plot_colored_uv(asim, title="Plot of the uv coverage",
    show=True, figsize=None,
    dpi=100, **kwargs):
    wls = asim.lambda_science_range
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for at in asim.sequence:
        asim.point(at, asim.target)
        mybs = asim.obs.uv
        auv = mybs[None,:,:]/wls[:,None,None]
        for asuv, wl in zip(auv, wls):
            # print(asuv)
            col = plt.cm.rainbow(util.trois(wl, wls[0], wls[-1], ymin=0.1))
            plt.scatter(*asuv.T, color=[col], **kwargs)
            plt.scatter(*(-asuv.T), color=[col], **kwargs)
    plt.xlabel(f"$ u = \\frac{{b}}{{\lambda}}$")
    plt.gca().set_aspect("equal")
    plt.title(title)
    if show:
        plt.show()
    return fig


def plot_multiple_maps(maplist, mag,  cmap="viridis", show=True, detector="E",layout=(1,1),
                       single_bar=True, adjust_params=None,
                       fontsize="x-small",titles=None,
                       remove_titles=[],
                       remove_xlabels=[],
                       **kwargs):
    """
    Uses matplotlib to show multiple sensitvity maps at a given star magnitude for
    comparison. 
    
    **hint:** 
    
    .. code-block:: language
    
        adjust_params = {"bottom":0.,
                        "right":0.9,
                        "top":1.,
                        "cax":[0.95, 0.1, 0.01, 0.7],# left, bottom, width, height
                        "orientation":"vertical",
                        "location":"top",
                        "tick_fontsize":7}

    **Parameters:**
    * maplist   : a list or array of map objects
    * mag      : The star magnitude
    * cmap       : The colormap to use
    * detector   : The type of detector test: "E" for energy detector
      "N" for Neyman-Pearson
    * fontsize   : The font size for the titles and captions
    * titles     : A list of titles to add to the plots
    * remove_titles: The indices for which to remove titles
    * remove_xlabels: the indices for which to remove xlabels
    * **kwargs : Keyword arguments to pass to `plt.figure()`

    **Returns** the figure object
    """
    if adjust_params is None:
        adjust_params = {"bottom":0.,
                        "right":0.9,
                        "top":1.,
                        "cax":[0.95, 0.1, 0.01, 0.7],# left, bottom, width, height
                        "orientation":"vertical",
                        "location":"top",
                        "tick_fontsize":7}
    # Looking for the common min and max of all the maps.
    amax = -np.inf
    amin = np.inf
    for i, amap in enumerate(maplist):
        magindex = np.argmin(np.abs(amap.mags-mag))
        if "E" in detector:
            themap = amap.mgs[magindex]
            title = f"Planet mag for $T_E$ dec={amap.dec:.1f} $m_{{star}}$={amap.mags[magindex]:.1f}"
        elif "N" in detector:
            themap = amap.mgs_TNP[magindex]
            title = f"Planet mag for $T_{{NP}}$ dec={amap.dec:.1f} $m_{{star}}$={amap.mags[magindex]:.1f}"
        # Just a hack to avoid the 
        themap_min = np.where(np.isinf(themap), 1000, themap)
        themap_max = np.where(np.isinf(themap), -1000, themap)
        amax = np.nanmax((amax, np.nanmax(themap_max)))
        amin = np.nanmin((amin, np.nanmin(themap_min)))
    fig = plt.figure(**kwargs)
    # Plotting the maps
    for i, amap in enumerate(maplist):
        magindex = np.argmin(np.abs(amap.mags-mag))
        extent = [amap.minx, amap.maxx, amap.minx, amap.maxx]
        if "E" in detector:
            themap = amap.mgs[magindex]
            title = f"Planet mag for $T_E$ dec={amap.dec:.1f}° $m_{{star}}$={amap.mags[magindex]:.1f}"
        elif "N" in detector:
            themap = amap.mgs_TNP[magindex]
            title = f"Planet mag for $T_{{NP}}$ dec={amap.dec:.1f}° $m_{{star}}$={amap.mags[magindex]:.1f}"
        if titles is not None:
            title = titles[i]
        #print(layout[0], layout[1], i+1)
        plt.subplot(layout[0], layout[1], i+1)
        # For single colorbar, we must use common min and max values
        if single_bar:
            plt.imshow(themap, extent=extent, cmap=cmap,
                      vmin=amin, vmax=amax)
        else:
            plt.imshow(themap, extent=extent, cmap=cmap)
        if not single_bar:
            plt.colorbar()
        plt.xlabel("Relative position [mas]", fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
    
    # Tidying up
    
    for i, anax in enumerate(fig.axes):
        if i in remove_titles:
            anax.set_title("")
        if i in remove_xlabels:
            anax.set_xlabel("")
    plt.tight_layout()
    if single_bar:
        plt.subplots_adjust(bottom=adjust_params["bottom"],
                          right=adjust_params["right"],
                          top=adjust_params["top"])
        cax = plt.axes(adjust_params["cax"])
        cbar = plt.colorbar(cax=cax, orientation=adjust_params["orientation"])
        cbar.ax.tick_params(labelsize=adjust_params["tick_fontsize"])
    if show:
        plt.show()
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
    Provides a view of the injector status. Plots the phase screen for each pupil.
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
        plt.imshow(np.abs(theinjector.focal_plane[i][0].getimage()), cmap="Blues")
        CS = plt.contour(theinjector.lpmap[0], levels=theinjector.map_quartiles[0], colors="black")
        plt.clabel(CS, inline=1, fontsize=6)
        if noticks:
            plt.xticks([])
            plt.yticks([])
        plt.subplot(2,theinjector.ntelescopes,i+1+theinjector.ntelescopes)
        plt.imshow(np.abs(theinjector.focal_plane[i][-1].getimage()), cmap="Reds")
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
        plt.bar(np.arange(4)+i*width, np.abs(theinjector.injected[:,i]), width, label="%.1f µm"%(theinjector.lambda_range[i]*1e6))
    plt.legend(loc="lower center",fontsize=7, title_fontsize=8)
    plt.ylabel("Injection amplitude")
    plt.title("Injection amplitude for each telescope by WL")
    if show:
        plt.show()

    phases = plt.figure(figsize=(6.,2.))
    width = 0.1
    for i in range(theinjector.injected.shape[1]):
        plt.bar(np.arange(4)+i*width, np.angle(theinjector.injected[:,i]), width, label="%.1f µm"%(theinjector.lambda_range[i]*1e6))
    plt.legend(loc="lower center",fontsize=7, title_fontsize=8)
    plt.ylabel("Injection phase [radians]")
    plt.title("Injection phase for each telescope by WL")
    if show:
        plt.show()
        
    return focal_plane, amplitudes, phases

def plot_output_sources(asim, integ, lambda_range, margins=4, show=True,
                       t_exp=0.):
    """
    Plot the content of all outputs for each of the different sources:
    
    **Arguments:**
    
    * integ  : An integrator object
    * lambda_range : (*array-like*) The wavelength of interest.
      Find it in ``simulator.lambda_science_range``
    * margins : The margins to leave between the spectra of outputs
      (in number of bar widths)
    * show : (*boolean*) Whether to call ``plt.show`` before returning.
    
    
    **Returns:**
    
    * signalplot:  A pyplot figure
    
    """
    
    shift_step = 1/(lambda_range.shape[0]+2)
    outputs = np.arange(integ.summed_signal.shape[2])
    isources = np.arange(len(integ.sums))
    bottom = np.zeros_like(integ.sums[0])
    pup = 1 # The pupil for which to plot the piston
    print(integ.sums[0].shape)
    signalplot = plt.figure(dpi=100)
    det_sources = [integ.det_sources[0][:,None],
                           integ.det_sources[1][None,None]*np.ones_like(integ.sums[0])]
    bars = []
    for ksource, (thesource, label) in enumerate(zip(integ.sums, integ.source_labels)):
        for ilamb in range(lambda_range.shape[0]):
            if ilamb == 0:
                bars.append(plt.bar(outputs+shift_step*ilamb, thesource[ilamb,:], bottom=bottom[ilamb,:],
                    label=label, width=shift_step, color="C%d"%ksource)) #yerr=noise[ilamb,:]
            else:
                bars.append(plt.bar(outputs+shift_step*ilamb, thesource[ilamb,:], bottom=bottom[ilamb,:],
                    width=shift_step,  color="C%d"%ksource)) #yerr=noise[ilamb,:]
        bottom += np.nan_to_num(thesource)
    for lsource, (thesource, label) in enumerate(zip(det_sources, integ.det_labels)):
        #print("bottom")
        #print(bottom)
        for ilamb in range(lambda_range.shape[0]):
            #print(thesource[ilamb,:])
            if ilamb == 0:
                bars.append(plt.bar(outputs+shift_step*ilamb, thesource[ilamb,:], bottom=bottom[ilamb,:],
                    label=label, width=shift_step, color="C%d"%(lsource+ksource+1))) #yerr=noise[ilamb,:]
            else:
                bars.append(plt.bar(outputs+shift_step*ilamb, thesource[ilamb,:], bottom=bottom[ilamb,:],
                    width=shift_step,  color="C%d"%(lsource+ksource+1))) #yerr=noise[ilamb,:]
        bottom += np.nan_to_num(thesource)
    #plt.legend((bars[i][0] for i in range(len(bars))), source_labels)
    #Handled the legend with an condition in the loop
    plt.legend(loc="upper left", fontsize="x-small")
    plt.xticks(outputs)
    plt.xlabel(r"Output and spectral channel %.1f to %.1f $\mu m$ ($R\approx %.0f$)"%(lambda_range[0]*1e6,
                                                                                     lambda_range[-1]*1e6,
                                                                                     asim.R.mean()))
    plt.title("Integration of %.2f s on %s"%(t_exp, asim.tarname))
    plt.ylabel("Number of photons")
    if show:
        plt.show()
    
    return signalplot

    
def plot_response_map(asim, outputs=None,
                      wavelength=None,
                      sequence_index=None,
                      show=True,
                      save=False,
                      figsize=(12,3),
                      dpi=100,
                      central_marker=True,
                      layout="h",
                      cbar=False,
                      **kwargs):
    """
    Plot the response map of the instrument for the target and sequence
    
    **Arguments:**
    
    * asim         : simulator object. The simulator must contains ``self.maps``
    * outputs      : array-like : An array of outputs for which the maps will be plotted
    * wavelength   : np.ndarray containing wl indices (if None: use all the wavelength channels)
    * sequence_index : The indices of the sequence to plot
    * show         : Whether to call plt.show() for each plot
    * save         : either False or a string containing the root of a path like `maps/figures_`
    * figsize      : 2 tuple to pass to plt.figure()
    * dpi          : The dpi for the maps
    * layout       : "h" for horizontal array of map, "v" for vertical
    * **kwargs     : Additional kwargs to pass to imshow
    * add_central_marker: Add a marker at the 0,0 location
    * central_marker_size: The size parameter to give the central marker
    * central_marker_type: The type of marker to use
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
            if layout == "h":
                plt.subplot(1,outputs.shape[0], oi+1)
            elif layout == "v":
                plt.subplot(outputs.shape[0],1, oi+1)
            elif isinstance(layout, tuple):
                plt.subplot(layout[0], layout[1], oi+1)
            #outmap = seqmap[:,:,:]
            if sumall:
                themap = seqmap.sum(axis=0)
            else :
                themap = seqmap[wavelength,:,:]
            plt.imshow(themap, extent=asim.map_extent, **kwargs)
            if cbar:
                plt.colorbar()
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
        figs.append(fig)
    return figs
            
def plot_differential_map(asim, kernel=None,
                      wavelength=None,
                      sequence_index=None,
                      show=True,
                      save=False,
                      layout=None,
                      figsize=(12,3),
                      dpi=100,
                      central_marker=True,
                      cbar=False,
                      return_difmaps=False,
                      use_precomputed=False,
                      **kwargs):
    """
    Plot the differential response map of the instrument for the target and sequence
    
    **Arguments:**
    
    * asim         : simulator object. The simulator must contains ``self.maps``
    * kernel       : A kernel matrix indicating the output combinations to take. defaults to ``None``
      which uses the ``asim.combiner.K`` matrix.
    * wavelength   : np.ndarray containing wl indices (if None: use all the wavelength channels)
    * sequence_index : The indices of the sequence to plot
    * show         : Whether to call plt.show() for each plot
    * save         : either False or a string containing the root of a path like `maps/figures_`
    * figsize      : 2 tuple to pass to plt.figure()
    * dpi          : The dpi for the maps
    * **kwargs     : Additional kwargs to pass to imshow
    * add_central_marker: Add a marker at the 0,0 location
    * central_marker_size: The size parameter to give the central marker
    * central_marker_type: The type of marker to use
    * use_precomputed: (False) if True, use asim.difmaps pre-computed by extract_difobs_map
    
    **returns:** fig (, difmaps)
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
    if kernel is None:
        if hasattr(asim.combiner, "K"):
            kernel = asim.combiner.K
        else:
            logit.error("Could not find a relevant kernel matrix to plot the map.")
            logit.error("Please pass it to the function or add it as simulator.combiner.K")
            raise AttributeError("could not find the kernel matrix.")
    n_kernels = kernel.shape[0]
    outputs = np.arange(n_kernels)
    if use_precomputed:
        difmaps = asim.difmaps
    else:
        difmaps = np.einsum("k o, s w o x y -> s w k x y", kernel,  asim.maps)
    
    if sumall:
        amax = np.max(np.abs(difmaps.sum(axis=1)))
    else:
        amax = np.max(np.abs(difmaps[:,wavelength,:,:,:]))
    
    
    figs = []
    for i in sequence_index:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        for oi, o in enumerate(outputs):
            seqmap = difmaps[i,:,o,:,:]
            if layout is None:
                plt.subplot(1,outputs.shape[0], oi+1)
            elif isinstance(layout, tuple):
                plt.subplot(layout[0], layout[1], oi+1)
            #outmap = seqmap[:,:,:]
            if sumall:
                themap = seqmap.sum(axis=0)
            else :
                themap = seqmap[wavelength,:,:]
            plt.imshow(themap, extent=asim.map_extent,
                       vmin=-amax, vmax=amax, **kwargs)
            if cbar:
                plt.colorbar()
            if central_marker is not False:
                plt.scatter(**central_marker)
            plt.title("Output %d"%(o))
            plt.xlabel("Position [mas]")
        if sumall:
            plt.suptitle("The compound differential maps for all wavelengths %.1f to %.1f $\mu m$ for block %d"%\
                        (np.min(asim.lambda_science_range)*1e6,
                         np.max(asim.lambda_science_range)*1e6,
                         i))
        else:
            plt.suptitle(r"The differential maps at %.2f $\mu m$"%(asim.lambda_science_range[wavelength]*1e6))

        plt.tight_layout()
        if save is not False:
            plt.savefig(save+"Compound_maps_%04d.png"%(i), dpi=dpi)
        if show:
            plt.show()
        figs.append(fig)
    if return_difmaps:
        return figs, difmaps
    else:
        return figs
            
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


from kernuller.diagrams import plot_chromatic_matrix as cmpc
import matplotlib.pyplot as plt
from scifysim.correctors import get_Is


def plot_corrector_tuning_angel_woolf(corrector,lambs,
                                      combiner, wv_model=None,
                                      out_label=None,
                                      show=True,
                                      maskout=[2,6]):
    """
    Plots some information on the tuning of the combiner using geometric piston
    and chromatic corrector plates.
    
    **Arguments:**
    
    * corrector : A corrector object
    * lambs     : The wavelengths to plot [m]
    * combiner  : A combiner object
    * wv_model  : A water vapor model (`wet_atmo`) to display along
    * show      : Whether to call ``plt.show``
    * out_label : A list of output labels to pass to 
    
    **Plots:**
    
    - Nulling contrast (before and after correction)
    - The corrected matrix matrix plot
    - The length of correction, compensation, and offset
    - Shape excursion (before and after correction)
    
    Currently works only with lambda_science_range
    """
    
    
    darkout_indices = np.arange(combiner.M.shape[0])[combiner.dark]
    # Normalisation factor is the peak intensity
    normalization = np.sum(np.abs(combiner.Mcn[:,3,:]), axis=-1)**2
    
    orig_vecs = np.array([np.ones_like(corrector.b),
                np.ones_like(corrector.c),
                np.ones_like(corrector.e)]).T
    current_vecs = np.array([corrector.b,
                             corrector.c,
                             corrector.e]).T
    origIs = get_Is(orig_vecs, combiner, corrector, lambs)
    bestIs = get_Is(current_vecs, combiner, corrector, lambs)

    nul_plot = plt.figure(dpi=150)
    for i, adarkout in enumerate(darkout_indices):
        plt.plot(lambs, 1/normalization * origIs[:,adarkout], label=f"Original {adarkout}", color=f"C{i}", linestyle=":")
        plt.plot(lambs, 1/normalization * bestIs[:,adarkout], label=f"Adjusted {adarkout}", color=f"C{i}")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Wavelength [m]")
    plt.ylabel(f"Output contrast ($\\frac{{I^-}}{{I^{{peak}}}}$)")
    plt.title("On-axis chromatic response")
    if show : plt.show()
    
    corphasor = corrector.get_phasor(lambs)
    
    original_plt_params = plt.rcParams
    plt.style.use("default")
    output_indices = np.arange(combiner.M.shape[0])[maskout[0]:maskout[1]]
    outlabels = [f"Output {i}" for i in output_indices]
    cmp_plot = cmpc(combiner.M[maskout[0]:maskout[1],:], combiner.lamb, lambs,
                plotout=corphasor, minfrac=0.9, show=show, out_label=outlabels)
    plt.rcParams = original_plt_params
    
    static_tuning_air = corrector.b
    static_tuning_glass = corrector.c
    static_correction_air = -(corrector.nmean-1)*static_tuning_glass
    total_air = static_tuning_air + static_correction_air
    total_glass = static_tuning_glass
    bar_width = 0.15
    
    bar_plot = plt.figure()
    plt.bar(np.arange(corrector.b.shape[0]),static_tuning_air, width=bar_width, label="Geometric tuning")
    plt.bar(np.arange(corrector.b.shape[0])+bar_width,static_correction_air,
            width=bar_width, label="Geometric comp. tuning")
    plt.bar(np.arange(corrector.b.shape[0])+4*bar_width, static_tuning_glass, width=bar_width, label="Glass tuning")
    if wv_model is not None:
        pointing_tuning_glass = wv_model[1,:]
        pointing_tuning_air = wv_model[0,:]
        pointing_correction_air = -(corrector.nmean-1)*pointing_tuning_glass
        total_air = total_air + pointing_tuning_air + pointing_correction_air
        total_glass = total_glass + pointing_tuning_glass
        
        for i in range(total_air.shape[0]):
            plt.text(np.arange(pointing_tuning_air.shape[0])[i]+2*bar_width,
                     total_air[i],  f"{total_air[i]*1000:.2f}mm")
            plt.text(np.arange(pointing_tuning_glass.shape[0])[i]+5*bar_width,
                     total_glass[i], f"{total_glass[i]*1000:.2f}mm")
        
        plt.bar(np.arange(pointing_tuning_air.shape[0])+2*bar_width, pointing_tuning_air,
                bottom=static_correction_air, width=bar_width, label="Geometric pointing K to L")
        plt.bar(np.arange(pointing_correction_air.shape[0])+2.5*bar_width, pointing_correction_air,
                bottom=static_correction_air+pointing_tuning_air, width=bar_width, label="Geometric comp. across L")
        plt.bar(np.arange(pointing_tuning_glass.shape[0])+5*bar_width, pointing_tuning_glass,
                bottom=static_tuning_glass, width=bar_width, label="Glass length atmo.")
    plt.axhline(0, color="k", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.xlabel("Input index")
    plt.ylabel("Path length [m]")
    plt.title("Path lengths of corrector")
    if show :
        plt.show()
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



def make_cursor(loc, size, extent=None, color="k",
               flipy=True, **kwargs):
    if flipy: s = -1
    else: s = 1
    plt.plot(np.ones(2) * loc[1],
             s*( np.array([loc[0] - 2*size, loc[0] - size])),
             color=color, **kwargs)
    plt.plot(np.array([loc[1] - 2*size, loc[1] - size]),
             s*(np.ones(2) * loc[0]),
             color=color, **kwargs)
    
def plot_disk(disk):
    """
    Plots the disk model and shows the distribution of physical parameters.
    
    **Arguments:**
    
    * disk : exozodi_simple object
    
    **Plots:**
    
    - Disk sampling. marker size related to surface density, 
    color related to temperature.
    - Radial distribution plot showing temperature, 
    surface density, and flux density
    """
    
    import astropy.units as u
    from matplotlib.colors import LogNorm

    distance = disk.distance * u.pc.to(u.au) # au
    out = []
    size_scale = LogNorm(vmin=disk.sigma.min(), vmax=disk.sigma.max())
    # color_scale = LogNorm(vmin=disk.t_dust.min().value, vmax=disk.t_dust.max().value)

    def mas2au(x):
        return x * u.mas.to(u.rad) * distance
    def au2mas(x):
        return x / distance * u.rad.to(u.mas)
    
    r, ind = np.unique(disk.r_au, return_index=True)
    t_dust = disk.t_dust.flatten()[ind] # [K]
    s_dens = disk.sigma.flatten()[ind] # [AU^2 / AU^2]
    flux = (disk.ss_orig / disk.ds.flatten()).sum(axis=0)[ind] # [ph / s / m^2 / sr]
    flux_total = disk.ss_orig.sum() # [ph / s / m^2]
    
    fig, ax_dist = plt.subplots()
    ax_dist.set_aspect(1)
    sc = ax_dist.scatter(disk.xx_f, disk.yy_f, s=size_scale(disk.sigma.flatten())*20, 
                         c=disk.t_dust.flatten().value, cmap='cool')
    ax_dist.set_title('Disk Sampling')
    ax_dist.set_xlabel('Relaveive Position [mas]')
    ax_dist.set_ylabel('Relaveive Position [mas]')
    fig.colorbar(sc, label='Temperature [K]')

    ax_au = ax_dist.secondary_xaxis('top', functions=(mas2au, au2mas))
    ax_au.set_xlabel('Relaveive Position [au]')
    
    fig.tight_layout()
    out.append(fig)

    subplot_kw = {'xscale': 'log', 'yscale': 'log'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=True,
                             subplot_kw=subplot_kw)
    fig.suptitle('Disk Properties')
    for ax in axes:
        ax.set_xlabel('Radius [AU]')

    ax_temp, ax_dens, ax_flux = axes
    
    ax_temp.plot(r, t_dust)
    ax_temp.set_ylabel('Temperature [K]')
    
    ax_dens.plot(r, s_dens)
    ax_dens.set_ylabel('Surface Density [$\mathrm{AU^2 / AU^2}$]')
    
    ax_flux.plot(r, flux)
    ax_flux.set_ylabel('Flux Density [$\mathrm{ph / s / m^2 / sr}$]')
    ax_flux.text(0.1, 0.1, f'tot = {flux_total:.1} [ph / s / m^2]', 
                 transform=ax_flux.transAxes, ha='left')
    
    fig.tight_layout()
    out.append(fig)

    return out


def plot_source_position(asim, use_time=True):
    """
    Plots the Alt-Az positon of the target star over the simulation sequence.
    
    **Arguments:**
    
    * asim : simulator object
    * use_time : If True, x-axis is observing time, else Az angle.
    
    """

    def remove_common_prefix(strings):
        """Removes the common prefix from a list of strings."""
        if not strings:
            return strings
        shortest_string = min(strings, key=len)
        for i, char in enumerate(shortest_string):
            if not all(string[i] == char for string in strings):
                return [string[i:] for string in strings]
        return [""] * len(strings)

    target = asim.target
    sequence = asim.sequence
    (altaz,) = asim.obs.get_positions(target, sequence)

    fig, ax = plt.subplots()
    x, y = [], []
    for coord in altaz:
        
        time = coord.obstime.value
        alt = coord.alt.deg
        az = coord.az.deg
        
        y.append(alt)
        if use_time:
            x.append(time[:-7])
        else:
            x.append(az)

    if use_time:
        x = remove_common_prefix(x)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel('Time UTC')
    else:
        ax.set_ylabel('Azimuth [deg]')
        
    ax.plot(x, y)
    ax.set_title('Source Position')
    ax.set_ylabel('Altitude [deg]')
    fig.tight_layout()

    return fig, ax
    

def plot_disk_sensitivity(asim, interp=True, spec_ind=-1, norm=False):
    """
    Plots the spatial distribution of photons received by the instrument
    at each output.
    
    **Arguments:**
    
    * asim : simulator object
    * interp : If True, images use interpolation (better interactive
        plot performance), else plots the sampled points (more accurate 
        representation of simulation). 
    * spec_ind : Spectal index for which the plots are displayed. 
    * norm : normalizes sensitivity maps in a way that increases contrast.
                                                          
    **Plots**
    top-left panel is the disk model, top-right is a photometric output
    middle row is constructive outputs, bottom row is destructive outputs.
    
    **Notes**
    Disk model has units of [ph / s / m^2]. All other plots have units of 
    [ph / sr] for one second of integration, unless norm=True, in which case
    scaling is applied.
    
    """
    if interp:
        import astropy.units as u
        from scipy.interpolate import griddata
        from matplotlib.patches import Ellipse

    disk = asim.src.disk
    x, y = disk.xx_f, disk.yy_f
    
    def plot_interp(x, y, z, ax, norm, cmap=None):

        extent = [x.min(), x.max(), y.min(), y.max()]
        x_axis = np.linspace(*extent[:2], 1_000)
        y_axis = np.linspace(*extent[2:], 1_000)
        xx, yy = np.meshgrid(x_axis, y_axis)
        points = np.array([x, y]).T
        out = griddata(points, z, (xx, yy), method='linear')
        
        width = 2 * disk.ang_r_in * u.rad.to(u.mas)
        height = width * np.cos(disk.angle_inc * u.deg)
        
        elp = Ellipse(disk.offset, width, height, angle=disk.angle_rot,
                      fill=True, ec='k', fc='k')
        
        ax.imshow(out, origin='lower', extent=extent, norm=norm, 
                  interpolation='bilinear', cmap=cmap)
        ax.add_patch(elp)
        
        return ax, out, extent


    array = asim.obs.get_projected_array()
    filtered_starlight = asim.diffuse[0].get_downstream_transmission(asim.lambda_science_range)
    collected = asim.injector.collecting * filtered_starlight * 1.0
    perfect_injection = np.ones((asim.lambda_science_range.shape[0], asim.ntelescopes))\
        * asim.corrector.get_phasor(asim.lambda_science_range)\
        * asim.phasor_disp
        
    outputs = asim.combine_light(disk, perfect_injection,
                                 array, collected,
                                 dosum=False)
    outputs = outputs.swapaxes(0, -1)
    outputs = outputs.swapaxes(0, 1)
    outputs = outputs / disk.ds.flatten()[None, None, :]
    tap_ratio = asim.config.getfloat('configuration', 'photometric_tap')
    
    if norm:
        norm_data = outputs[spec_ind]
        norm = plt.Normalize(norm_data.min(), norm_data.max()*0.20)
    else:
        norm = None
    
    fig, axes = plt.subplots(3, 2, figsize=(8,6.7), sharex=True, sharey=True)
    for ax in axes.flatten():
        ax.set_aspect(1)

    # photometric
    axes[0,0].set_title('Disk Model')
    axes[0,1].set_title('Photometric Sensitivity')
    
    # constructive
    axes[1,0].set_title('Constructive Output 2')
    axes[1,1].set_title('Constructive Output 5')
    
    # destructive
    axes[2,0].set_title('Destructive Output 3')
    axes[2,1].set_title('Destructive Output 4')
    
    if interp:
        ax, out, extent =  plot_interp(x, y, disk.ss_orig[spec_ind], axes[0,0], None, 'inferno')
        ax, out, extent =  plot_interp(x, y, outputs[spec_ind,0,:]/tap_ratio, axes[0,1], norm)
        
        ax, out, extent =  plot_interp(x, y, outputs[spec_ind,2,:], axes[1,0], norm)
        ax, out, extent =  plot_interp(x, y, outputs[spec_ind,5,:], axes[1,1], norm)
        
        ax, out, extent =  plot_interp(x, y, outputs[spec_ind,3,:], axes[2,0], norm)
        ax, out, extent =  plot_interp(x, y, outputs[spec_ind,4,:], axes[2,1], norm)
    else:
        axes[0,0].scatter(x, y, c=disk.ss_orig[spec_ind], s=1, cmap='inferno')
        axes[0,1].scatter(x, y, c=outputs[spec_ind,0,:]/tap_ratio, norm=norm, s=1)

        axes[1,0].scatter(x, y, c=outputs[spec_ind,2,:], norm=norm, s=1)
        axes[1,1].scatter(x, y, c=outputs[spec_ind,5,:], norm=norm, s=1)

        axes[2,0].scatter(x, y, c=outputs[spec_ind,3,:], norm=norm, s=1)
        axes[2,1].scatter(x, y, c=outputs[spec_ind,4,:], norm=norm, s=1)
    
    fig.supxlabel('Relative Position [mas]')
    fig.supylabel('Relative Position [mas]')
    fig.tight_layout()
    
    return fig, axes
    
    
    
    