import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

"""
** Usage:**

import map_manager
themap = map_manager.map_meta(file="data/metamap_dec_XXXX.fits")

**You can directly query the map like so:**

.. code-blocks::
    
    sensitivity = themap.get_value((10,14), 7.)

**You may also plot the map:**

.. code-blocks::

    fig = themap.plot_map(mag=8.)
"""



class map_meta(object):
    def __init__(self, asim=None, starmags=None,
                 mgs=None, mgs_TNP=None, file=None,
                metadata={}, dummy=False):
        """
        Create or load a meta map object:
        
        **LOAD:**
        
            **Parameters:**
            
            * file    :  The path to a fits file
            
        **CREATE:**
        
            **Parameters:**
            
            * asim    : Simulator object from which map parameters are taken
            * starmags : An array of star magnitudes
            * mgs     : The planet magnitude maps
            * mgs_TNP : The planet magnitude maps for the NeymanPearson test
            * dummy : If ``True`` Create an empty object. Can be use to facilitate
              simple map operations.
            
        """
        if isinstance(file, str):
            self.from_fits(file)
        else:
            if not dummy:
                self.from_sim(asim, starmags, metadata=metadata)
                self.mgs = mgs
                self.mgs_TNP = mgs_TNP
            else:
                metadata = None
        
        
        
    def from_sim(self, asim, starmags, metadata):
        """
        Builds the meta map from simulator
        
        **Parameters:**
        
        * asim    : Simulator object from which map parameters are taken
        * starmags : An array of star magnitudes
        * mgs     : The planet magnitude maps
        """
        self.minx, self.maxx = np.min(asim.map_extent), np.max(asim.map_extent)
        self.resx = asim.maps.shape[-1]
        self.xs, self.ys = np.linspace(self.minx, self.maxx, self.resx), np.linspace(self.minx, self.maxx, self.resx)
        self.mags = starmags
        self.resm = self.mags.shape[0]
        self.minmag, self.maxmag = np.min(self.mags), np.max(self.mags)
        self.metadata = metadata
        self.dec = metadata["DEC"][0]
        self.input_order = str(asim.order).replace(" ", "_")
        
    def to_fits(self, file, overwrite=True):
        hdr = fits.Header()
        hdr["MINX"] = (self.minx, "Minimum value along x [mas]")
        hdr["MAXX"] = (self.maxx, "Maximum value along x [mas]")
        hdr["RESX"] = (self.resx, "The resolution of the map [pixels]")
        hdr["MINMAG"] = (self.minmag, "The minimum of the range of star magnitudes [mag]")
        hdr["MAXMAG"] = (self.maxmag, "The maximum of the range of star magnitudes [mag]")
        hdr["RESM"] = (self.resm, "The number of magnitudes evaluated")
        hdr["INPUT_ODRDER"] = (self.input_order, "The order in which telescopes are input on the combiner")
        
        for key in self.metadata.keys():
            hdr[key] = self.metadata[key]
        
        primary_hdu = fits.PrimaryHDU(self.mgs, header=hdr)
        if self.mgs_TNP is not None:
            TNP_hdu = fits.ImageHDU(self.mgs_TNP, name="MAGS_TNP")
        mags_hdu = fits.ImageHDU(self.mags, name="MAGS")
        xs_hdu = fits.ImageHDU(self.xs, name="XS")
        ys_hdu = fits.ImageHDU(self.ys, name="YS")
        lhdus = [primary_hdu, mags_hdu, xs_hdu, ys_hdu]
        if self.mgs_TNP is not None:
            lhdus.append(TNP_hdu)
        hdul = fits.HDUList(hdus=lhdus)
        hdul.writeto(file, overwrite=overwrite)
        
    def from_fits(self, file):
        """
        Loads meta map from file
        
        **Parameters:**
        
        * file    :  The path to a fits file
        """
        hdul = fits.open(file)
        self.minx, self.maxx = hdul[0].header["MINX"], hdul[0].header["MAXX"]
        self.resx = hdul[0].header["RESX"]
        self.xs, self.ys = hdul["XS"].data, hdul["YS"].data
        self.mags = hdul["MAGS"].data
        self.resm = hdul[0].header["RESM"]
        self.minmag, self.maxmag = hdul[0].header["MINMAG"], hdul[0].header["MAXMAG"]
        self.mgs = hdul[0].data
        try :
            self.mgs_TNP = hdul["MAGS_TNP"].data
        except KeyError:
            self.mgs_TNP = None
        self.header = hdul[0].header
        self.dec = hdul[0].header["DEC"]
        self.input_order = hdul[0].header["HIERARCH INPUT_ODRDER"]
        self.metadata = dict(self.header)
        hdul.close()
        
        
    def get_loc(self, loc, mag):
        """
        Get the index location from a relative location and star magnitude. Returns 
        a tuple index to query ``self.mgs``
        
        **Parameters:**
        
        * loc      : 2-tuple Relative location on sky in the band of interest
        * mag      : Magnitude of the host star 
        """
        magindex = np.argmin(np.abs(self.mags-mag))
        xindex = np.argmin(np.abs(self.xs-loc[1]))
        yindex = np.argmin(np.abs(self.ys-loc[0]))
        return (magindex, yindex, xindex)
    
    def get_value(self, loc, mag, detector="E"):
        """
        Get the limiting magnitude for a planet at the relative location **loc**
        around a star of magnitude **mag**.
        
        **Parameters:**
        
        * loc      : 2-tuple Relative location on sky in the band of interest
        * mag      : Magnitude of the host star 
        * detector   : The type of detector test: "E" for energy detector
          "N" for Neyman-Pearson
        
        
        **Examples:**
        
        * pmag_limit = mymap.get_value((5.2, 4.8), 4.)
        """
        if "E" in detector:
            mag = self.mgs[self.get_loc(loc, mag)]
        elif "N" in detector:
            mag = self.mgs_TNP[self.get_loc(loc, mag)]
        return mag
    
    def plot_map(self, mag, cmap="viridis", show=True, detector="E", **kwargs):
        """
        Uses matplotlib to show a map at a given star magnitude.
        
        **Parameters:**
        
        * mag      : The star magnitude
        * cmap       : The colormap to use
        * detector   : The type of detector test: "E" for energy detector
          "N" for Neyman-Pearson
        * **kwargs : Keyword arguments to pass to plt.figure()
        
        **Returns** the figure object
        """
        
        magindex = np.argmin(np.abs(self.mags-mag))
        extent = [self.minx, self.maxx, self.minx, self.maxx]
        
        if "E" in detector:
            themap = self.mgs[magindex]
            title = f"Planet mag for $T_E$ dec={self.dec:.1f} m_star={self.mags[magindex]:.1f}"
        elif "N" in detector:
            themap = self.mgs_TNP[magindex]
            title = f"Planet mag for $T_{{NP}}$ dec={self.dec:.1f} m_star={self.mags[magindex]:.1f}"
        fig = plt.figure(**kwargs)
        plt.imshow(themap, extent=extent, cmap=cmap)
        plt.colorbar()
        plt.xlabel("Relative position [mas]")
        plt.title(title)
        if show:
            plt.show()
        return fig