import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

"""
Usage:
------

import map_manager
themap = map_manager.map_meta(file="data/metamap_dec_XXXX.fits")

# You can directly query the map like so:
sensitivity = themap.get_value((10,14), 7.)

# You may also plot the map
fig = themap.plot_map(mag=8.)
"""



class map_meta(object):
    def __init__(self, asim=None, starmags=None, mgs=None, file=None,
                metadata={}):
        """
        Create or load a meta map object:
        
        LOAD:
        -----
            Parameters:
            -----------
            file    :  The path to a fits file
            
        CREATE:
        -------
            Parameters:
            -----------
            asim    : Simulator object from which map parameters are taken
            starmags : An array of star magnitudes
            mgs     : The planet magnitude maps
            
        """
        if isinstance(file, str):
            self.from_fits(file)
        else:
            self.from_sim(asim, starmags, metadata=metadata)
            self.mgs = mgs
        
        
        
    def from_sim(self, asim, starmags, metadata):
        """
        Builds the meta map from simulator
        
        asim    : Simulator object from which map parameters are taken
        starmags : An array of star magnitudes
        mgs     : The planet magnitude maps
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
        mags_hdu = fits.ImageHDU(self.mags, name="MAGS")
        xs_hdu = fits.ImageHDU(self.xs, name="XS")
        ys_hdu = fits.ImageHDU(self.ys, name="YS")
        hdul = fits.HDUList(hdus=[primary_hdu, mags_hdu, xs_hdu, ys_hdu])
        hdul.writeto(file, overwrite=overwrite)
        
    def from_fits(self, file):
        """
        Loads meta map from file
        
        file    :  The path to a fits file
        """
        hdul = fits.open(file)
        self.minx, self.maxx = hdul[0].header["MINX"], hdul[0].header["MAXX"]
        self.resx = hdul[0].header["RESX"]
        self.xs, self.ys = hdul["XS"].data, hdul["YS"].data
        self.mags = hdul["MAGS"].data
        self.resm = hdul[0].header["RESM"]
        self.minmag, self.maxmag = hdul[0].header["MINMAG"], hdul[0].header["MAXMAG"]
        self.mgs = hdul[0].data
        self.header = hdul[0].header
        self.dec = hdul[0].header["DEC"]
        self.input_order = hdul[0].header["HIERARCH INPUT_ODRDER"]
        self.metadata = dict(self.header)
        hdul.close()
        
        
    def get_loc(self, loc, mag):
        """
        Get the index location from a relative location and star magnitude. Returns 
        a tuple index to query self.mgs
        
        Parameters:
        -----------
        loc      : 2-tuple Relative location on sky in the band of interest
        mag      : Magnitude of the host star 
        """
        magindex = np.argmin(np.abs(self.mags-mag))
        xindex = np.argmin(np.abs(self.xs-loc[1]))
        yindex = np.argmin(np.abs(self.ys-loc[0]))
        return (magindex, yindex, xindex)
    
    def get_value(self, loc, mag):
        """
        Get the limiting magnitude for a planet at the relative location **loc**
        around a star of magnitude **mag**.
        
        Parameters:
        -----------
        loc      : 2-tuple Relative location on sky in the band of interest
        mag      : Magnitude of the host star 
        
        Examples:
        --------
        pmag_limit = mymap.get_value((5.2, 4.8), 4.)
        """
        return self.mgs[self.get_loc(loc, mag)]
    
    def plot_map(self, mag, cmap="viridis", show=True, **kwargs):
        """
        Uses matplotlib to show a map at a given star magnitude.
        
        Parameters:
        -----------
        mag      : The star magnitude
        cmap       : The colormap to use
        **kwargs : Keyword arguments to pass to plt.figure()
        Returns the figure object
        """
        
        magindex = np.argmin(np.abs(self.mags-mag))
        extent = [self.minx, self.maxx, self.minx, self.maxx]
        fig = plt.figure(**kwargs)
        plt.imshow(self.mgs[magindex], extent=extent, cmap=cmap)
        plt.colorbar()
        plt.xlabel("Relative position [mas]")
        plt.title("Planet mag for dec=%1.f m_star=%.1f"%(self.dec, self.mags[magindex]))
        if show:
            plt.show()
        return fig