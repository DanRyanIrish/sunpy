# -*- coding: utf-8 -*-
# Author: Daniel Ryan <ryand5@tcd.ie>

import numpy as np
from astropy.io import fits
from astropy.units.quantity import Quantity
from astropy.table import Table

#from sunpy.spectra.spectrum import SlitSpectrum

class IRISSpectrum():
    """A class to handle IRIS Spectral data."""
    def __init__(self, filename):
        # Open IRIS spectral file and attach header to class.
        hdulist = fits.open(filename)
        self.meta = [hdu.header for hdu in hdulist]
        # Make useful metadata class attributes.
        self.n_windows = int(self.meta[0]["NWIN"])
        self.observation_id = int(self.meta[0]["OBSID"])
        self.observation_description = self.meta[0]["OBS_DESC"]
        self.telescope = self.meta[0]["TELESCOP"]
        self.instrument = self.meta[0]["INSTRUME"]
        self.exposure_time = Quantity(self.meta[0]["EXPTIME"], "s")
        self.dsun = Quantity(self.meta[0]["DSUN_OBS"], unit="m")
        # Extract information on spectral windows.
        self.windows = Table([
            [self.meta[0]["TDESC{0}".format(i)] for i in range(1, self.n_windows+1)],
            [self.meta[0]["TDET{0}".format(i)] for i in range(1, self.n_windows+1)],
            Quantity([self.meta[0]["TWAVE{0}".format(i)]
                      for i in range(1, self.n_windows+1)], unit="angstrom"),
            Quantity([self.meta[0]["TWMIN{0}".format(i)]
                      for i in range(1, self.n_windows+1)], unit="angstrom"),
            Quantity([self.meta[0]["TWMAX{0}".format(i)]
                      for i in range(1, self.n_windows+1)], unit="angstrom")],
            names=("name", "detector type", "brightest wavelength",
                   "min wavelength", "max wavelength"))
        # Extract data from file for each spectral window.
        time_axis = np.array(
            [parse_time(self.meta[0]["STARTOBS"])+datetime.timedelta(seconds=i)
             for i in hdulist[-2].data[:,self.meta[-2]["TIME"]]])
        self.data = Table([SlitSpectrum(hdulist[i+1].data, time_axis,
                                        self._get_axis("slit", self.windows["name"][i]),
                                        self._get_axis("spectral", self.windows["name"][i]))
                           for i in range(self.n_windows)], names=tuple(self.windows["name"]))

    def _get_axis(self, axis_type, window_name):
        if axis_type == "spectral":
            axis_index = 1
        elif axis_type == "slit":
            axis_index = 2
        elif axis_type == "raster":
            axis_index = 3
        else:
            raise ValueError("axis_type must be 'spectral', 'slit', 'raster' or 'time'.")
        window_index = np.where(self.windows["name"] == window_name)[0][0]+1
        header = self.meta[window_index]
        axis = Quantity(header["CRVAL{0}".format(axis_index)] + \
                        header["CDELT{0}".format(axis_index)] * \
                        np.arange(0, header["NAXIS{0}".format(axis_index)]),
                        header["CUNIT{0}".format(axis_index)])
        return axis
