# -*- coding: utf-8 -*-
# Author: Daniel Ryan <ryand5@tcd.ie>

import datetime
import collections

import numpy as np
from astropy.io import fits
from astropy.units.quantity import Quantity
from astropy.table import Table, vstack

from sunpy.time import parse_time

#from sunpy.spectra.spectrum import SlitSpectrum
import imp
import os.path
spectrum = imp.load_source(
    "spectrum", os.path.join(os.path.expanduser("~"), "sunpy_dev", "sunpy",
                             "sunpy", "spectra", "spectrum.py"))
from spectrum import SlitSpectrum

class IRISSpectrum():
    """A class to handle IRIS Spectral data."""
    def __init__(self, filenames):
        # If a single filename has been entered as a string, convert
        # to a list of length 1 for consistent syntax below.
        if type(filenames) is str:
            filenames = [filenames]
        # Open first IRIS spectral file and attach useful metadata
        # common to all files.
        hdulist = fits.open(filenames[0])
        self.meta = [[hdu.header for hdu in hdulist]]
        window_headers = [hdu.header for hdu in hdulist[1:-2]]
        self.n_windows = int(hdulist[0].header["NWIN"])
        self.observation_id = int(hdulist[0].header["OBSID"])
        self.observation_description = hdulist[0].header["OBS_DESC"]
        self.telescope = hdulist[0].header["TELESCOP"]
        self.instrument = hdulist[0].header["INSTRUME"]
        self.dsun = Quantity(hdulist[0].header["DSUN_OBS"], unit="m")
        self.observation_start = hdulist[0].header["STARTOBS"]
        self.observation_end = hdulist[0].header["ENDOBS"]
        # Extract information on spectral windows.
        self.windows = Table([
            [hdulist[0].header["TDESC{0}".format(i)] for i in range(1, self.n_windows+1)],
            [hdulist[0].header["TDET{0}".format(i)] for i in range(1, self.n_windows+1)],
            Quantity([hdulist[0].header["TWAVE{0}".format(i)]
                      for i in range(1, self.n_windows+1)], unit="angstrom"),
            Quantity([hdulist[0].header["TWMIN{0}".format(i)]
                      for i in range(1, self.n_windows+1)], unit="angstrom"),
            Quantity([hdulist[0].header["TWMAX{0}".format(i)]
                      for i in range(1, self.n_windows+1)], unit="angstrom")],
            names=("name", "detector type", "brightest wavelength",
                   "min wavelength", "max wavelength"))
        # Convert data from each spectral window in first FITS file to
        # an array.  Then combine these arrays into a list.  Each array
        # can then be concatenated with data from subsequent FITS files.
        data = [np.array(hdulist[i].data) for i in range(1, self.n_windows+1)]
        # Convert auxilary data from first FITS file to an array.
        # Auxilary data from subsequent FITS files can then be
        # concatenated with this array.
        auxilary_data = Table(rows=hdulist[-2].data, names=hdulist[-2].header[7:])
        # Convert level 1 info from first FITS file to a recarray.
        # Level 1 info from subsequent FITS files can then be
        # concatenated with this array.
        level1_info = np.array(hdulist[-1].data)
        # Close file
        hdulist.close()
        # If more than one FITS file is supplied, Open each file and
        # read out data.
        if len(filenames) > 1:
            for filename in filenames[1:]:
                hdulist = fits.open(filename)
                # Raise error if file not part of same OBS.
                if hdulist[0].header["STARTOBS"] != self.observation_start:
                    raise IOError(
                        "Files must be part of same observation. Current file has different " + \
                        "OBS start time from first file.\n" + \
                        "First file: {0}\n".format(filenames[0]) + \
                        "Current file: {0}\n".format(filename) + \
                        "OBS start time of first file: {0}\n".format(self.observation_start) + \
                        "OBS start time of current file: {0}".format(hdulist[0].header["STARTOBS"])
                        )
                for i in range(self.n_windows):
                    data[i] = np.concatenate((data[i], np.array(hdulist[i+1].data)), axis=0)
                auxilary_data = vstack(
                    [auxilary_data, Table(rows=hdulist[-2].data, names=hdulist[-2].header[7:])])
                level1_info = np.concatenate((
                    level1_info, np.array(hdulist[-1].data, dtype=hdulist[-1].data.dtype)))
                hdulist.close()
        # Extract and convert time to datetime objects and then delete
        # time from auxilary data.
        time_axis = np.array([parse_time(self.observation_start)+datetime.timedelta(seconds=i)
                              for i in auxilary_data["TIME"]])
        del(auxilary_data["TIME"])
        # Generate OrderedDict of SlitSpectrum objects for each spectral window.
        self.data = collections.OrderedDict([
            (self.windows["name"][i],
             SlitSpectrum(data[i], time_axis, self._get_axis("slit", self.windows["name"][i]),
                          self._get_axis("spectral", self.windows["name"][i])))
            for i in range(self.n_windows)])

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
        header = self.meta[0][window_index]
        axis = Quantity(header["CRVAL{0}".format(axis_index)] + \
                        header["CDELT{0}".format(axis_index)] * \
                        np.arange(0, header["NAXIS{0}".format(axis_index)]),
                        header["CUNIT{0}".format(axis_index)])
        return axis
