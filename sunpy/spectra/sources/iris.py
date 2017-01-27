# -*- coding: utf-8 -*-
# Author: Daniel Ryan <ryand5@tcd.ie>

import datetime
import collections

import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.units.quantity import Quantity
from astropy.table import Table, vstack
from astropy import wcs
from spectral_cube import SpectralCube

from sunpy.time import parse_time
#from sunpy.spectra.spectrum import SlitSpectrum

import imp
import os.path
spectrum = imp.load_source(
    "spectrum", os.path.join(os.path.expanduser("~"), "sunpy_dev", "sunpy",
                             "sunpy", "spectra", "spectrum.py"))
from spectrum import SlitSpectrum


class IRISSpectrum:
    """A class to handle IRIS spectrograph data.

    Attributes
    ----------
    data : `collections.OrderedDict`
        Ordered dictionary of `sunpy.spectra.spectrum.SlitSpectrum` objects.
        One for each spectral window.
    auxilary_data : `astropy.table.Table`
       Auxilary data from penultimate FITS extension of IRIS FITS file.
    level1_info : `numpy.recarray`
       Additional data from last FITS extension of IRIS FITS file including
       level 1 file names from which data was derived.
    _meta : `list` of lists
        Each sublist gives the various headers for an IRIS spectral file used
        to generate object.
    meta : `dict`
        Useful metadata made easily accessible.  Contains the following keys:
            observation_id : `int`
                IRIS observation ID number for observing mode used during campaign.
            observation_description : `str`
                Description of observing mode used for particular campaign.
            telescope : `str`
                "IRIS"
            instrument : `str`
                "SPEC".  Designates that IRIS spectrograph was used to take observations
                as opposed to slit-jaw imager (SJI).
            dsun : `astropy.units.quantity.Quantity`
                Distance from IRIS to Sun during campaign.
            observation_start : `datetime.datetime`
                Start time of observing campaign.
            observation_end : `datetime.datetime`
                End time of observing campaign.
            n_spectral_windows : `int`
                Number of spectral windows read into object.
            spectral_windows : `astropy.table.Table`
                Information on the spectral windows read into object.  Columns:
                name : `str`
                    Name of spectral window.
                detector type : `str`
                    Spectral range/section of CCD from which data was recorded.
                brightest wavelength : `float`, unit aware
                    Brightest wavelength in spectral window.
                min wavelength : `float`, unit aware
                    Minimum wavelength in spectral window.
                max wavelength : `float`, unit aware
                    Maximum wavelength in spectral window.

    """
    def __init__(self, filenames, spectral_windows="All"):
        """
        Initialises an IRISSpectrum object from a list of filenames.

        Parameters
        ----------
        filenames : `str` or iterable of strings, e.g. list, tuple etc.
            Filenames from which to read data into object.  Filenames must be from
            the same IRIS observing campaign, i.e. have the same OBS start time.
        spectral_windows : "All" or iterable of strings, , e.g. list, tuple etc.
            Spectral windows to be read into object. Default="All". If not default,
            must be an iterable of strings of the window names:
            "C II 1336", "O I 1356", "Si IV 1394", "Si IV 1403", "2832", "2814",
            "Mg II k 2796".

        """
        # If a single filename has been entered as a string, convert
        # to a list of length 1 for consistent syntax below.
        if type(filenames) is str:
            filenames = [filenames]
        # Open first IRIS spectral file and attach useful metadata
        # common to all files.
        hdulist = fits.open(filenames[0])
        self._meta = [[hdu.header for hdu in hdulist]]
        window_headers = [hdu.header for hdu in hdulist[1:-2]]
        self.meta = {"observation_id": int(hdulist[0].header["OBSID"]),
                     "observation_description": hdulist[0].header["OBS_DESC"],
                     "telescope": hdulist[0].header["TELESCOP"],
                     "instrument": hdulist[0].header["INSTRUME"],
                     "dsun": Quantity(hdulist[0].header["DSUN_OBS"], unit="m"),
                     "observation_start": hdulist[0].header["STARTOBS"],
                     "observation_end": hdulist[0].header["ENDOBS"],
                     "satellite_roll_angle": Quantity(float(hdulist[0].header["SAT_ROT"]), unit=u.deg)}
        n_spectral_windows = int(hdulist[0].header["NWIN"])
        # Check user desired spectral windows are in file.
        windows_in_obs = np.array([hdulist[0].header["TDESC{0}".format(i)] for i in range(1, n_spectral_windows+1)])
        if spectral_windows == "All":
            spectral_windows = windows_in_obs
        else:
            spectral_windows = np.asarray(spectral_windows, dtype="U")
            window_is_in_obs = np.asarray([window in windows_in_obs for window in spectral_windows])
            if not all(window_is_in_obs):
                missing_windows = window_is_in_obs == False
                raise ValueError("Spectral windows {0} not in file {1}".format(spectral_windows[missing_windows],
                                                                               filenames[0]))
        # Get indices of FITS extensions corresponding to desired
        # spectral windows.
        window_fits_indices = np.nonzero(np.in1d(windows_in_obs, spectral_windows))[0]+1
        # Extract information on desired spectral windows.
        self.spectral_windows = Table([
            [hdulist[0].header["TDESC{0}".format(i)] for i in window_fits_indices],
            [hdulist[0].header["TDET{0}".format(i)] for i in window_fits_indices],
            Quantity([hdulist[0].header["TWAVE{0}".format(i)] for i in window_fits_indices], unit="angstrom"),
            Quantity([hdulist[0].header["TWMIN{0}".format(i)] for i in window_fits_indices], unit="angstrom"),
            Quantity([hdulist[0].header["TWMAX{0}".format(i)] for i in window_fits_indices], unit="angstrom")],
            names=("name", "detector type", "brightest wavelength", "min wavelength", "max wavelength"))
        # Convert data from each spectral window in first FITS file to
        # an array.  Then combine these arrays into a list.  Each array
        # can then be concatenated with data from subsequent FITS files.
        data = [np.array(hdulist[i].data) for i in window_fits_indices]
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
                if hdulist[0].header["STARTOBS"] != self.meta["observation_start"]:
                    raise IOError(
                        "Files must be part of same observation. Current file has different " +
                        "OBS start time from first file.\n" +
                        "First file: {0}\n".format(filenames[0]) +
                        "Current file: {0}\n".format(filename) +
                        "OBS start time of first file: {0}\n".format(self.meta["observation_start"]) +
                        "OBS start time of current file: {0}".format(hdulist[0].header["STARTOBS"])
                        )
                self._meta.append([hdu.header for hdu in hdulist])
                for i, j in enumerate(window_fits_indices):
                    data[i] = np.concatenate((data[i], np.array(hdulist[j].data)), axis=0)
                auxilary_data = vstack(
                    [auxilary_data, Table(rows=hdulist[-2].data, names=hdulist[-2].header[7:])])
                level1_info = np.concatenate((
                    level1_info, np.array(hdulist[-1].data, dtype=hdulist[-1].data.dtype)))
                hdulist.close()
        # Extract and convert time to datetime objects and then delete
        # time from auxilary data.
        time_axis = np.array([
            parse_time(self.meta["observation_start"])+datetime.timedelta(seconds=i)
            for i in auxilary_data["TIME"]])
        del(auxilary_data["TIME"])
        # Generate dictionary of SlitSpectrum objects for each spectral window.
        self.data = dict([(self.spectral_windows["name"][i],
                           SlitSpectrum(data[i], time_axis, self._get_axis("slit", self.spectral_windows["name"][i]),
                                        self._get_axis("spectral", self.spectral_windows["name"][i])))
                          for i in range(len(window_fits_indices))])
        # Attach auxilary data and level1 info to object.
        self.auxilary_data = auxilary_data
        self.level1_info = level1_info

    def _get_axis(self, axis_type, spectral_window):
        """Retrieves the values along the axis of an IRIS spectrum.

        Parameters
        ----------
        axis_type : `str`
            Axis to retrieve. Value must be: 'spectral', 'slit', 'raster' or 'time'.
        spectral_window : `str`
            Name of spectral window for which axis is being retrieved.

        Returns
        -------
        axis : `astropy.units.quantity.Quantity`
            Values at each point along axis.
        """
        if axis_type == "spectral":
            axis_index = 1
        elif axis_type == "slit":
            axis_index = 2
        elif axis_type == "raster":
            axis_index = 3
        else:
            raise ValueError("axis_type must be 'spectral', 'slit', 'raster' or 'time'.")
        window_index = np.where(self.spectral_windows["name"] == spectral_window)[0][0]+1
        header = self._meta[0][window_index]
        axis = Quantity(header["CRVAL{0}".format(axis_index)] +
                        header["CDELT{0}".format(axis_index)] *
                        np.arange(0, header["NAXIS{0}".format(axis_index)]),
                        header["CUNIT{0}".format(axis_index)])
        return axis
