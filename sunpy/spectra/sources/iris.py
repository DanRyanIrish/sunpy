# -*- coding: utf-8 -*-
# Author: Daniel Ryan <ryand5@tcd.ie>

from datetime import timedelta
from collections import OrderedDict

import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.units.quantity import Quantity
from astropy.table import Table, vstack
from astropy import wcs
from spectral_cube import SpectralCube
import xarray

from sunpy.time import parse_time
from sunpy.instr.iris import iris
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


class IRISRaster_SpectralCube(object):
    """A class to handle data from an IRIS spectrograph file.

    Attributes
    ----------
    data : `dict`
        dictionary of `spectral_cube.SpectralCube` objects. One for each spectral window.
    auxilary_data : `astropy.table.Table`
       Auxilary data from penultimate FITS extension of IRIS FITS file.
    level1_info : `numpy.recarray`
       Additional data from last FITS extension of IRIS FITS file including
       level 1 file names from which data was derived.
    meta : `dict`
        Useful metadata made easily accessible.

    Parameters
    ----------
    filename : `str`
        Filename from which to read data into object.

    """

    def __init__(self, filename):
        """Initialize an IRISRaster object."""
        # Open file.
        hdulist = fits.open(filename)
        # Define indices of hdulist containing primary data.
        window_fits_indices = range(1, len(hdulist)-2)
        # Create table of spectral window info in OBS.
        self.spectral_windows = Table([
            [hdulist[0].header["TDESC{0}".format(i)] for i in window_fits_indices],
            [hdulist[0].header["TDET{0}".format(i)] for i in window_fits_indices],
            Quantity([hdulist[0].header["TWAVE{0}".format(i)] for i in window_fits_indices], unit="angstrom"),
            Quantity([hdulist[0].header["TWMIN{0}".format(i)] for i in window_fits_indices], unit="angstrom"),
            Quantity([hdulist[0].header["TWMAX{0}".format(i)] for i in window_fits_indices], unit="angstrom")],
            names=("name", "detector type", "brightest wavelength", "min wavelength", "max wavelength"))
        # Put data into dict with each entry representing a spectral window
        self.data = dict([(self.spectral_windows["name"][i],
                           SpectralCube(data=np.array(hdulist[j].data), wcs=wcs.WCS(hdulist[j].header)))
                          for i, j in enumerate(window_fits_indices)])
        # Attach auxiliary data.
        self.auxilary_data = Table(rows=hdulist[-2].data, names=hdulist[-2].header[7:])
        # Attach level 1 info
        self.level1_info = np.array(hdulist[-1].data)
        # Put useful metadata into meta attribute.
        self.meta = {"date data created": parse_time(hdulist[0].header["DATE"]),
                     "telescope": hdulist[0].header["TELESCOP"],
                     "instrument": hdulist[0].header["INSTRUME"],
                     "data level": hdulist[0].header["DATA_LEV"],
                     "level 2 reformatting version": hdulist[0].header["VER_RF2"],
                     "level 2 reformatting date": parse_time(hdulist[0].header["DATE_RF2"]),
                     "data_src": hdulist[0].header["DATA_SRC"],
                     "origin": hdulist[0].header["origin"],
                     "build version": hdulist[0].header["BLD_VERS"],
                     "look-up table ID": hdulist[0].header["LUTID"],
                     "observation ID": int(hdulist[0].header["OBSID"]),
                     "observation description": hdulist[0].header["OBS_DESC"],
                     "observation label": hdulist[0].header["OBSLABEL"],
                     "observation title": hdulist[0].header["OBSTITLE"],
                     "observation start": hdulist[0].header["STARTOBS"],
                     "observation end": hdulist[0].header["ENDOBS"],
                     "observation repetitions": hdulist[0].header["OBSREP"],
                     "camera": hdulist[0].header["CAMERA"],
                     "status": hdulist[0].header["STATUS"],
                     "data quantity": hdulist[0].header["BTYPE"],
                     "data unit": hdulist[0].header["BUNIT"],
                     "BSCALE": hdulist[0].header["BSCALE"],
                     "BZERO": hdulist[0].header["BZERO"],
                     "high latitude flag": hdulist[0].header["HLZ"],
                     "SAA": bool(int(hdulist[0].header["SAA"])),
                     "satellite roll angle": Quantity(float(hdulist[0].header["SAT_ROT"]), unit=u.deg),
                     "AEC exposures in OBS": hdulist[0].header["AECNOBS"],
                     "dsun": Quantity(hdulist[0].header["DSUN_OBS"], unit="m"),
                     "IAECEVFL": bool(),
                     "IAECFLAG": bool(),
                     "IAECFLFL": bool(),
                     "FOV Y axis": Quantity(float(hdulist[0].header["FOVY"]), unit="arcsec"),
                     "FOV X axis": Quantity(float(hdulist[0].header["FOVX"]), unit="arcsec"),
                     "FOV center Y axis": Quantity(float(hdulist[0].header["YCEN"]), unit="arcsec"),
                     "FOV center X axis": Quantity(float(hdulist[0].header["XCEN"]), unit="arcsec"),
                     "spectral summing NUV": hdulist[0].header["SUMSPTRN"],
                     "spectral summing FUV": hdulist[0].header["SUMSPTRF"],
                     "spatial summing": hdulist[0].header["SUMSPAT"],
                     "exposure time mean": hdulist[0].header["EXPTIME"],
                     "exposure time min": hdulist[0].header["EXPMIN"],
                     "exposure time max": hdulist[0].header["EXPMAX"],
                     "total exposures in OBS": hdulist[0].header["NEXPOBS"],
                     "number unique raster positions": hdulist[0].header["NRASTERP"],
                     "raster step size mean": hdulist[0].header["STEPS_AV"],
                     "raster step size sigma": hdulist[0].header["STEPS_DV"],
                     "time step size mean": hdulist[0].header["STEPT_AV"],
                     "time step size sigma": hdulist[0].header["STEPT_DV"],
                     "number spectral windows": hdulist[0].header["NWIN"]}
        # Translate some metadata to be more helpful.
        if hdulist[0].header["IAECEVFL"] == "YES":
            self.meta["IAECEVFL"] = True
        if hdulist[0].header["IAECFLAG"] == "YES":
            self.meta["IAECFLAG"] = True
        if hdulist[0].header["IAECFLFL"] == "YES":
            self.meta["IAECFLFL"] = True
        if self.meta["data level"] == 2.:
            if self.meta["camera"] == 1:
                self.meta["camera"] = "spectra"
            elif self.meta["camera"] == 2:
                self.meta["camera"] = "SJI"


class IRISRaster_Xarray(object):
    """An object to hold data from multiple IRIS raster scans."""
    def __init__(self, filenames, spectral_windows="All"):
        """Initializes an IRISRaster object."""
        # If a single filename has been entered as a string, convert
        # to a list of length 1 for consistent syntax below.
        if type(filenames) is str:
            filenames = [filenames]
        # Define some empty variable.
        wcs_objects = dict()
        raster_index_to_file = []
        raster_positions = []
        # Open files and extract data.
        for f, filename in enumerate(filenames):
            hdulist = fits.open(filename)
            # If this is the first file, extract some common metadata.
            if f == 0:
                # Check user desired spectral windows are in file and
                # find corresponding indices of HDUs.
                n_win = int(hdulist[0].header["NWIN"])
                windows_in_obs = np.array([hdulist[0].header["TDESC{0}".format(i)] for i in range(1, n_win+1)])
                if spectral_windows == "All":
                    spectral_windows = windows_in_obs
                    window_fits_indices = range(1, len(hdulist)-2)
                else:
                    spectral_windows = np.asarray(spectral_windows, dtype="U")
                    window_is_in_obs = np.asarray([window in windows_in_obs for window in spectral_windows])
                    if not all(window_is_in_obs):
                        missing_windows = window_is_in_obs == False
                        raise ValueError(
                            "Spectral windows {0} not in file {1}".format(spectral_windows[missing_windows],
                                                                          filenames[0]))
                    window_fits_indices = np.nonzero(np.in1d(windows_in_obs, spectral_windows))[0]+1
                # Create table of spectral window info in OBS.
                self.spectral_windows = Table([
                    [hdulist[0].header["TDESC{0}".format(i)] for i in window_fits_indices],
                    [hdulist[0].header["TDET{0}".format(i)] for i in window_fits_indices],
                    Quantity([hdulist[0].header["TWAVE{0}".format(i)] for i in window_fits_indices], unit="angstrom"),
                    Quantity([hdulist[0].header["TWMIN{0}".format(i)] for i in window_fits_indices], unit="angstrom"),
                    Quantity([hdulist[0].header["TWMAX{0}".format(i)] for i in window_fits_indices], unit="angstrom")],
                    names=("name", "detector type", "brightest wavelength", "min wavelength", "max wavelength"))
                # Set spectral window name as table index
                self.spectral_windows.add_index("name")
                # Find wavelength represented by each pixel in the
                # spectral dimension by using a WCS object for each spectral
                # window.
                spectral_coords = dict()
                for i, window_name in enumerate(self.spectral_windows["name"]):
                    wcs_spectral = wcs.WCS(hdulist[window_fits_indices[i]].header).sub(1)
                    spectral_coords[window_name] = Quantity(wcs_spectral.all_pix2world(np.arange(
                        hdulist[window_fits_indices[i]].header["NAXIS1"]), 0),
                        unit=wcs_spectral.wcs.cunit[0]).to("Angstrom")[0]
                # Put useful metadata into meta attribute.
                self.meta = {"date data created": parse_time(hdulist[0].header["DATE"]),
                             "telescope": hdulist[0].header["TELESCOP"],
                             "instrument": hdulist[0].header["INSTRUME"],
                             "data level": hdulist[0].header["DATA_LEV"],
                             "level 2 reformatting version": hdulist[0].header["VER_RF2"],
                             "level 2 reformatting date": parse_time(hdulist[0].header["DATE_RF2"]),
                             "DATA_SRC": hdulist[0].header["DATA_SRC"],
                             "origin": hdulist[0].header["origin"],
                             "build version": hdulist[0].header["BLD_VERS"],
                             "look-up table ID": hdulist[0].header["LUTID"],
                             "observation ID": int(hdulist[0].header["OBSID"]),
                             "observation description": hdulist[0].header["OBS_DESC"],
                             "observation label": hdulist[0].header["OBSLABEL"],
                             "observation title": hdulist[0].header["OBSTITLE"],
                             "observation start": hdulist[0].header["STARTOBS"],
                             "observation end": hdulist[0].header["ENDOBS"],
                             "observation repetitions": hdulist[0].header["OBSREP"],
                             "camera": hdulist[0].header["CAMERA"],
                             "status": hdulist[0].header["STATUS"],
                             "data quantity": hdulist[0].header["BTYPE"],
                             "data unit": hdulist[0].header["BUNIT"],
                             "BSCALE": hdulist[0].header["BSCALE"],
                             "BZERO": hdulist[0].header["BZERO"],
                             "high latitude flag": hdulist[0].header["HLZ"],
                             "SAA": bool(int(hdulist[0].header["SAA"])),
                             "satellite roll angle": Quantity(float(hdulist[0].header["SAT_ROT"]), unit=u.deg),
                             "AEC exposures in OBS": hdulist[0].header["AECNOBS"],
                             "dsun": Quantity(hdulist[0].header["DSUN_OBS"], unit="m"),
                             "IAECEVFL": bool(),
                             "IAECFLAG": bool(),
                             "IAECFLFL": bool(),
                             "FOV Y axis": Quantity(float(hdulist[0].header["FOVY"]), unit="arcsec"),
                             "FOV X axis": Quantity(float(hdulist[0].header["FOVX"]), unit="arcsec"),
                             "FOV center Y axis": Quantity(float(hdulist[0].header["YCEN"]), unit="arcsec"),
                             "FOV center X axis": Quantity(float(hdulist[0].header["XCEN"]), unit="arcsec"),
                             "spectral summing NUV": hdulist[0].header["SUMSPTRN"],
                             "spectral summing FUV": hdulist[0].header["SUMSPTRF"],
                             "spatial summing": hdulist[0].header["SUMSPAT"],
                             "exposure time mean": hdulist[0].header["EXPTIME"],
                             "exposure time min": hdulist[0].header["EXPMIN"],
                             "exposure time max": hdulist[0].header["EXPMAX"],
                             "total exposures in OBS": hdulist[0].header["NEXPOBS"],
                             "number unique raster positions": hdulist[0].header["NRASTERP"],
                             "raster step size mean": hdulist[0].header["STEPS_AV"],
                             "raster step size sigma": hdulist[0].header["STEPS_DV"],
                             "time step size mean": hdulist[0].header["STEPT_AV"],
                             "time step size sigma": hdulist[0].header["STEPT_DV"],
                             "spectral windows in OBS": windows_in_obs,
                             "spectral windows in object": spectral_windows,
                             "detector gain": iris.DETECTOR_GAIN,
                             "detector yield": iris.DETECTOR_YIELD,
                             "readout noise": iris.READOUT_NOISE}
                # Translate some metadata to be more helpful.
                if hdulist[0].header["IAECEVFL"] == "YES":
                    self.meta["IAECEVFL"] = True
                if hdulist[0].header["IAECFLAG"] == "YES":
                    self.meta["IAECFLAG"] = True
                if hdulist[0].header["IAECFLFL"] == "YES":
                    self.meta["IAECFLFL"] = True
                if self.meta["data level"] == 2.:
                    if self.meta["camera"] == 1:
                        self.meta["camera"] = "spectra"
                    elif self.meta["camera"] == 2:
                        self.meta["camera"] = "SJI"
                # Define empty dictionary with keys corresponding to
                # spectral windows.  The value of each key will be a
                # list of xarray data arrays, one for each raster scan.
                data_dict = dict([(window_name, None) for window_name in self.spectral_windows["name"]])
                # Record header info of auxiliary data.  Should be
                # consistent between files of same OBS.
                auxiliary_header = hdulist[-2].header
            # Extract the data and meta/auxiliary data.
            # Create WCS object from FITS header and add WCS object
            # wcs dictionary.
            wcs_celestial = wcs.WCS(hdulist[1].header).celestial
            scan_label = "scan{0}".format(f)
            wcs_objects[scan_label] = wcs_celestial
            # Append to list representing the scan labels of each
            # spectrum.
            len_raster_axis = hdulist[1].header["NAXIS3"]
            raster_index_to_file = raster_index_to_file+[scan_label]*len_raster_axis
            # Append to list representing the raster positions of each
            # spectrum.
            raster_positions = raster_positions+list(range(len_raster_axis))
            # Concatenate auxiliary data arrays from each file.
            try:
                auxiliary_data = np.concatenate((auxiliary_data, np.array(hdulist[-2].data)), axis=0)
            except UnboundLocalError as e:
                if e.args[0] == "local variable 'auxiliary_data' referenced before assignment":
                    auxiliary_data = np.array(hdulist[-2].data)
                else:
                    raise e
            # For each spectral window, concatenate data from each file.
            for i, window_name in enumerate(self.spectral_windows["name"]):
                try:
                    data_dict[window_name] = np.concatenate((data_dict[window_name],
                                                             hdulist[window_fits_indices[i]].data))
                except ValueError as e:
                    if e.args[0] == "zero-dimensional arrays cannot be concatenated":
                        data_dict[window_name] = hdulist[window_fits_indices[i]].data
                    else:
                        raise e
            # Close file.
            hdulist.close()
        # Having combined various data from files into common objects,
        # convert into final data formats and attach to class.
        # Convert auxiliary data into Table and attach to class.
        self.auxiliary_data = Table()
        # Enter certain properties into auxiliary data table as
        # quantities with units.
        auxiliary_colnames = [key for key in auxiliary_header.keys()][7:]
        quantity_colnames = [("TIME", "s"), ("PZTX", "arcsec"), ("PZTY", "arcsec"), ("EXPTIMEF", "s"),
                             ("EXPTIMEN", "s"), ("XCENIX", "arcsec"), ("YCENIX", "arcsec")]
        for col in quantity_colnames:
            self.auxiliary_data[col[0]] = _enter_column_into_table_as_quantity(
                col[0], auxiliary_header, auxiliary_colnames, auxiliary_data, col[1])
        # Enter remaining properties into table without units/
        for i, colname in enumerate(auxiliary_colnames):
            self.auxiliary_data[colname] = auxiliary_data[:, auxiliary_header[colname]]
        # Reorder columns so they reflect order in data file.
        self.auxiliary_data = self.auxiliary_data[[key for key in auxiliary_header.keys()][7:]]
        # Rename some columns to be more user friendly.
        rename_colnames = [("EXPTIMEF", "FUV EXPOSURE TIME"), ("EXPTIMEN", "NUV EXPOSURE TIME")]
        for col in rename_colnames:
            self.auxiliary_data.rename_column(col[0], col[1])
        # Add column designating what scan/file number each spectra
        # comes from.  This can be used to determine the corresponding
        # wcs object and level 1 info.
        self.auxiliary_data["scan"] = raster_index_to_file
        # Attach dictionary containing level 1 and wcs info for each file used.
        self.wcs_celestial = wcs_objects
        # Calculate measurement time of each spectrum.
        times = [parse_time(self.meta["observation start"])+timedelta(seconds=s) for s in self.auxiliary_data["TIME"]]
        # Convert data for each spectral window into an an
        # xarray.DataArray and enter into data dictionary.
        self.data = dict([(window_name, xarray.DataArray(data=data_dict[window_name],
                                                         dims=["raster_axis", "slit_axis", "spectral_axis"],
                                                         coords={"wavelength": ("spectral_axis",
                                                                                spectral_coords[window_name].value),
                                                                 "raster_position": ("raster_axis", raster_positions),
                                                                 "time": ("raster_axis", times)},
                                                         name="Intensity [DN]",
                                                         attrs=OrderedDict([(
                                                             "units", {"wavelength": spectral_coords[window_name].unit,
                                                                       "intensity": "DN"})])))
                          for window_name in self.spectral_windows["name"]])

    def convert_DN_to_photons(self, spectral_window):
        """Converts DataArray from DN to photon counts."""
        # Check that DataArray is in units of DN.
        if "DN" not in self.data[spectral_window].attrs["units"]["intensity"]:
            raise ValueError("Intensity units of DataArray are not DN.")
        self.data[spectral_window].data = iris.convert_DN_to_photons(spectral_window)
        self.data[spectral_window].name = "Intensity [photons]"
        self.data[spectral_window].atrrs["units"]["intensity"] = "photons"

    def convert_photons_to_DN(self, spectral_window):
        """Converts DataArray from DN to photon counts."""
        # Check that DataArray is in units of DN.
        if "photons" not in self.data[spectral_window].attrs["units"]["intensity"]:
            raise ValueError("Intensity units of DataArray are not DN.")
        self.data[spectral_window].data = iris.convert_photons_to_DN(spectral_window)
        self.data[spectral_window].name = "Intensity [DN]"
        self.data[spectral_window].atrrs["units"]["intensity"] = "DN"

    def apply_exposure_time_correction(self, spectral_window):
        """Converts DataArray from DN or photons to DN or photons per second."""
        # Check that DataArray is in units of DN.
        if "/s" in self.data[spectral_window].attrs["units"]["intensity"]:
            raise ValueError("Data seems to already be in units per second. '/s' in intensity unit string.")
        detector_type = self.spectral_windows[spectral_window]["detector type"][:3]
        exp_time_s = self.auxiliary_data["{0} EXPOSURE TIME".format(detector_type)].to("s").value
        for i in new_da.data[spectral_window].raster_axis.values:
            self.data[spectral_window].sel(raster_axis=i).data = \
                self.data[spectral_window].sel(raster_axis=i).data/exp_time_s[i]
        # Make new unit reflecting the division by time.
        unit_str = self.data[spectral_window].atrrs["units"]["intensity"]+"/s"
        self.data[spectral_window].atrrs["units"]["intensity"] = unit_str
        name_split = self.data[spectral_window].name.split("[")
        self.data[spectral_window].name = "{0}[{1}]".format(name_split[0], unit_str)

    def calculate_intensity_fractional_uncertainty(self, spectral_window):
        return iris.calculate_intensity_fractional_uncertainty(
            self.data[spectral_window].data, self.data[spectral_window].atrrs["units"][intensity],
            self.spectral_windows[spectral_window]["detector type"][:3])


def _enter_column_into_table_as_quantity(header_property_name, header, header_colnames, data, unit):
    """Used in initiation of IRISRaster to convert auxiliary data to Quantities."""
    index = np.where(np.array(header_colnames) == header_property_name)[0]
    if len(index) == 1:
        index = index[0]
    else:
        raise ValueError("Multiple property names equal to {0}".format(header_property_name))
    pop_colname = header_colnames.pop(index)
    return Quantity(data[:, header[pop_colname]], unit=unit)


def calc_lat_lon_for_raster_position():
    # Find latitude and longitude represented by each
    # pixel in the spectral dimension by using WCS
    # conversion.
    pix = np.array([[y, x] for y in range(hdulist[window_fits_indices[i]].header["NAXIS2"])
                    for x in range(hdulist[window_fits_indices[i]].header["NAXIS3"])])
    latlon = wcs_celestial.celestial.all_pix2world(pix, 0)
    latitude = Quantity(latlon[:, 0].reshape(hdulist[window_fits_indices[i]].header["NAXIS2"],
                                             hdulist[window_fits_indices[i]].header["NAXIS3"]),
                        unit=wcs_celestial.celestial.wcs.cunit[0]).to("arcsec")
    longitude = Quantity(latlon[:, 1].reshape(hdulist[window_fits_indices[i]].header["NAXIS2"],
                                              hdulist[window_fits_indices[i]].header["NAXIS3"]),
                         unit=wcs_celestial.celestial.wcs.cunit[1]).to("arcsec")
