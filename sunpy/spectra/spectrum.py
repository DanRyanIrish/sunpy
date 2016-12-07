# -*- coding: utf-8 -*-
# Author: Florian Mayer <florian.mayer@bitsrc.org>

from __future__ import absolute_import

import datetime

import numpy as np
from astropy.nddata import NDData
from matplotlib import pyplot as plt

from sunpy.time import parse_time

__all__ = ['Spectrum', 'SlitSpectrum']

class Spectrum(np.ndarray):
    """
    Class representing a spectrum.

    Attributes
    ----------
    freq_axis : np.ndarray
        one-dimensional array with the frequency values at every data point

    data : np.ndarray
        one-dimensional array which the intensity at a particular frequency at every data-point.
    """
    def __new__(cls, data, *args, **kwargs):
        return np.asarray(data).view(cls)

    def __init__(self, data, freq_axis):
        self.data = data
        self.freq_axis = freq_axis

    def plot(self, axes=None, **matplot_args):
        """
        Plot spectrum onto current axes. Behaves like matplotlib.pylot.plot()

        Parameters
        ----------
        axes: matplotlib.axes object or None
            If provided the spectrum will be plotted on the given axes.
            Else the current matplotlib axes will be used.
        """

        #Get current axes
        if not axes:
            axes = plt.gca()

        params = {}
        params.update(matplot_args)

        #This is taken from mpl.pyplot.plot() as we are trying to
        #replicate that functionality

        # allow callers to override the hold state by passing hold=True|False
        washold = axes.ishold()
        hold = matplot_args.pop('hold', None)

        if hold is not None:
            axes.hold(hold)
        try:
            lines = axes.plot(self.freq_axis, self, **params)
        finally:
            axes.hold(washold)

        return lines

    def peek(self, **matplot_args):
        """
        Plot spectrum onto a new figure.
        """

        figure = plt.figure()

        lines = self.plot(**matplot_args)

        figure.show()

        return figure


class SlitSpectrum(NDData):
    """A class for handling slit spectrograph data.

    Attributes
    ----------
    data : `astropy.nddata.NDData`
        3D (T x Y x lambda) array giving the intensity at each spectral
        position (lambda) and each position along the slit (Y) for each
        time (T).
    time_axis : array-like of datetime objects.
        Time of each spectrum measurement.
    slit_axis : `numpy.ndarray`
        Array of distance along the slit.
    spectral_axis : `astropy.units.quantity.Quantity`
        wavelength axis of spectrograph observations.
    raster_positions : `numpy.ndarray`
        Array of length T giving raster position of each spectrum.
        Default=None
    slit_coords : `numpy.ndarray`
        TxY dimension array of position on sky of each pixel along slit
        at each measurement time.  Default=None

    """
    def __init__(self, data, time_axis, slit_axis, spectral_axis,
                 data_unit=None, slit_coords=None, raster_positions=None):
        """Initializes a SlitSpectrum object.

        Parameters
        ----------
        data : `numpy.ndarray`-like of `astropy.nddata.NDData`-like
            3D (T x Y x lambda) array giving the intensity at each spectral
            position (lambda) and each position along the slit (Y) for each
            time (T).
        time_axis : array-like of datetime objects.
            Time of each spectrum measurement.
        slit_axis : `astropy.units.quantity.Quantity`
            Array of distance along the slit.
        spectral_axis : `astropy.units.quantity.Quantity`
            wavelength axis of spectrograph observations.

        """
        # Check that input dimensions are consistent.
        if data.shape[0] != len(time_axis):
            raise ValueError("1st dimension of data must equal number" +
                             " of measurement times.")
        if data.shape[1] != len(slit_axis):
            raise ValueError("2nd dimension of data must equal length of slit_axis.")
        if data.shape[2] != len(spectral_axis):
            raise ValueError("3rd dimension of data must equal length of spectral_axis.")
        if raster_positions is not None:
            if data.shape[:2] != slit_positions.shape:
                raise ValueError("1st 2 dimensions of data must " +
                                 "equal dimensions of slit_position. " +
                                 "If you do not want to define " +
                                 "slit_positions, set to None.")
        # Ensure slit and spectral axes are Quantities.
        if type(slit_axis) is not astropy.units.quantity.Quantity:
            raise TypeError("slit_axis must be an astropy Quantity.")
        if type(spectral_axis) is not astropy.units.quantity.Quantity:
            raise TypeError("spectral_axis must be an astropy Quantity.")
        NDData.__init__(self, data, data_unit)
        self.spectral_axis = spectral_axis
        self.slit_axis = slit_axis
        self.time_axis = time_axis
        self.raster_positiosn = raster_positions

    def extract_spectrum1D(self, time_min, time_max, slit_min, slit_max,
                           spectral_min, spectral_max):
        """Extracts 1D spectrum from a section of the slit, a spectral range at a given time."""
        # Determine indices in each dimension of data corresponding to
        # user input spectral, slit, and time ranges.
        w_time = np.arange(len(self.time_axis))[
            np.logical_and(self.time_axis >= time_min, self.time_axis <= time_max)]
        w_spectral = np.arange(len(self.spectral_axis))[
            np.logical_and(self.spectral_axis >= spectral_min,
                           self.spectral_axis <= spectral_max)]
        w_slit = np.arange(len(self.slit_axis))[
            np.logical_and(self.slit_axis >= slit_min,
                           self.slit_axis <= slit_max)]
        # Determine 
        return Spectrum(
            self.data[w_time, :, :].mean(axis=0)[w_slit, :].sum(axis=0)[w_spectral],
            self.spectral_axis)

    def plot_spectrum1D(spectral_min, spectral_max, slit_min, slit_max, spectrum_time):
        """Plots a 1D spectrum from a section of the slit, a spectral range at a given time."""
        self.extract_spectrum1D(spectral_min, spectral_max, slit_min, slit_max,
                                spectrum_time).peek()
