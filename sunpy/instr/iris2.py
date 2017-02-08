import datetime
import warnings

from astropy.io import fits
import astropy.units as u
from astropy import wcs
from astropy.modeling import fitting
from astropy.modeling.models import custom_model
from astropy.convolution import convolve, Box1DKernel
from astropy.table import Table
from scipy import interpolate

import sunpy.io
from sunpy.time import parse_time
import sunpy.map

import imp
import os.path
sources_iris = imp.load_source("iris", os.path.expanduser(os.path.join("~", "sunpy_dev", "sunpy", "sunpy",
                                                                       "spectra", "sources", "iris.py")))


def calculate_orbital_wavelength_variation(files, slit_pixel_range=None, spline_smoothing=False,
                                           fit_individual_profiles=False):
    """Calculates orbital corrections of spectral line positions using level 2 files.

    For data generated from the April 2014 pipeline, thermal and spacecraft velocity components
    have both been subtracted in the level 2 files.  Therefore, this routine calculates the
    residual orbital (thermal) variation.  For data generated from the Oct 2013 pipeline,
    this routine calculates the total of thermal and spacecraft velocity components.

    Parameters
    ----------
    files: iterable of `str`, e.g. `list` or `tuple`.
        IRIS level-2 spectral data files of the same OBS.
    slit_pixel_range: `tuple`
        Region along slit (in pixels) used of averaging spectral fitting.
        First element=lower pixel in range, second elements=upper pixel in range.
        Default is entire slit.
    spline_smoothing: `bool`
        If True, perform 5 minute smoothing and spline fit to eliminate the
        5min photospheric oscillation.  Default=True
    fit_individual_profiles: `bool`
        If True, fit line profiles from each slit pixel, then average the line positions
        for each raster position.  If False, get the average line spectrum over the entire
        slit region then fit to get the average line position for each raster position.
        Default=False

    Returns
    -------
    orbital_wavelength_variation: `astropy.table.Table`
        Contains the following columns:
        time: `datetime.datetime` objects
            Observation times of wavelength variations.
        FUV: `astropy.quantity.Quantity`
            Wavelength variation in the FUV.
        NUV: `astropy.quantity.Quantity`
            Wavelength variation in the NUV.

    """
    # Define spectral window name containing Ni I line.
    spectral_window = "Mg II k 2796"
    # Define vacuum rest wavelength of Ni I 2799 line.
    wavelength_nii = 2799.474*u.Angstrom
    # Define factor converting NUV spectral pixel size to Angstrom
    specsize = 0.0255
    # Extract data of selected spectral window from files.
    raster = sources_iris.IRISRaster(files, spectral_windows=spectral_window)
    # Get date file was created to know which version of code to run.
    date_created = raster.meta["date data created"]
    date_new_pipeline = datetime.datetime(2014, 4, 1)
    # Extract some auxiliary data from raster object required for data
    # reduction of files produced in old pipeline.
    if date_created < date_new_pipeline:
        spacecraft_velocity = raster.auxiliary_data["OBS_VRIX"]
        orbital_phase = 2. * np.pi * raster.auxiliary_data["OPHASEIX"]
        roll_angle = raster.meta["satellite roll angle"]
        # Check that there are measurement times with good values of
        # spacecraft velocity and orbital phase.
        bad_aux = np.asarray(np.isfinite(spacecraft_velocity)*np.isfinite(orbital_phase)*(-1), dtype=bool)
    # Generate wavelength vector containing only Ni I line.
    wavelength_window = Quantity(raster.data[spectral_window].coords["wavelength"].values,
                                 unit=raster.data[spectral_window].attrs["units"]["wavelength"])
    wavelength_roi_index = np.arange(len(wavelength_window))[np.logical_and(wavelength_window >= 2799.3*u.Angstrom,
                                                                            wavelength_window <= 2799.8*u.Angstrom)]
    # Check that there are at least 5 points in wavelength region.
    # Must have at least this many for a gaussian fit.
    if len(wavelength_roi_index) < 5:
        wavelength_roi_index = np.arange(5)+wavelength_roi_index[0]
    # Extract wavelength of region around Ni I line as array in units
    # of Angstroms.
    wavelength_roi = wavelength_window.to(u.Angstrom).value[wavelength_roi_index]
    # Keep only data within wavelength region of interest.
    raster = raster.data[spectral_window].isel(spectral_axis=slice(wavelength_roi_index[0], wavelength_roi_index[-1]+1))
    # If user selected a sub-region of the slit, reduce data to just
    # that region.
    if slit_pixel_range:
        if len(slit_pixel_range) == 2:
            raster = raster.isel(slit_axis, slice(slit_pixel_range[0], slit_pixel_range[1]))
        else:
            raise TypeError("slit_pixel_range must be tuple of length 2 giving lower and " +
                            "upper bounds of section of slit over which to average line fits.")

    # Derive residual orbital variation.
    # Define array to hold averaged position of Ni I line at different
    # times.
    mean_line_wavelengths = np.empty(len(raster.time))*np.nan
    # Define initial guess for gaussian model.
    g_init = gaussian1d_on_linear_bg(amplitude=-2., mean=wavelength_nii.value,
                                     standard_deviation=2., constant_term=50., linear_term=1.5)
    # Define fitting method.
    fit_g = fitting.LevMarLSQFitter()
    # Depending on user choice, either fit line as measured by each
    # pixel then average line position, or fit average line spectrum
    # from all slit pixels.
    if fit_individual_profiles:
        pixels_in_slit = len(raster.slit_axis)
        for k in range(len(raster.time)):
            pixel_line_wavelengths = np.empty(pixels_in_slit)*np.nan
            data_single_time = raster.isel(raster_axis=k)
            # Iterate through each pixel along slit and perform fit to
            # Ni I line.
            for j in range(2, pixels_in_slit-2):
                # Average over 5 pixels to improve signal-to-noise.
                intensity_mean_5pix = data_single_time.isel(slit_axis=slice(j-2, j+3)).mean(axis=0)
                # Fit gaussian to Ni I line.
                g = fit_g(g_init, wavelength_roi, intensity_mean_5pix)
                # Check that fit is within physically reasonable
                # limits.  If so, store line center wavelength in
                # mean_line_wavelengths array. Else leave element as
                # defined, i.e. NaN.
                if np.isfinite(g.amplitude) and g.amplitude < 0. and \
                            wavelength_roi[0] < g.mean < wavelength_roi[-1]:
                    pixel_line_wavelengths[j] = g.mean
            # Take average of Ni I line position from fits in each
            # pixel.
            mean_line_wavelengths[k] = np.nanmean(pixel_line_wavelengths)
    else:
        # Else average all line profiles then perform fit.
        # Iterate through each measurement time and fit a gaussian to
        # Ni I line.
        for k in range(len(raster.time)):
            # Get data averaged over slit.
            data_single_time = raster.isel(raster_axis=k)
            data_slit_averaged = data_single_time.to_masked_array().mean(axis=0).data
            # Fit Ni I line with a gaussian.
            # Perform fit.
            g = fit_g(g_init, wavelength_roi, data_slit_averaged)
            # Check that fit is within physically reasonable limits.
            # If so, store line center wavelength in
            # mean_line_wavelengths array. Else leave element as
            # defined, i.e. NaN.
            if np.isfinite(g.amplitude) and g.amplitude < 0. and \
                        wavelength_roi[0] < g.mean < wavelength_roi[-1]:
                mean_line_wavelengths[k] = g.mean
            # If data produced by old pipeline, subtract spacecraft velocity
            # from the line position.
            if date_created < date_new_pipeline:
                mean_line_wavelengths[k] = \
                    mean_line_wavelengths[k]-spacecraft_velocity[k]/3e8*wavelength_nii.to(u.Angstrom).value

    # Mark abnormal values.  Thermal drift is of the order of 2
    # unsummed wavelength pixels peak-to-peak.
    w_abnormal = np.where(np.abs(mean_line_wavelengths-np.nanmedian(mean_line_wavelengths)) >= specsize*2)[0]
    if len(w_abnormal) > 0:
        mean_line_wavelengths[w_abnormal] = np.nan
    # Further data reduction required for files from old pipeline.
    if date_created < date_new_pipeline:
        dw_th_A = mean_line_wavelengths - np.nanmean(mean_line_wavelengths)
        # Change the unit from Angstrom into unsummed wavelength pixel.
        dw_th_p = dw_th_A/specsize
        # Adjust reference wavelength using orbital phase information.
        if not(np.isfinite(orbital_phase)).all():
            warnings.warn("Orbital phase values are invalid.  Thermal drift may be offset by at most one pixel.")
            dw_th = dw_th
            # For absolute wavelength calibration of NUV, the
            # following amount (unit Angstrom) has to be
            # subtracted from the wavelengths.
            abswvl_nuv = np.nanmean(mean_line_wavelengths)-wavelength_nii.to(u.Angstrom).value
        else:
            # Define empirical sine fitting at 0 roll angle shifted by
            # different phase.
            sine_params = [-0.66615146, -1.0, 53.106583-roll_angle/360.*2*np.pi]
            phase_adj=np.nanmean(sine_params[0]*np.sin(sine_params[1]*orbital_phase+sine_params[2]))
            # thermal component of the orbital variation, in the unit of unsummed wavelength pixel
            dw_th=dw_th_p+phase_adj
            # For absolute wavelength calibration of NUV the following
            # amount (unit Angstrom) has to be subtracted from the
            # wavelengths.
            abswvl_nuv = np.nanmean(mean_line_wavelengths)-wavelength_nii.to(u.Angstrom).value-phase_adj*specsize
    else:
        # Calculate relative variation of the line position.
        dw_th = mean_line_wavelengths-np.nanmean(mean_line_wavelengths)

    # If spline_smoothing=True, perform spline fit a smoothing to
    # eliminate the 5 minute photospheric oscillation.
    if spline_smoothing:
        # Define spacing of spline knots in seconds.
        spline_knot_spacing = 300.
        # Create array of time in seconds from first time and
        # calculate duration of fitting period.
        time_s = np.asarray(x.coords["time"]-x.coords["time"][0], dtype=float)/1e9
        duration = time_s[-1]-time_s[0]
        # Check whether there is enough good data for a spline fit.
        if duration < spline_knot_spacing:
            raise ValueError("Not enough data for spline fit.")
        # Check whether there is enough good data for a spline fit.
        wgood = np.isfinite(mean_line_wavelengths)
        ngood = float(sum(wgood))
        wbad = not(np.isfinite(mean_line_wavelengths))
        nbad = float(sum(wbad))
        if nbad/ngood > 0.25:
            raise ValuError("Not enough good data for spline fit.")
        # Smooth residual thermal variation curve to eliminate the
        # 5-min photospheric oscillation.
        # Determine number of smoothing point using 3 point
        # lagrangian derivative.
        deriv_time = np.array([(time_s[i+1]-time_s[i-1])/2. for i in range(1,len(time_s)-1)])
        deriv_time = np.insert(deriv_time, 0, (-3*time_s[0]+4*time_s[1]-time_s[2])/2)
        deriv_time = np.insert(deriv_time, -1, (3*time_s[-1]-4*time_s[-2]+time_s[-3])/2)
        n_smooth = int(spline_knot_spacing/deriv_time.mean())
        if n_smooth < len(wgood):
            dw_good = convolve(dw_th[good], Box1DKernel(n_smooth))
        else:
            dw_good = dw_th[good]
        time_good = time_s[good]
        # Fit spline.
        tck = interpolate.splrep(time_good, dw_good, s=0)
        dw_th = interpolate.splev(time_s, tck)

    # Derive residual orbital curves in FUV and NUV and store
    # in a table.
    times = [datetime.datetime.utcfromtimestamp(t/1e9) for t in raster.coords["time"].values.tolist()]
    # Depeding on which pipeline produced the files...
    if date_created < date_new_pipeline:
        dw_orb_fuv = dw_th * (-0.013) + spacecraft_velocity.to(u.km/u.s).value / (3.e5) * 1370. * u.Angstrom
        dw_orb_nuv = dw_th * 0.0255 + spacecraft_velocity.to(u.km/u.s).value / (3.e5) * 2800. * u.Angstrom
    else:
        dw_orb_fuv = dw_th*(-1)*u.Angstrom
        dw_orb_nuv = dw_th*u.Angstrom

    orbital_wavelength_variation = Table([times, dw_orb_fuv, dw_orb_nuv],
                                         names=("time", "wavelength variation FUV", "wavelength variation NUV"))
    return orbital_wavelength_variation


@custom_model
def gaussian1d_on_linear_bg(x, amplitude=None, mean=None, standard_deviation=None,
                            constant_term=None, linear_term=None):
    return amplitude*np.exp(-((x-mean)/standard_deviation)**2) + constant_term + linear_term*x

