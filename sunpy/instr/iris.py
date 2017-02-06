"""
Some very beta tools for IRIS
"""

import sunpy.io
import sunpy.time
import sunpy.map

__all__ = ['SJI_to_cube']

# Define some properties of IRIS detectors.  Source: IRIS instrument
# paper.
DETECTOR_GAIN = {"NUV": 18., "FUV": 6., "SJI": 18.}
DETECTOR_YIELD = {"NUV": 1., "FUV": 1.5, "SJI": 1.},
READOUT_NOISE = {"NUV": {"value": 1.2, "unit": "DN"}, "FUV": {"value": 3.1, "unit": "DN"},
                 "SJI": {"value": 1.2, "unit": "DN"}}


def SJI_to_cube(filename, start=0, stop=None, hdu=0):
    """
    Read a SJI file and return a MapCube

    .. warning::
        This function is a very early beta and is not stable. Further work is
        on going to improve SunPy IRIS support.

    Parameters
    ----------
    filename: string
        File to read

    start: int
        Temporal axis index to create MapCube from

    stop: int
        Temporal index to stop MapCube at

    hdu: int
        Choose hdu index

    Returns
    -------
    iris_cube: sunpy.map.MapCube
        A map cube of the SJI sequence
    """

    hdus = sunpy.io.read_file(filename)
    # Get the time delta
    time_range = sunpy.time.TimeRange(hdus[hdu][1]['STARTOBS'],
                                      hdus[hdu][1]['ENDOBS'])
    splits = time_range.split(hdus[hdu][0].shape[0])

    if not stop:
        stop = len(splits)

    headers = [hdus[hdu][1]]*(stop-start)
    datas = hdus[hdu][0][start:stop]

    # Make the cube:
    iris_cube = sunpy.map.Map(list(zip(datas, headers)), cube=True)
    # Set the date/time

    for i, m in enumerate(iris_cube):
        m.meta['DATE-OBS'] = splits[i].center.isoformat()

    return iris_cube

def convert_DN_to_photons(data, detector_type):
    return DETECTOR_GAIN[detector_type]/DETECTOR_YIELD[detector_type]*data

def convert_photons_to_DN(data, detector_type):
    return DETECTOR_YIELD[detector_type]/DETECTOR_GAIN[detector_type]*data

def calculate_intensity_fractional_uncertainty(data, data_unit, detector_type):
    photons_per_dn = DETECTOR_GAIN[detector_type]/DETECTOR_YIELD[detector_type]
    if data_unit == "DN":
        intensity_ph = photons_per_dn*self._convert_DN_to_photons(spectral_window)
    elif data_unit == "photons":
        intensity_ph = self.data[spectral_window].data
    else:
        raise ValueError("Data not in recognized units: {0}".format(data_unit)
    readout_noise_ph = READOUT_NOISE[detector_type]["value"]*photons_per_dn
    uncertainty_ph = np.sqrt(intensity_ph+readout_noise_ph**2.)
    return uncertainty_ph/intensity_ph
