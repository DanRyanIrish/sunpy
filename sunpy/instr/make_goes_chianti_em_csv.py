"""
Create csv file of temperature as a function of GOES long channel flux.

This code produces a csv file of the relationship between isothermal
temperature and the GOES long channel (1-8 angstrom) flux for each 
GOES satellite.  This relationship is used in finding GOES emission 
measure.  Two files are calculated: one assuming coronal abundances and 
one assuming photospheric abundances.  This relationship is determined
using CHIANTI for each GOES satellite and then hard-coded into the SSW
IDL routine goes_get_chianti_em.pro.  This code here simply uses the
hard-coded values copied-and-pasted from goes_get_chianti_em.pro and
turns them into a csv file.  In the future this should be replaced by a
far more elegant solution such as a direct interface between python
and CHIANTI.
WARNING: As with goes_get_chianti_em.pro, the fluxes are assumed
to be true.  So smoothing, background subtraction, corrections to
calibration etc. are not accounted for here.

"""
import sys

import numpy
import csv

# The data here was taken from goes_get_chianti_em.pro on date below.
DATETAKEN = '2014-04-18'
# Define current number of GOES satellites
NUMSATS = 15

# Initialize arrays
# Array of temperatures from 0-100 MK in log_10-space in MK
log10temp = numpy.arange(0,2.02,0.02)
# Arrays to hold long channel flux values for each temperature for each
# satellite, one for coronal abundances and one for photospheric.
b8_cor = numpy.ndarray((NUMSATS,101))
b8_pho = numpy.ndarray((NUMSATS,101))

# Copy and paste long channel flux values into b8_cor from
# goes_get_chianti_em.pro assuming coronal abundances.
b8_cor[0,:] = [5.17e-05,9.68e-05,1.78e-04,3.17e-04,5.49e-04,9.19e-04,1.49e-03,
               2.37e-03,3.66e-03,5.57e-03,8.33e-03,1.23e-02,1.80e-02,2.59e-02,
               3.71e-02,5.24e-02,7.30e-02,1.00e-01,1.36e-01,1.81e-01,2.38e-01,
               3.08e-01,3.93e-01,4.97e-01,6.21e-01,7.69e-01,9.42e-01,1.15e+00,
               1.38e+00,1.65e+00,1.97e+00,2.32e+00,2.73e+00,3.18e+00,3.68e+00,
               4.24e+00,4.85e+00,5.52e+00,6.24e+00,7.02e+00,7.85e+00,8.74e+00,
               9.67e+00,1.06e+01,1.16e+01,1.27e+01,1.37e+01,1.48e+01,1.60e+01,
               1.71e+01,1.82e+01,1.94e+01,2.05e+01,2.17e+01,2.28e+01,2.39e+01,
               2.49e+01,2.59e+01,2.69e+01,2.78e+01,2.87e+01,2.96e+01,3.05e+01,
               3.15e+01,3.25e+01,3.35e+01,3.46e+01,3.57e+01,3.69e+01,3.82e+01,
               3.95e+01,4.09e+01,4.24e+01,4.39e+01,4.54e+01,4.71e+01,4.87e+01,
               5.04e+01,5.21e+01,5.38e+01,5.55e+01,5.73e+01,5.90e+01,6.07e+01,
               6.24e+01,6.41e+01,6.57e+01,6.73e+01,6.88e+01,7.03e+01,7.17e+01,
               7.30e+01,7.42e+01,7.54e+01,7.64e+01,7.74e+01,7.83e+01,7.91e+01,
               7.99e+01,8.05e+01,8.11e+01]
b8_cor[1,:] = [5.31e-05,9.95e-05,1.83e-04,3.26e-04,5.64e-04,9.44e-04,1.54e-03,
               2.43e-03,3.76e-03,5.72e-03,8.56e-03,1.26e-02,1.85e-02,2.67e-02,
               3.81e-02,5.38e-02,7.50e-02,1.03e-01,1.39e-01,1.86e-01,2.44e-01,
               3.16e-01,4.04e-01,5.11e-01,6.38e-01,7.90e-01,9.68e-01,1.18e+00,
               1.42e+00,1.70e+00,2.02e+00,2.39e+00,2.80e+00,3.26e+00,3.78e+00,
               4.35e+00,4.98e+00,5.67e+00,6.41e+00,7.22e+00,8.07e+00,8.98e+00,
               9.93e+00,1.09e+01,1.20e+01,1.30e+01,1.41e+01,1.52e+01,1.64e+01,
               1.76e+01,1.87e+01,1.99e+01,2.11e+01,2.23e+01,2.34e+01,2.45e+01,
               2.56e+01,2.66e+01,2.76e+01,2.86e+01,2.95e+01,3.04e+01,3.14e+01,
               3.24e+01,3.34e+01,3.44e+01,3.55e+01,3.67e+01,3.80e+01,3.93e+01,
               4.06e+01,4.21e+01,4.36e+01,4.51e+01,4.67e+01,4.84e+01,5.00e+01,
               5.18e+01,5.35e+01,5.53e+01,5.71e+01,5.88e+01,6.06e+01,6.24e+01,
               6.41e+01,6.58e+01,6.75e+01,6.91e+01,7.07e+01,7.22e+01,7.37e+01,
               7.50e+01,7.63e+01,7.75e+01,7.86e+01,7.96e+01,8.05e+01,8.13e+01,
               8.21e+01,8.27e+01,8.33e+01]
b8_cor[2,:] = [5.31e-05,9.95e-05,1.83e-04,3.26e-04,5.64e-04,9.44e-04,1.54e-03,
               2.43e-03,3.76e-03,5.72e-03,8.56e-03,1.26e-02,1.85e-02,2.67e-02,
               3.81e-02,5.38e-02,7.50e-02,1.03e-01,1.39e-01,1.86e-01,2.44e-01,
               3.16e-01,4.04e-01,5.11e-01,6.38e-01,7.90e-01,9.68e-01,1.18e+00,
               1.42e+00,1.70e+00,2.02e+00,2.39e+00,2.80e+00,3.26e+00,3.78e+00,
               4.35e+00,4.98e+00,5.67e+00,6.41e+00,7.22e+00,8.07e+00,8.98e+00,
               9.93e+00,1.09e+01,1.20e+01,1.30e+01,1.41e+01,1.52e+01,1.64e+01,
               1.76e+01,1.87e+01,1.99e+01,2.11e+01,2.23e+01,2.34e+01,2.45e+01,
               2.56e+01,2.66e+01,2.76e+01,2.86e+01,2.95e+01,3.04e+01,3.14e+01,
               3.24e+01,3.34e+01,3.44e+01,3.55e+01,3.67e+01,3.80e+01,3.93e+01,
               4.06e+01,4.21e+01,4.36e+01,4.51e+01,4.67e+01,4.84e+01,5.00e+01,
               5.18e+01,5.35e+01,5.53e+01,5.71e+01,5.88e+01,6.06e+01,6.24e+01,
               6.41e+01,6.58e+01,6.75e+01,6.91e+01,7.07e+01,7.22e+01,7.37e+01,
               7.50e+01,7.63e+01,7.75e+01,7.86e+01,7.96e+01,8.05e+01,8.13e+01,
               8.21e+01,8.27e+01,8.33e+01]
b8_cor[3,:] = [4.64e-05,8.68e-05,1.59e-04,2.85e-04,4.92e-04,8.24e-04,1.34e-03,
               2.12e-03,3.29e-03,4.99e-03,7.47e-03,1.10e-02,1.61e-02,2.33e-02,
               3.33e-02,4.70e-02,6.54e-02,8.99e-02,1.22e-01,1.62e-01,2.13e-01,
               2.76e-01,3.53e-01,4.46e-01,5.57e-01,6.89e-01,8.45e-01,1.03e+00,
               1.24e+00,1.48e+00,1.76e+00,2.08e+00,2.44e+00,2.85e+00,3.30e+00,
               3.80e+00,4.35e+00,4.95e+00,5.60e+00,6.30e+00,7.04e+00,7.84e+00,
               8.67e+00,9.54e+00,1.04e+01,1.14e+01,1.23e+01,1.33e+01,1.43e+01,
               1.53e+01,1.64e+01,1.74e+01,1.84e+01,1.94e+01,2.04e+01,2.14e+01,
               2.24e+01,2.32e+01,2.41e+01,2.49e+01,2.57e+01,2.66e+01,2.74e+01,
               2.82e+01,2.91e+01,3.01e+01,3.10e+01,3.21e+01,3.31e+01,3.43e+01,
               3.55e+01,3.67e+01,3.80e+01,3.94e+01,4.08e+01,4.22e+01,4.37e+01,
               4.52e+01,4.67e+01,4.82e+01,4.98e+01,5.14e+01,5.29e+01,5.45e+01,
               5.60e+01,5.75e+01,5.89e+01,6.04e+01,6.17e+01,6.30e+01,6.43e+01,
               6.55e+01,6.66e+01,6.76e+01,6.86e+01,6.94e+01,7.02e+01,7.10e+01,
               7.16e+01,7.22e+01,7.27e+01]
b8_cor[4,:] = [4.37e-05,8.18e-05,1.50e-04,2.68e-04,4.64e-04,7.77e-04,1.26e-03,
               2.00e-03,3.10e-03,4.70e-03,7.04e-03,1.04e-02,1.52e-02,2.19e-02,
               3.13e-02,4.42e-02,6.16e-02,8.47e-02,1.15e-01,1.53e-01,2.01e-01,
               2.60e-01,3.32e-01,4.20e-01,5.25e-01,6.49e-01,7.96e-01,9.68e-01,
               1.17e+00,1.40e+00,1.66e+00,1.96e+00,2.30e+00,2.68e+00,3.11e+00,
               3.58e+00,4.10e+00,4.66e+00,5.27e+00,5.93e+00,6.64e+00,7.38e+00,
               8.17e+00,8.99e+00,9.84e+00,1.07e+01,1.16e+01,1.25e+01,1.35e+01,
               1.44e+01,1.54e+01,1.64e+01,1.74e+01,1.83e+01,1.93e+01,2.02e+01,
               2.11e+01,2.19e+01,2.27e+01,2.35e+01,2.43e+01,2.50e+01,2.58e+01,
               2.66e+01,2.74e+01,2.83e+01,2.92e+01,3.02e+01,3.12e+01,3.23e+01,
               3.34e+01,3.46e+01,3.58e+01,3.71e+01,3.84e+01,3.98e+01,4.11e+01,
               4.26e+01,4.40e+01,4.55e+01,4.69e+01,4.84e+01,4.99e+01,5.13e+01,
               5.27e+01,5.41e+01,5.55e+01,5.69e+01,5.81e+01,5.94e+01,6.06e+01,
               6.17e+01,6.27e+01,6.37e+01,6.46e+01,6.54e+01,6.62e+01,6.69e+01,
               6.75e+01,6.80e+01,6.85e+01]
b8_cor[5,:] = [4.45e-05,8.29e-05,1.51e-04,2.69e-04,4.64e-04,7.74e-04,1.26e-03,
               1.98e-03,3.06e-03,4.65e-03,6.94e-03,1.02e-02,1.49e-02,2.15e-02,
               3.06e-02,4.32e-02,6.01e-02,8.24e-02,1.11e-01,1.48e-01,1.95e-01,
               2.52e-01,3.22e-01,4.06e-01,5.07e-01,6.27e-01,7.68e-01,9.33e-01,
               1.12e+00,1.35e+00,1.60e+00,1.89e+00,2.21e+00,2.58e+00,2.98e+00,
               3.43e+00,3.92e+00,4.46e+00,5.04e+00,5.67e+00,6.34e+00,7.05e+00,
               7.79e+00,8.57e+00,9.37e+00,1.02e+01,1.11e+01,1.19e+01,1.28e+01,
               1.37e+01,1.46e+01,1.55e+01,1.65e+01,1.74e+01,1.82e+01,1.91e+01,
               1.99e+01,2.07e+01,2.14e+01,2.22e+01,2.29e+01,2.36e+01,2.43e+01,
               2.50e+01,2.58e+01,2.66e+01,2.74e+01,2.83e+01,2.93e+01,3.02e+01,
               3.13e+01,3.24e+01,3.35e+01,3.47e+01,3.59e+01,3.71e+01,3.84e+01,
               3.97e+01,4.10e+01,4.24e+01,4.37e+01,4.51e+01,4.64e+01,4.78e+01,
               4.91e+01,5.04e+01,5.17e+01,5.29e+01,5.41e+01,5.53e+01,5.63e+01,
               5.74e+01,5.83e+01,5.92e+01,6.01e+01,6.08e+01,6.15e+01,6.22e+01,
               6.27e+01,6.32e+01,6.37e+01]
b8_cor[6,:] = [3.84e-05,7.31e-05,1.36e-04,2.46e-04,4.29e-04,7.24e-04,1.18e-03,
               1.89e-03,2.94e-03,4.49e-03,6.76e-03,1.00e-02,1.48e-02,2.14e-02,
               3.08e-02,4.38e-02,6.13e-02,8.46e-02,1.15e-01,1.54e-01,2.03e-01,
               2.63e-01,3.38e-01,4.28e-01,5.37e-01,6.66e-01,8.18e-01,9.97e-01,
               1.21e+00,1.45e+00,1.73e+00,2.04e+00,2.40e+00,2.81e+00,3.26e+00,
               3.76e+00,4.31e+00,4.92e+00,5.58e+00,6.29e+00,7.05e+00,7.86e+00,
               8.71e+00,9.61e+00,1.05e+01,1.15e+01,1.25e+01,1.35e+01,1.46e+01,
               1.56e+01,1.67e+01,1.78e+01,1.89e+01,2.00e+01,2.10e+01,2.21e+01,
               2.31e+01,2.41e+01,2.50e+01,2.59e+01,2.68e+01,2.78e+01,2.87e+01,
               2.96e+01,3.06e+01,3.17e+01,3.27e+01,3.39e+01,3.51e+01,3.63e+01,
               3.76e+01,3.90e+01,4.04e+01,4.19e+01,4.34e+01,4.50e+01,4.66e+01,
               4.83e+01,4.99e+01,5.16e+01,5.33e+01,5.50e+01,5.67e+01,5.84e+01,
               6.00e+01,6.16e+01,6.32e+01,6.48e+01,6.63e+01,6.77e+01,6.90e+01,
               7.03e+01,7.15e+01,7.27e+01,7.37e+01,7.47e+01,7.55e+01,7.63e+01,
               7.70e+01,7.77e+01,7.82e+01]
b8_cor[7,:] = [6.48e-05,1.19e-04,2.15e-04,3.77e-04,6.45e-04,1.07e-03,1.72e-03,
               2.71e-03,4.16e-03,6.28e-03,9.33e-03,1.37e-02,1.99e-02,2.85e-02,
               4.05e-02,5.68e-02,7.88e-02,1.08e-01,1.45e-01,1.93e-01,2.53e-01,
               3.26e-01,4.16e-01,5.24e-01,6.53e-01,8.05e-01,9.85e-01,1.19e+00,
               1.44e+00,1.72e+00,2.04e+00,2.40e+00,2.81e+00,3.26e+00,3.77e+00,
               4.33e+00,4.95e+00,5.62e+00,6.34e+00,7.12e+00,7.94e+00,8.82e+00,
               9.73e+00,1.07e+01,1.17e+01,1.27e+01,1.37e+01,1.48e+01,1.59e+01,
               1.70e+01,1.81e+01,1.92e+01,2.03e+01,2.14e+01,2.24e+01,2.34e+01,
               2.44e+01,2.53e+01,2.61e+01,2.70e+01,2.78e+01,2.86e+01,2.94e+01,
               3.02e+01,3.11e+01,3.20e+01,3.30e+01,3.40e+01,3.51e+01,3.62e+01,
               3.74e+01,3.87e+01,4.00e+01,4.14e+01,4.28e+01,4.42e+01,4.57e+01,
               4.73e+01,4.88e+01,5.04e+01,5.20e+01,5.35e+01,5.51e+01,5.67e+01,
               5.83e+01,5.98e+01,6.13e+01,6.27e+01,6.41e+01,6.54e+01,6.67e+01,
               6.79e+01,6.90e+01,7.01e+01,7.11e+01,7.20e+01,7.28e+01,7.35e+01,
               7.42e+01,7.48e+01,7.53e+01]
b8_cor[8,:] = [6.83e-05,1.25e-04,2.25e-04,3.95e-04,6.73e-04,1.11e-03,1.79e-03,
               2.81e-03,4.32e-03,6.50e-03,9.65e-03,1.41e-02,2.05e-02,2.94e-02,
               4.17e-02,5.84e-02,8.09e-02,1.11e-01,1.49e-01,1.98e-01,2.59e-01,
               3.34e-01,4.25e-01,5.35e-01,6.67e-01,8.22e-01,1.00e+00,1.22e+00,
               1.46e+00,1.75e+00,2.07e+00,2.44e+00,2.85e+00,3.32e+00,3.83e+00,
               4.40e+00,5.02e+00,5.70e+00,6.43e+00,7.21e+00,8.04e+00,8.92e+00,
               9.85e+00,1.08e+01,1.18e+01,1.28e+01,1.39e+01,1.49e+01,1.60e+01,
               1.71e+01,1.82e+01,1.93e+01,2.04e+01,2.15e+01,2.26e+01,2.36e+01,
               2.45e+01,2.54e+01,2.63e+01,2.71e+01,2.79e+01,2.87e+01,2.95e+01,
               3.03e+01,3.12e+01,3.21e+01,3.30e+01,3.40e+01,3.51e+01,3.62e+01,
               3.74e+01,3.87e+01,4.00e+01,4.13e+01,4.27e+01,4.42e+01,4.56e+01,
               4.72e+01,4.87e+01,5.03e+01,5.18e+01,5.34e+01,5.50e+01,5.65e+01,
               5.81e+01,5.96e+01,6.11e+01,6.25e+01,6.39e+01,6.52e+01,6.65e+01,
               6.77e+01,6.88e+01,6.98e+01,7.08e+01,7.17e+01,7.25e+01,7.32e+01,
               7.39e+01,7.45e+01,7.50e+01]
b8_cor[9,:] = [4.33e-05,8.18e-05,1.51e-04,2.72e-04,4.72e-04,7.94e-04,1.30e-03,
               2.06e-03,3.20e-03,4.87e-03,7.31e-03,1.08e-02,1.59e-02,2.30e-02,
               3.29e-02,4.66e-02,6.51e-02,8.97e-02,1.22e-01,1.62e-01,2.14e-01,
               2.77e-01,3.55e-01,4.50e-01,5.63e-01,6.97e-01,8.56e-01,1.04e+00,
               1.26e+00,1.51e+00,1.80e+00,2.12e+00,2.49e+00,2.91e+00,3.38e+00,
               3.89e+00,4.46e+00,5.08e+00,5.75e+00,6.48e+00,7.25e+00,8.08e+00,
               8.94e+00,9.85e+00,1.08e+01,1.18e+01,1.28e+01,1.38e+01,1.49e+01,
               1.59e+01,1.70e+01,1.81e+01,1.92e+01,2.03e+01,2.14e+01,2.24e+01,
               2.34e+01,2.43e+01,2.53e+01,2.62e+01,2.71e+01,2.79e+01,2.88e+01,
               2.98e+01,3.07e+01,3.17e+01,3.28e+01,3.39e+01,3.51e+01,3.63e+01,
               3.76e+01,3.89e+01,4.03e+01,4.18e+01,4.33e+01,4.48e+01,4.64e+01,
               4.80e+01,4.97e+01,5.13e+01,5.30e+01,5.47e+01,5.63e+01,5.80e+01,
               5.96e+01,6.12e+01,6.28e+01,6.43e+01,6.58e+01,6.72e+01,6.85e+01,
               6.98e+01,7.10e+01,7.21e+01,7.31e+01,7.40e+01,7.49e+01,7.57e+01,
               7.64e+01,7.70e+01,7.76e+01]
b8_cor[10,:] = [5.40e-05,1.01e-04,1.83e-04,3.26e-04,5.61e-04,9.36e-04,1.52e-03,
                2.40e-03,3.70e-03,5.61e-03,8.37e-03,1.23e-02,1.80e-02,2.59e-02,
                3.70e-02,5.21e-02,7.24e-02,9.94e-02,1.34e-01,1.79e-01,2.35e-01,
                3.04e-01,3.88e-01,4.90e-01,6.11e-01,7.56e-01,9.26e-01,1.12e+00,
                1.36e+00,1.62e+00,1.93e+00,2.27e+00,2.66e+00,3.10e+00,3.59e+00,
                4.13e+00,4.73e+00,5.37e+00,6.07e+00,6.83e+00,7.63e+00,8.48e+00,
                9.38e+00,1.03e+01,1.13e+01,1.23e+01,1.33e+01,1.43e+01,1.54e+01,
                1.65e+01,1.76e+01,1.87e+01,1.98e+01,2.09e+01,2.19e+01,2.30e+01,
                2.39e+01,2.49e+01,2.58e+01,2.66e+01,2.75e+01,2.83e+01,2.91e+01,
                3.00e+01,3.09e+01,3.19e+01,3.29e+01,3.40e+01,3.51e+01,3.63e+01,
                3.75e+01,3.88e+01,4.02e+01,4.16e+01,4.30e+01,4.45e+01,4.60e+01,
                4.76e+01,4.92e+01,5.08e+01,5.24e+01,5.40e+01,5.57e+01,5.73e+01,
                5.89e+01,6.04e+01,6.19e+01,6.34e+01,6.48e+01,6.62e+01,6.75e+01,
                6.87e+01,6.99e+01,7.10e+01,7.20e+01,7.29e+01,7.37e+01,7.45e+01,
                7.52e+01,7.58e+01,7.63e+01]
b8_cor[11,:] = [5.27e-05,9.82e-05,1.79e-04,3.19e-04,5.50e-04,9.19e-04,1.49e-03,
                2.36e-03,3.64e-03,5.52e-03,8.25e-03,1.22e-02,1.77e-02,2.56e-02,
                3.65e-02,5.15e-02,7.16e-02,9.83e-02,1.33e-01,1.77e-01,2.32e-01,
                3.01e-01,3.84e-01,4.85e-01,6.06e-01,7.49e-01,9.18e-01,1.12e+00,
                1.34e+00,1.61e+00,1.91e+00,2.26e+00,2.65e+00,3.08e+00,3.57e+00,
                4.11e+00,4.70e+00,5.34e+00,6.04e+00,6.79e+00,7.59e+00,8.44e+00,
                9.33e+00,1.03e+01,1.12e+01,1.22e+01,1.32e+01,1.43e+01,1.54e+01,
                1.64e+01,1.75e+01,1.86e+01,1.97e+01,2.08e+01,2.19e+01,2.29e+01,
                2.39e+01,2.48e+01,2.57e+01,2.66e+01,2.74e+01,2.83e+01,2.91e+01,
                3.00e+01,3.09e+01,3.19e+01,3.29e+01,3.40e+01,3.51e+01,3.63e+01,
                3.75e+01,3.88e+01,4.02e+01,4.16e+01,4.30e+01,4.45e+01,4.61e+01,
                4.77e+01,4.92e+01,5.09e+01,5.25e+01,5.41e+01,5.57e+01,5.73e+01,
                5.89e+01,6.05e+01,6.20e+01,6.35e+01,6.49e+01,6.63e+01,6.76e+01,
                6.89e+01,7.00e+01,7.11e+01,7.21e+01,7.30e+01,7.39e+01,7.46e+01,
                7.53e+01,7.59e+01,7.65e+01]
b8_cor[12,:] = [5.43e-05,1.01e-04,1.84e-04,3.27e-04,5.63e-04,9.39e-04,1.52e-03,
                2.40e-03,3.71e-03,5.63e-03,8.40e-03,1.24e-02,1.80e-02,2.60e-02,
                3.70e-02,5.22e-02,7.26e-02,9.96e-02,1.35e-01,1.79e-01,2.35e-01,
                3.04e-01,3.89e-01,4.91e-01,6.12e-01,7.57e-01,9.27e-01,1.13e+00,
                1.36e+00,1.62e+00,1.93e+00,2.28e+00,2.67e+00,3.11e+00,3.60e+00,
                4.14e+00,4.73e+00,5.38e+00,6.08e+00,6.84e+00,7.64e+00,8.49e+00,
                9.39e+00,1.03e+01,1.13e+01,1.23e+01,1.33e+01,1.44e+01,1.54e+01,
                1.65e+01,1.76e+01,1.87e+01,1.98e+01,2.09e+01,2.19e+01,2.30e+01,
                2.39e+01,2.49e+01,2.58e+01,2.66e+01,2.75e+01,2.83e+01,2.92e+01,
                3.00e+01,3.09e+01,3.19e+01,3.29e+01,3.40e+01,3.51e+01,3.63e+01,
                3.75e+01,3.88e+01,4.01e+01,4.16e+01,4.30e+01,4.45e+01,4.60e+01,
                4.76e+01,4.92e+01,5.08e+01,5.24e+01,5.40e+01,5.56e+01,5.72e+01,
                5.88e+01,6.04e+01,6.19e+01,6.34e+01,6.48e+01,6.62e+01,6.75e+01,
                6.87e+01,6.99e+01,7.09e+01,7.19e+01,7.29e+01,7.37e+01,7.45e+01,
                7.51e+01,7.57e+01,7.63e+01]
b8_cor[13,:] = [5.43e-05,1.01e-04,1.84e-04,3.27e-04,5.63e-04,9.39e-04,1.52e-03,
                2.40e-03,3.71e-03,5.63e-03,8.40e-03,1.24e-02,1.80e-02,2.60e-02,
                3.70e-02,5.22e-02,7.26e-02,9.96e-02,1.35e-01,1.79e-01,2.35e-01,
                3.04e-01,3.89e-01,4.91e-01,6.12e-01,7.57e-01,9.27e-01,1.13e+00,
                1.36e+00,1.62e+00,1.93e+00,2.28e+00,2.67e+00,3.11e+00,3.60e+00,
                4.14e+00,4.73e+00,5.38e+00,6.08e+00,6.84e+00,7.64e+00,8.49e+00,
                9.39e+00,1.03e+01,1.13e+01,1.23e+01,1.33e+01,1.44e+01,1.54e+01,
                1.65e+01,1.76e+01,1.87e+01,1.98e+01,2.09e+01,2.19e+01,2.30e+01,
                2.39e+01,2.49e+01,2.58e+01,2.66e+01,2.75e+01,2.83e+01,2.92e+01,
                3.00e+01,3.09e+01,3.19e+01,3.29e+01,3.40e+01,3.51e+01,3.63e+01,
                3.75e+01,3.88e+01,4.01e+01,4.16e+01,4.30e+01,4.45e+01,4.60e+01,
                4.76e+01,4.92e+01,5.08e+01,5.24e+01,5.40e+01,5.56e+01,5.72e+01,
                5.88e+01,6.04e+01,6.19e+01,6.34e+01,6.48e+01,6.62e+01,6.75e+01,
                6.87e+01,6.99e+01,7.09e+01,7.19e+01,7.29e+01,7.37e+01,7.45e+01,
                7.51e+01,7.57e+01,7.63e+01]
b8_cor[14,:] = [6.27e-05,1.15e-04,2.09e-04,3.68e-04,6.29e-04,1.05e-03,1.69e-03,
                2.65e-03,4.08e-03,6.16e-03,9.17e-03,1.35e-02,1.95e-02,2.81e-02,
                3.99e-02,5.61e-02,7.78e-02,1.06e-01,1.43e-01,1.91e-01,2.50e-01,
                3.23e-01,4.11e-01,5.18e-01,6.46e-01,7.98e-01,9.76e-01,1.18e+00,
                1.42e+00,1.70e+00,2.02e+00,2.38e+00,2.79e+00,3.24e+00,3.75e+00,
                4.30e+00,4.92e+00,5.58e+00,6.30e+00,7.07e+00,7.90e+00,8.77e+00,
                9.68e+00,1.06e+01,1.16e+01,1.26e+01,1.37e+01,1.47e+01,1.58e+01,
                1.69e+01,1.80e+01,1.91e+01,2.02e+01,2.13e+01,2.23e+01,2.34e+01,
                2.43e+01,2.52e+01,2.61e+01,2.69e+01,2.77e+01,2.85e+01,2.94e+01,
                3.02e+01,3.11e+01,3.20e+01,3.30e+01,3.40e+01,3.51e+01,3.62e+01,
                3.74e+01,3.87e+01,4.00e+01,4.14e+01,4.28e+01,4.43e+01,4.58e+01,
                4.73e+01,4.89e+01,5.05e+01,5.20e+01,5.36e+01,5.52e+01,5.68e+01,
                5.84e+01,5.99e+01,6.14e+01,6.28e+01,6.42e+01,6.56e+01,6.69e+01,
                6.81e+01,6.92e+01,7.03e+01,7.12e+01,7.21e+01,7.30e+01,7.37e+01,
                7.44e+01,7.50e+01,7.55e+01]

# Copy and paste ratio values into b8_pho from goes_get_chianti_em.pro
# assuming photospheric abundances.
b8_pho[0,:] = [3.87e-05,6.90e-05,1.21e-04,2.07e-04,3.47e-04,5.67e-04,9.06e-04,
               1.42e-03,2.17e-03,3.27e-03,4.86e-03,7.11e-03,1.03e-02,1.47e-02,
               2.07e-02,2.89e-02,3.98e-02,5.41e-02,7.24e-02,9.54e-02,1.24e-01,
               1.59e-01,2.01e-01,2.51e-01,3.10e-01,3.80e-01,4.61e-01,5.54e-01,
               6.62e-01,7.86e-01,9.27e-01,1.09e+00,1.27e+00,1.47e+00,1.69e+00,
               1.94e+00,2.21e+00,2.51e+00,2.83e+00,3.18e+00,3.56e+00,3.97e+00,
               4.40e+00,4.86e+00,5.35e+00,5.86e+00,6.40e+00,6.97e+00,7.56e+00,
               8.18e+00,8.83e+00,9.49e+00,1.02e+01,1.09e+01,1.16e+01,1.23e+01,
               1.31e+01,1.38e+01,1.46e+01,1.54e+01,1.61e+01,1.69e+01,1.77e+01,
               1.86e+01,1.95e+01,2.03e+01,2.13e+01,2.22e+01,2.32e+01,2.42e+01,
               2.53e+01,2.64e+01,2.75e+01,2.87e+01,2.99e+01,3.11e+01,3.24e+01,
               3.37e+01,3.50e+01,3.63e+01,3.77e+01,3.91e+01,4.04e+01,4.18e+01,
               4.32e+01,4.45e+01,4.59e+01,4.72e+01,4.86e+01,4.99e+01,5.12e+01,
               5.24e+01,5.37e+01,5.49e+01,5.60e+01,5.72e+01,5.82e+01,5.93e+01,
               6.03e+01,6.13e+01,6.22e+01]
b8_pho[1,:] = [3.98e-05,7.09e-05,1.24e-04,2.13e-04,3.57e-04,5.83e-04,9.31e-04,
               1.46e-03,2.23e-03,3.36e-03,4.99e-03,7.30e-03,1.06e-02,1.51e-02,
               2.13e-02,2.97e-02,4.09e-02,5.56e-02,7.44e-02,9.81e-02,1.27e-01,
               1.63e-01,2.07e-01,2.58e-01,3.19e-01,3.90e-01,4.73e-01,5.70e-01,
               6.81e-01,8.08e-01,9.53e-01,1.12e+00,1.30e+00,1.51e+00,1.74e+00,
               1.99e+00,2.27e+00,2.58e+00,2.91e+00,3.27e+00,3.66e+00,4.07e+00,
               4.52e+00,4.99e+00,5.49e+00,6.02e+00,6.58e+00,7.16e+00,7.77e+00,
               8.41e+00,9.07e+00,9.76e+00,1.05e+01,1.12e+01,1.19e+01,1.27e+01,
               1.34e+01,1.42e+01,1.50e+01,1.58e+01,1.66e+01,1.74e+01,1.82e+01,
               1.91e+01,2.00e+01,2.09e+01,2.19e+01,2.28e+01,2.39e+01,2.49e+01,
               2.60e+01,2.71e+01,2.83e+01,2.95e+01,3.07e+01,3.20e+01,3.33e+01,
               3.46e+01,3.60e+01,3.73e+01,3.87e+01,4.01e+01,4.15e+01,4.29e+01,
               4.44e+01,4.58e+01,4.72e+01,4.86e+01,4.99e+01,5.13e+01,5.26e+01,
               5.39e+01,5.52e+01,5.64e+01,5.76e+01,5.87e+01,5.98e+01,6.09e+01,
               6.20e+01,6.29e+01,6.39e+01]
b8_pho[2,:] = [3.98e-05,7.09e-05,1.24e-04,2.13e-04,3.57e-04,5.83e-04,9.31e-04,
               1.46e-03,2.23e-03,3.36e-03,4.99e-03,7.30e-03,1.06e-02,1.51e-02,
               2.13e-02,2.97e-02,4.09e-02,5.56e-02,7.44e-02,9.81e-02,1.27e-01,
               1.63e-01,2.07e-01,2.58e-01,3.19e-01,3.90e-01,4.73e-01,5.70e-01,
               6.81e-01,8.08e-01,9.53e-01,1.12e+00,1.30e+00,1.51e+00,1.74e+00,
               1.99e+00,2.27e+00,2.58e+00,2.91e+00,3.27e+00,3.66e+00,4.07e+00,
               4.52e+00,4.99e+00,5.49e+00,6.02e+00,6.58e+00,7.16e+00,7.77e+00,
               8.41e+00,9.07e+00,9.76e+00,1.05e+01,1.12e+01,1.19e+01,1.27e+01,
               1.34e+01,1.42e+01,1.50e+01,1.58e+01,1.66e+01,1.74e+01,1.82e+01,
               1.91e+01,2.00e+01,2.09e+01,2.19e+01,2.28e+01,2.39e+01,2.49e+01,
               2.60e+01,2.71e+01,2.83e+01,2.95e+01,3.07e+01,3.20e+01,3.33e+01,
               3.46e+01,3.60e+01,3.73e+01,3.87e+01,4.01e+01,4.15e+01,4.29e+01,
               4.44e+01,4.58e+01,4.72e+01,4.86e+01,4.99e+01,5.13e+01,5.26e+01,
               5.39e+01,5.52e+01,5.64e+01,5.76e+01,5.87e+01,5.98e+01,6.09e+01,
               6.20e+01,6.29e+01,6.39e+01]
b8_pho[3,:] = [3.47e-05,6.19e-05,1.08e-04,1.86e-04,3.11e-04,5.09e-04,8.12e-04,
               1.27e-03,1.95e-03,2.93e-03,4.35e-03,6.37e-03,9.22e-03,1.32e-02,
               1.86e-02,2.59e-02,3.57e-02,4.85e-02,6.49e-02,8.56e-02,1.11e-01,
               1.43e-01,1.80e-01,2.25e-01,2.78e-01,3.41e-01,4.13e-01,4.97e-01,
               5.94e-01,7.05e-01,8.32e-01,9.75e-01,1.14e+00,1.32e+00,1.52e+00,
               1.74e+00,1.98e+00,2.25e+00,2.54e+00,2.85e+00,3.19e+00,3.56e+00,
               3.94e+00,4.36e+00,4.79e+00,5.26e+00,5.74e+00,6.25e+00,6.78e+00,
               7.34e+00,7.92e+00,8.51e+00,9.13e+00,9.77e+00,1.04e+01,1.11e+01,
               1.17e+01,1.24e+01,1.31e+01,1.38e+01,1.45e+01,1.52e+01,1.59e+01,
               1.67e+01,1.74e+01,1.82e+01,1.91e+01,1.99e+01,2.08e+01,2.17e+01,
               2.27e+01,2.37e+01,2.47e+01,2.58e+01,2.68e+01,2.79e+01,2.91e+01,
               3.02e+01,3.14e+01,3.26e+01,3.38e+01,3.50e+01,3.63e+01,3.75e+01,
               3.87e+01,3.99e+01,4.12e+01,4.24e+01,4.36e+01,4.48e+01,4.59e+01,
               4.70e+01,4.81e+01,4.92e+01,5.03e+01,5.13e+01,5.22e+01,5.32e+01,
               5.41e+01,5.49e+01,5.58e+01]
b8_pho[4,:] = [3.27e-05,5.83e-05,1.02e-04,1.75e-04,2.93e-04,4.79e-04,7.65e-04,
               1.20e-03,1.83e-03,2.76e-03,4.10e-03,6.01e-03,8.68e-03,1.24e-02,
               1.75e-02,2.44e-02,3.37e-02,4.57e-02,6.12e-02,8.06e-02,1.05e-01,
               1.34e-01,1.70e-01,2.12e-01,2.62e-01,3.21e-01,3.89e-01,4.68e-01,
               5.60e-01,6.64e-01,7.84e-01,9.19e-01,1.07e+00,1.24e+00,1.43e+00,
               1.64e+00,1.87e+00,2.12e+00,2.39e+00,2.69e+00,3.01e+00,3.35e+00,
               3.72e+00,4.11e+00,4.52e+00,4.95e+00,5.41e+00,5.89e+00,6.39e+00,
               6.91e+00,7.46e+00,8.02e+00,8.61e+00,9.20e+00,9.81e+00,1.04e+01,
               1.11e+01,1.17e+01,1.23e+01,1.30e+01,1.36e+01,1.43e+01,1.50e+01,
               1.57e+01,1.64e+01,1.72e+01,1.80e+01,1.88e+01,1.96e+01,2.05e+01,
               2.14e+01,2.23e+01,2.33e+01,2.43e+01,2.53e+01,2.63e+01,2.74e+01,
               2.85e+01,2.96e+01,3.07e+01,3.19e+01,3.30e+01,3.42e+01,3.53e+01,
               3.65e+01,3.76e+01,3.88e+01,3.99e+01,4.11e+01,4.22e+01,4.32e+01,
               4.43e+01,4.54e+01,4.64e+01,4.73e+01,4.83e+01,4.92e+01,5.01e+01,
               5.09e+01,5.18e+01,5.25e+01]
b8_pho[5,:] = [3.36e-05,5.95e-05,1.04e-04,1.77e-04,2.95e-04,4.81e-04,7.66e-04,
               1.19e-03,1.83e-03,2.74e-03,4.06e-03,5.93e-03,8.55e-03,1.22e-02,
               1.72e-02,2.39e-02,3.29e-02,4.45e-02,5.95e-02,7.84e-02,1.02e-01,
               1.30e-01,1.64e-01,2.05e-01,2.53e-01,3.10e-01,3.75e-01,4.51e-01,
               5.38e-01,6.38e-01,7.52e-01,8.81e-01,1.03e+00,1.19e+00,1.37e+00,
               1.57e+00,1.79e+00,2.02e+00,2.28e+00,2.56e+00,2.87e+00,3.19e+00,
               3.53e+00,3.90e+00,4.29e+00,4.70e+00,5.13e+00,5.58e+00,6.05e+00,
               6.55e+00,7.06e+00,7.59e+00,8.14e+00,8.70e+00,9.27e+00,9.85e+00,
               1.04e+01,1.10e+01,1.16e+01,1.22e+01,1.28e+01,1.35e+01,1.41e+01,
               1.48e+01,1.54e+01,1.61e+01,1.69e+01,1.76e+01,1.84e+01,1.92e+01,
               2.00e+01,2.09e+01,2.18e+01,2.27e+01,2.36e+01,2.46e+01,2.56e+01,
               2.66e+01,2.76e+01,2.87e+01,2.97e+01,3.08e+01,3.19e+01,3.29e+01,
               3.40e+01,3.51e+01,3.61e+01,3.72e+01,3.82e+01,3.93e+01,4.03e+01,
               4.13e+01,4.22e+01,4.32e+01,4.41e+01,4.50e+01,4.58e+01,4.66e+01,
               4.74e+01,4.82e+01,4.89e+01]
b8_pho[6,:] = [2.83e-05,5.11e-05,9.08e-05,1.57e-04,2.66e-04,4.40e-04,7.08e-04,
               1.12e-03,1.72e-03,2.61e-03,3.91e-03,5.76e-03,8.38e-03,1.21e-02,
               1.71e-02,2.40e-02,3.33e-02,4.54e-02,6.11e-02,8.09e-02,1.06e-01,
               1.36e-01,1.73e-01,2.17e-01,2.69e-01,3.30e-01,4.01e-01,4.85e-01,
               5.81e-01,6.92e-01,8.19e-01,9.63e-01,1.13e+00,1.31e+00,1.51e+00,
               1.74e+00,1.99e+00,2.26e+00,2.56e+00,2.88e+00,3.23e+00,3.60e+00,
               4.01e+00,4.43e+00,4.89e+00,5.37e+00,5.88e+00,6.41e+00,6.97e+00,
               7.56e+00,8.17e+00,8.80e+00,9.46e+00,1.01e+01,1.08e+01,1.15e+01,
               1.22e+01,1.30e+01,1.37e+01,1.44e+01,1.52e+01,1.60e+01,1.68e+01,
               1.76e+01,1.84e+01,1.93e+01,2.02e+01,2.11e+01,2.21e+01,2.31e+01,
               2.41e+01,2.52e+01,2.63e+01,2.74e+01,2.86e+01,2.98e+01,3.10e+01,
               3.23e+01,3.35e+01,3.48e+01,3.61e+01,3.75e+01,3.88e+01,4.01e+01,
               4.15e+01,4.28e+01,4.41e+01,4.54e+01,4.67e+01,4.80e+01,4.93e+01,
               5.05e+01,5.17e+01,5.28e+01,5.40e+01,5.51e+01,5.61e+01,5.71e+01,
               5.81e+01,5.91e+01,6.00e+01]
b8_pho[7,:] = [4.95e-05,8.69e-05,1.50e-04,2.53e-04,4.18e-04,6.75e-04,1.07e-03,
               1.65e-03,2.51e-03,3.75e-03,5.52e-03,8.01e-03,1.15e-02,1.63e-02,
               2.29e-02,3.17e-02,4.34e-02,5.85e-02,7.79e-02,1.02e-01,1.32e-01,
               1.69e-01,2.13e-01,2.65e-01,3.26e-01,3.97e-01,4.80e-01,5.76e-01,
               6.86e-01,8.12e-01,9.54e-01,1.12e+00,1.30e+00,1.50e+00,1.72e+00,
               1.97e+00,2.24e+00,2.53e+00,2.85e+00,3.19e+00,3.56e+00,3.96e+00,
               4.38e+00,4.83e+00,5.30e+00,5.80e+00,6.32e+00,6.87e+00,7.44e+00,
               8.03e+00,8.65e+00,9.28e+00,9.94e+00,1.06e+01,1.13e+01,1.20e+01,
               1.27e+01,1.34e+01,1.41e+01,1.48e+01,1.55e+01,1.62e+01,1.70e+01,
               1.78e+01,1.86e+01,1.94e+01,2.02e+01,2.11e+01,2.20e+01,2.30e+01,
               2.39e+01,2.50e+01,2.60e+01,2.71e+01,2.82e+01,2.93e+01,3.05e+01,
               3.17e+01,3.29e+01,3.41e+01,3.53e+01,3.66e+01,3.79e+01,3.91e+01,
               4.04e+01,4.16e+01,4.29e+01,4.41e+01,4.54e+01,4.66e+01,4.78e+01,
               4.89e+01,5.00e+01,5.11e+01,5.22e+01,5.32e+01,5.42e+01,5.52e+01,
               5.61e+01,5.70e+01,5.79e+01]
b8_pho[8,:] = [5.24e-05,9.16e-05,1.58e-04,2.66e-04,4.38e-04,7.06e-04,1.11e-03,
               1.72e-03,2.61e-03,3.89e-03,5.72e-03,8.30e-03,1.19e-02,1.68e-02,
               2.36e-02,3.26e-02,4.46e-02,6.01e-02,8.00e-02,1.05e-01,1.36e-01,
               1.73e-01,2.17e-01,2.70e-01,3.33e-01,4.05e-01,4.89e-01,5.87e-01,
               6.98e-01,8.26e-01,9.70e-01,1.13e+00,1.32e+00,1.52e+00,1.75e+00,
               1.99e+00,2.27e+00,2.56e+00,2.88e+00,3.23e+00,3.60e+00,4.00e+00,
               4.42e+00,4.87e+00,5.35e+00,5.84e+00,6.37e+00,6.92e+00,7.49e+00,
               8.08e+00,8.70e+00,9.33e+00,9.99e+00,1.07e+01,1.13e+01,1.20e+01,
               1.27e+01,1.34e+01,1.41e+01,1.48e+01,1.55e+01,1.63e+01,1.70e+01,
               1.78e+01,1.86e+01,1.94e+01,2.02e+01,2.11e+01,2.20e+01,2.29e+01,
               2.39e+01,2.49e+01,2.60e+01,2.70e+01,2.81e+01,2.93e+01,3.04e+01,
               3.16e+01,3.28e+01,3.40e+01,3.53e+01,3.65e+01,3.78e+01,3.90e+01,
               4.03e+01,4.15e+01,4.28e+01,4.40e+01,4.52e+01,4.64e+01,4.76e+01,
               4.88e+01,4.99e+01,5.10e+01,5.20e+01,5.31e+01,5.41e+01,5.50e+01,
               5.59e+01,5.68e+01,5.76e+01]
b8_pho[9,:] = [3.21e-05,5.78e-05,1.02e-04,1.76e-04,2.96e-04,4.86e-04,7.79e-04,
               1.22e-03,1.88e-03,2.85e-03,4.24e-03,6.23e-03,9.04e-03,1.30e-02,
               1.84e-02,2.57e-02,3.55e-02,4.83e-02,6.48e-02,8.56e-02,1.12e-01,
               1.43e-01,1.81e-01,2.27e-01,2.81e-01,3.45e-01,4.19e-01,5.05e-01,
               6.05e-01,7.19e-01,8.49e-01,9.97e-01,1.16e+00,1.35e+00,1.56e+00,
               1.79e+00,2.04e+00,2.32e+00,2.62e+00,2.95e+00,3.30e+00,3.69e+00,
               4.09e+00,4.53e+00,4.98e+00,5.47e+00,5.98e+00,6.52e+00,7.08e+00,
               7.67e+00,8.28e+00,8.91e+00,9.57e+00,1.02e+01,1.09e+01,1.16e+01,
               1.23e+01,1.31e+01,1.38e+01,1.45e+01,1.53e+01,1.60e+01,1.68e+01,
               1.76e+01,1.85e+01,1.93e+01,2.02e+01,2.11e+01,2.21e+01,2.31e+01,
               2.41e+01,2.51e+01,2.62e+01,2.74e+01,2.85e+01,2.97e+01,3.09e+01,
               3.21e+01,3.34e+01,3.47e+01,3.60e+01,3.73e+01,3.86e+01,3.99e+01,
               4.12e+01,4.25e+01,4.38e+01,4.51e+01,4.64e+01,4.77e+01,4.89e+01,
               5.01e+01,5.13e+01,5.25e+01,5.36e+01,5.47e+01,5.57e+01,5.67e+01,
               5.77e+01,5.86e+01,5.95e+01]
b8_pho[10,:] = [4.08e-05,7.23e-05,1.26e-04,2.15e-04,3.58e-04,5.83e-04,9.27e-04,
                1.44e-03,2.21e-03,3.32e-03,4.91e-03,7.16e-03,1.03e-02,1.47e-02,
                2.07e-02,2.89e-02,3.97e-02,5.37e-02,7.18e-02,9.45e-02,1.23e-01,
                1.57e-01,1.98e-01,2.47e-01,3.05e-01,3.73e-01,4.52e-01,5.43e-01,
                6.49e-01,7.69e-01,9.06e-01,1.06e+00,1.24e+00,1.43e+00,1.65e+00,
                1.89e+00,2.15e+00,2.44e+00,2.75e+00,3.08e+00,3.45e+00,3.84e+00,
                4.25e+00,4.69e+00,5.16e+00,5.65e+00,6.17e+00,6.71e+00,7.28e+00,
                7.87e+00,8.48e+00,9.12e+00,9.77e+00,1.04e+01,1.11e+01,1.18e+01,
                1.25e+01,1.32e+01,1.39e+01,1.47e+01,1.54e+01,1.61e+01,1.69e+01,
                1.77e+01,1.85e+01,1.94e+01,2.02e+01,2.11e+01,2.20e+01,2.30e+01,
                2.40e+01,2.50e+01,2.61e+01,2.72e+01,2.83e+01,2.95e+01,3.07e+01,
                3.19e+01,3.31e+01,3.44e+01,3.56e+01,3.69e+01,3.82e+01,3.95e+01,
                4.08e+01,4.21e+01,4.33e+01,4.46e+01,4.58e+01,4.71e+01,4.83e+01,
                4.95e+01,5.06e+01,5.17e+01,5.28e+01,5.39e+01,5.49e+01,5.59e+01,
                5.68e+01,5.77e+01,5.86e+01]
b8_pho[11,:] = [3.97e-05,7.05e-05,1.23e-04,2.10e-04,3.50e-04,5.71e-04,9.09e-04,
                1.42e-03,2.17e-03,3.26e-03,4.83e-03,7.05e-03,1.02e-02,1.45e-02,
                2.05e-02,2.85e-02,3.92e-02,5.31e-02,7.10e-02,9.35e-02,1.21e-01,
                1.55e-01,1.96e-01,2.45e-01,3.03e-01,3.70e-01,4.48e-01,5.39e-01,
                6.44e-01,7.63e-01,9.00e-01,1.05e+00,1.23e+00,1.42e+00,1.64e+00,
                1.88e+00,2.14e+00,2.42e+00,2.73e+00,3.07e+00,3.43e+00,3.82e+00,
                4.23e+00,4.67e+00,5.14e+00,5.63e+00,6.15e+00,6.69e+00,7.26e+00,
                7.85e+00,8.46e+00,9.10e+00,9.75e+00,1.04e+01,1.11e+01,1.18e+01,
                1.25e+01,1.32e+01,1.39e+01,1.46e+01,1.54e+01,1.61e+01,1.69e+01,
                1.77e+01,1.85e+01,1.93e+01,2.02e+01,2.11e+01,2.21e+01,2.30e+01,
                2.40e+01,2.51e+01,2.61e+01,2.72e+01,2.84e+01,2.95e+01,3.07e+01,
                3.19e+01,3.31e+01,3.44e+01,3.57e+01,3.70e+01,3.82e+01,3.95e+01,
                4.08e+01,4.21e+01,4.34e+01,4.47e+01,4.59e+01,4.71e+01,4.84e+01,
                4.95e+01,5.07e+01,5.18e+01,5.29e+01,5.40e+01,5.50e+01,5.60e+01,
                5.69e+01,5.78e+01,5.87e+01]
b8_pho[12,:] = [4.10e-05,7.26e-05,1.26e-04,2.16e-04,3.59e-04,5.85e-04,9.30e-04,
                1.45e-03,2.21e-03,3.33e-03,4.92e-03,7.18e-03,1.04e-02,1.48e-02,
                2.08e-02,2.89e-02,3.97e-02,5.39e-02,7.20e-02,9.47e-02,1.23e-01,
                1.57e-01,1.99e-01,2.48e-01,3.06e-01,3.74e-01,4.53e-01,5.44e-01,
                6.50e-01,7.70e-01,9.08e-01,1.06e+00,1.24e+00,1.43e+00,1.65e+00,
                1.89e+00,2.15e+00,2.44e+00,2.75e+00,3.09e+00,3.45e+00,3.84e+00,
                4.26e+00,4.70e+00,5.16e+00,5.66e+00,6.17e+00,6.72e+00,7.28e+00,
                7.87e+00,8.49e+00,9.12e+00,9.78e+00,1.05e+01,1.11e+01,1.18e+01,
                1.25e+01,1.32e+01,1.39e+01,1.47e+01,1.54e+01,1.61e+01,1.69e+01,
                1.77e+01,1.85e+01,1.94e+01,2.02e+01,2.11e+01,2.20e+01,2.30e+01,
                2.40e+01,2.50e+01,2.61e+01,2.72e+01,2.83e+01,2.95e+01,3.07e+01,
                3.19e+01,3.31e+01,3.44e+01,3.56e+01,3.69e+01,3.82e+01,3.95e+01,
                4.08e+01,4.20e+01,4.33e+01,4.46e+01,4.58e+01,4.71e+01,4.83e+01,
                4.95e+01,5.06e+01,5.17e+01,5.28e+01,5.39e+01,5.49e+01,5.59e+01,
                5.68e+01,5.77e+01,5.86e+01]
b8_pho[13,:] = [4.10e-05,7.26e-05,1.26e-04,2.16e-04,3.59e-04,5.85e-04,9.30e-04,
                1.45e-03,2.21e-03,3.33e-03,4.92e-03,7.18e-03,1.04e-02,1.48e-02,
                2.08e-02,2.89e-02,3.97e-02,5.39e-02,7.20e-02,9.47e-02,1.23e-01,
                1.57e-01,1.99e-01,2.48e-01,3.06e-01,3.74e-01,4.53e-01,5.44e-01,
                6.50e-01,7.70e-01,9.08e-01,1.06e+00,1.24e+00,1.43e+00,1.65e+00,
                1.89e+00,2.15e+00,2.44e+00,2.75e+00,3.09e+00,3.45e+00,3.84e+00,
                4.26e+00,4.70e+00,5.16e+00,5.66e+00,6.17e+00,6.72e+00,7.28e+00,
                7.87e+00,8.49e+00,9.12e+00,9.78e+00,1.05e+01,1.11e+01,1.18e+01,
                1.25e+01,1.32e+01,1.39e+01,1.47e+01,1.54e+01,1.61e+01,1.69e+01,
                1.77e+01,1.85e+01,1.94e+01,2.02e+01,2.11e+01,2.20e+01,2.30e+01,
                2.40e+01,2.50e+01,2.61e+01,2.72e+01,2.83e+01,2.95e+01,3.07e+01,
                3.19e+01,3.31e+01,3.44e+01,3.56e+01,3.69e+01,3.82e+01,3.95e+01,
                4.08e+01,4.20e+01,4.33e+01,4.46e+01,4.58e+01,4.71e+01,4.83e+01,
                4.95e+01,5.06e+01,5.17e+01,5.28e+01,5.39e+01,5.49e+01,5.59e+01,
                5.68e+01,5.77e+01,5.86e+01]
b8_pho[14,:] = [4.78e-05,8.41e-05,1.45e-04,2.46e-04,4.07e-04,6.58e-04,1.04e-03,
                1.61e-03,2.46e-03,3.67e-03,5.41e-03,7.87e-03,1.13e-02,1.60e-02,
                2.25e-02,3.12e-02,4.27e-02,5.77e-02,7.69e-02,1.01e-01,1.31e-01,
                1.67e-01,2.10e-01,2.62e-01,3.23e-01,3.93e-01,4.76e-01,5.71e-01,
                6.80e-01,8.05e-01,9.47e-01,1.11e+00,1.29e+00,1.49e+00,1.71e+00,
                1.96e+00,2.22e+00,2.52e+00,2.84e+00,3.18e+00,3.55e+00,3.94e+00,
                4.36e+00,4.81e+00,5.28e+00,5.78e+00,6.30e+00,6.84e+00,7.41e+00,
                8.01e+00,8.62e+00,9.26e+00,9.92e+00,1.06e+01,1.13e+01,1.20e+01,
                1.27e+01,1.34e+01,1.41e+01,1.48e+01,1.55e+01,1.62e+01,1.70e+01,
                1.78e+01,1.86e+01,1.94e+01,2.02e+01,2.11e+01,2.20e+01,2.30e+01,
                2.40e+01,2.50e+01,2.60e+01,2.71e+01,2.82e+01,2.94e+01,3.05e+01,
                3.17e+01,3.29e+01,3.42e+01,3.54e+01,3.67e+01,3.79e+01,3.92e+01,
                4.05e+01,4.17e+01,4.30e+01,4.42e+01,4.55e+01,4.67e+01,4.79e+01,
                4.90e+01,5.02e+01,5.13e+01,5.23e+01,5.34e+01,5.44e+01,5.53e+01,
                5.63e+01,5.72e+01,5.80e+01]

# Enter values into csv files.
# Create string list of each row then enter it into csv file
# Firstly csv file for coronal abundances using b8_cor array.
with open('goes_chianti_em_cor.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=';')
    # First create header containing column names and read into csv file
    header = ['longfluxGOES'+str(i) for i in range(1,NUMSATS+1)]
    header.insert(0, "log10temp")
    header.append("date_taken_ssw")
    csvwriter.writerow(header)
    # Write in data row by row.
    row = ["{:1.2e}".format(number) for number in b8_cor[:,0]]
    row.insert(0, log10temp[0])
    row.append(DATETAKEN)
    csvwriter.writerow(row)
    for i in range(1, len(log10temp)):
        row = ["{:1.2e}".format(number) for number in b8_cor[:,i]]
        row.insert(0, log10temp[i])
        csvwriter.writerow(row)

# Next write csv file for photospheric abundances using b8_pho array.
with open('goes_chianti_em_pho.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=';')
    # First create header containing column names and read into csv file
    header = ['longfluxGOES'+str(i) for i in range(1,NUMSATS+1)]
    header.insert(0, "log10temp_MK")
    header.append("date_taken_ssw")
    csvwriter.writerow(header)
    # Write in data row by row.
    row = ["{:1.2e}".format(number) for number in b8_cor[:,0]]
    row.insert(0, log10temp[0])
    row.append(DATETAKEN)
    csvwriter.writerow(row)
    for i in range(1, len(log10temp)):
        row = ["{:1.2e}".format(number) for number in b8_pho[:,i]]
        row.insert(0, log10temp[i])
        csvwriter.writerow(row)