import numpy as np
import scipy.interpolate as sci
import matplotlib.pyplot as plt

from anapymods3.plots import get_binmids


exp = np.load("data/IC86_I_data.npy")

time = exp["timeMJD"]
sinDec = np.sin(exp["dec"])

# 5 day bins in time, arbitrary 50 bins in sinDec
time_bins = np.linspace(np.amin(time), np.amax(time), 73 + 1)
sinDec_bins = np.linspace(-1, 1, 50)

h, b = np.histogram2d(time, sinDec, bins=[time_bins, sinDec_bins])

m = get_binmids(b)

# Make grid for x, y points in 2D plane
# xx, yy = map(np.ravel)

spl = sci.RectBivariateSpline(m[0], m[1], z=h, s=)
