"""
Maje a 'true' integration over 1 dim of the KDE to obtain a marginalized
2D distribtuion (here in logE, sindec) over a 2D grid.

Use multiprocessing, because it takes time on a single processor.
"""

import numpy as np
import scipy.integrate as scint
import pickle
from multiprocessing import Pool
import sklearn.neighbors as skn

f = "KDE_model_selector_20_exp_IC86_I_followup_2nd_pass.pickle"
fname = "/home/tmenne/scripts/kde_CV/kde_cv/" + f
with open(fname, "rb") as f:
    model_selector = pickle.load(f)

kde = model_selector.best_estimator_
bw = model_selector.best_params_["bandwidth"]
print("Best bandwidth : {:.3f}".format(bw))

# We maybe just want to stick with the slightly overfitting kernel to
# be as close as possible to data
OVERFIT = True
if OVERFIT:
    bw = 0.075
    kde = skn.KernelDensity(bandwidth=bw, kernel="gaussian", rtol=1e-8)
print("Used bandwidth : {:.3f}".format(bw))

# KDE sample must be cut in sigma before fitting, similar to range in hist
exp = np.load("data/IC86_I_data.npy")
_exp = exp[exp["sigma"] <= np.deg2rad(5)]

fac_logE = 1.5
fac_dec = 2.5
fac_sigma = 2.

_logE = fac_logE * _exp["logE"]
_sigma = fac_sigma * np.rad2deg(_exp["sigma"])
_dec = fac_dec * _exp["dec"]

kde_sample = np.vstack((_logE, _dec, _sigma)).T

# Fit KDE best model to sample
kde.fit(kde_sample)


# We need to integrate on a sinDec grid, but KDE is in dec and scaled...
# First get the binning from MC to build the ratio with the same binning
mc = np.load("data/IC86_I_mc.npy")

# Make 2D hist from MC, use binning for the gridpoints of the integral
mc_sindec = np.sin(mc["dec"])
mc_logE = mc["logE"]
bins = [50, 50]
range = np.array([[1, 10], [-1, 1]])
mc_h, bx, by = np.histogram2d(mc_logE, mc_sindec, bins=bins, range=range,
                              normed=True)


# Integration range in sigma, after 10° not much contribution left
irng = np.array([0, 10]) * fac_sigma

# Create the integration grid on the bin mids in scales KDE space
x = 0.5 * (bx[:-1] + bx[1:]) * fac_logE
y = 0.5 * (by[:-1] + by[1:])
y = np.arcsin(y) * fac_dec  # Scale after convertion to dec
xx, yy = map(np.ravel, np.meshgrid(x, y))

# Scans x line by x line with incresing y
grid_pts = np.vstack((xx, yy)).T


def pdf(x, *args):
    # Make single point to evaluate the KDE at
    xgridpt, ygridpt = args
    pt = [xgridpt, ygridpt, x]
    pt = np.array(pt)[np.newaxis, :]

    return np.exp(kde.score_samples(X=pt))


def integrate(args):
    pdf, irng, gp = args
    intgrl = scint.quad(pdf, irng[0], irng[1], args=(gp[0], gp[1]))
    return intgrl[0]


# Create param list for multiprocessing
integrals = np.zeros_like(xx, dtype=np.float)
params = [[pdf, irng, gp] for gp in grid_pts]

try:
    pool = Pool(30)
    integrals = pool.map(integrate, params)
finally:
    pool.close()
    pool.join()

# Save original logE, sinDec bins and integral values (flattened)
np.save(arr=integrals, file="./logE_sinDec_int_50x50.npy")
b = np.vstack((bx, by))
np.save(arr=b, file="./logE_sinDec_bins_50x50.npy")
