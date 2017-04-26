"""
Maje a 'true' integration over 2 dims of the KDE to obtain a marginalized
1D distribtuion (here in logE).

Use multiprocessing, because it takes time on a single processor.
"""

import numpy as np
import scipy.integrate as scint
import sklearn.neighbors as skn
from multiprocessing import Pool

exp = np.load("./data/IC86_I_data.npy")

kde = skn.KernelDensity(bandwidth=0.1, kernel="gaussian", rtol=1e-8)

# KDE sample must be cut in sigma before fitting, similar to range in hist
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


# Now the integration over dec and sigma
# Choose axis in which the PDF is differential
xax = 0  # logE
xrng = np.array([2, 7]) * fac_logE

# Integration ranges
i1rng = np.array([-np.pi / 2., np.pi / 2.]) * fac_dec
i2rng = np.array([0, 5]) * fac_sigma

# Create the integration grid on the bin mids
nbinsx = 1000
xbins = np.linspace(2, 10, nbinsx + 1)
x = 0.5 * (xbins[:-1] + xbins[1:])


def pdf(y, x, *args):
    # y must be the first argument, x the second for scint.dblquad
    # This must match the defintion of the borders in the dblquad call
    xgridpt = args[0]
    pt = [xgridpt, x, y]
    pt = np.array(pt)[np.newaxis, :]

    return np.exp(kde.score_samples(X=pt))


# Double integral over two remaining axes
def integrate(args):
    pdf, i2rng, i2rng, xi = args
    intgrl = scint.dblquad(pdf, i1rng[0], i1rng[1],
                           lambda x: i2rng[0], lambda x: i2rng[1],
                           args=(xi,), epsrel=1e-3)
    return intgrl[0]


# Create param list for multiprocessing
integrals = np.zeros_like(x, dtype=np.float)
params = [[pdf, i1rng, i2rng, xi] for xi in x]

try:
    pool = Pool(30)
    integrals = pool.map(integrate, params)
finally:
    pool.close()
    pool.join()

# Store binmids and corresponding integral
grid_and_vals = np.vstack((x, integrals))
np.save(arr=grid_and_vals, file="./bins_and_vals.npy")
