# coding: utf8

"""
Cross validated Kernel Densitiy Estimation bandwidth grid search using sklearn.

The KDE is fitted to a 3D pdf derived from data to be able to draw non-descrete
events to inject as background in a time dependent point source search.

We use an adaptive kernel bandwidth algorithm (original implementation from
S. Schoenen and L. Raedel, RTWH Aachen) and get the optimal paramters from a
10-fold CV in `glob_bw` and `alpha`. Additionally, we check if a correlated or
non-correlated covariance matrix as a kernel works best.
"""

import numpy as np
import sklearn.model_selection as skms
import anapymods3.stats.KDE as KDE
import pickle


# Load data and cut in sigma to remove extremely bad reconstructed tracks.
# These are modeled with the tails coming from the adaptive kernels in sparse
# regions anyway.
exp = np.load("/Users/tmenne/git/misc/time/data/IC86_I_data.npy")

# ndata = len(exp)
# idx = np.random.choice(np.arange(ndata), replace=False, size=int(ndata / 10))
# logE = exp["logE"][idx]
# dec = exp["dec"][idx]
# sigma = exp["sigma"][idx]

logE = exp["logE"]
dec = exp["dec"]
sigma = exp["sigma"]

# Cut sigma (almost no stats after 20°, look at hist)
sig_max_deg = 20
cut = (sigma <= np.deg2rad(sig_max_deg))
logE = logE[cut]
dec = dec[cut]
sigma = sigma[cut]
sample = np.vstack((logE, dec, sigma)).T

# Make a 10-fold CV, with a 10 x 10 x 2 grid (= 2000 trials)
cv = 10
par_grid = {"glob_bw": np.linspace(0.01, 0.1, 10),
            "alpha": np.linspace(0, 1, 10), "diag_cov": [True, False]}

estimator = KDE.GaussianKDE()
selector = skms.GridSearchCV(estimator=estimator,
                  param_grid=par_grid,
                  n_jobs=40,
                  cv=cv)


fname = "./awKDE_selector_{:d}_exp_IC86_I.pickle".format(cv)
with open(fname, "wb") as f:
    pickle.dump(file=f, obj=selector)

# Get best bandwidth
print(selector.best_params_)
