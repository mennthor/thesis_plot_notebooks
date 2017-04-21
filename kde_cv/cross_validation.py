# coding: utf8

"""
Cross validated Kernel Densitiy Estimation bandwidth grid search using sklearn.

The KDE is used to draw event from the BG distribution.
The scale factors are guessed to make the feature scales in each dimension
approximately equal, because we have only a symmetric and global kernel.
These scales have always to be used with the best bandwidth estimate.

Used in 3 passes with a 20-fold VC each to narrow in on the best bandwidth:
1. Coarse    : bandwidth [0.01, 3.] in 0.5 steps.     Best: 0.51
2. Finer     : bandwidth [0.01, 1.] in 0.01 steps.    Best: 0.11
3. Very Fine : bandwidth [0.10, 0.12] in 0.001 steps. Best: 0.114
"""

from __future__ import print_function, division
import numpy as np
import sklearn.neighbors as skn
import sklearn.model_selection as skms
import pickle


# Load data and scale to match global feature scale because we have a
# symmetric kernel.
exp = np.load("data/IC86_I_data.npy")

# Remove sigma > 5° degree to throw away outliers.
# Tail is then a non-peaked gaussian tail which is OK
m = exp["sigma"] <= np.deg2rad(5)
exp = exp[m]

# ###########################################################################
# Scale factors to compensate for symmetric, global kernel
# These have to be taken into account again, when sampling from the KDE
fac_logE = 1.5
fac_dec = 2.5
fac_sigma = 2.
# ###########################################################################

logE = fac_logE * exp["logE"]
sigma = fac_sigma * np.rad2deg(exp["sigma"])
dec = fac_dec * exp["dec"]

sample = np.vstack((
    fac_logE * exp["logE"],
    fac_dec * exp["dec"],  # Normal space to have no hard cuts at the edges
    fac_sigma * np.rad2deg(exp["sigma"])  # In deg to match scale
)).T

# Optimize bandwidth in a cross validation.
kde_estimator = skn.KernelDensity(kernel="gaussian", rtol=1e-6)

# Scan grid. See comment on top on parameter ranges
SCAN = "followup_2nd_pass"
start = 0.1
step = 0.001
stop = 0.12 + step

bandwidths = np.arange(start, stop, step)
ncv = 20
param_grid = {"bandwidth": bandwidths}

model_selector = skms.GridSearchCV(
    estimator=kde_estimator,
    cv=ncv,
    param_grid=param_grid,
    n_jobs=40,
)

# Fit again
model_selector.fit(sample)

fname = "./KDE_model_selector_{:d}_exp_IC86_I_{:s}.pickle".format(ncv, SCAN)
with open(fname, "wb") as f:
    pickle.dump(file=f, obj=model_selector)

# Get best bandwidth
best_bandwidth = model_selector.best_params_["bandwidth"]

print("Tested bandwidths : {}".format(bandwidths))
print("Best bandwidth    : {}".format(best_bandwidth))
