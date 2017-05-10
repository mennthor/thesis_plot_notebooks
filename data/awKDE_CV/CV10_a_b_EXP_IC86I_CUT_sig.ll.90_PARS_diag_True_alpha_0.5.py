# coding: utf8

"""
Cross validated Kernel Densitiy Estimation bandwidth grid search using sklearn.

The KDE is fitted to a 3D pdf derived from data to be able to draw non-descrete
events to inject as background in a time dependent point source search.

An adaptive kernel bandwidth algorithm (original implementation from
S. Schoenen and L. Raedel, RTWH Aachen) to adapt to different data scales.
"""

import numpy as np
import sklearn.model_selection as skms
import anapymods3.stats.KDE as KDE
import pickle


# Load data
exp = np.load("/home/tmenne/scripts/data/IC86_I_data.npy")

logE = exp["logE"]
dec = exp["dec"]
sigma = exp["sigma"]

# Cut sigma (almost no stats after 90°, look at hist)
sig_max_deg = 90
cut = (sigma <= np.deg2rad(sig_max_deg))
logE = logE[cut]
dec = dec[cut]
sigma = sigma[cut]
sample = np.vstack((logE, dec, sigma)).T

# Make a 10-fold CV in glob_bw only, as suggested in paper
cv = 10
# Use scaling only, non-correlated kernel
diag = True
alpha = 0.5
estimator = KDE.GaussianKDE(alpha=0.5, diag_cov=True, max_gb=1.)
par_grid = {"glob_bw": np.arange(0.01, 0.1, 0.01)}

selector = skms.GridSearchCV(estimator=estimator,
                             param_grid=par_grid,
                             n_jobs=30,
                             cv=cv)

selector.fit(sample)

# It must be somehow possible to that info NOT in the filename but as metadata
# in the file itself...
parstr = "_".join(par_grid.keys())
fname = "./CV{:d}_{}_EXP_IC86I_CUT_sig.ll.{:d}_PARS_diag_{}_alpha_{:.1f}.pickle".format(
        int(cv), parstr, int(sig_max_deg), diag, alpha)
with open(fname, "wb") as f:
    pickle.dump(file=f, obj=selector)

# Get best bandwidth
print("glob_bw:\n",par_grid["glob_bw"])
print(selector.best_params_)

