# coding: utf-8

"""
Put the interesting parts of the CV result in a JSON file, so it is independent
of the python version used to pickle the full model selector object.

The pickled files are using python 3 so this script only runs in python 3.
"""

import pickle
import json

p1 = "./CV10_glob_bw_alpha_EXP_IC86I_CUT_sig.ll.20_PARS_diag_True_pass2"
p2 = "./CV10_glob_bw_EXP_IC86I_CUT_sig.ll.90_PARS_diag_True_alpha_0.5_pass2"

ms1 = pickle.load(open(p1 + ".pickle", "rb"))
ms2 = pickle.load(open(p2 + ".pickle", "rb"))

# Get sklearn (0.18.1) model selector objects
kde1 = ms1.best_estimator_
kde2 = ms2.best_estimator_

# Get best params and cached KDE values for both estimators
bp1 = kde1.get_params()
bp1["kde_vals"] = list(kde1._kde_values)
bp1["kde_X_std"] = [list(Xi) for Xi in kde1._std_X]
bp1["kde_X_mean"] = list(kde1.mean)
bp1["kde_X_cov"] = [list(Xi) for Xi in kde1.cov]

bp2 = kde2.get_params()
bp2["kde_vals"] = list(kde2._kde_values)
bp2["kde_X_std"] = [list(Xi) for Xi in kde2._std_X]
bp2["kde_X_mean"] = list(kde2.mean)
bp2["kde_X_cov"] = [list(Xi) for Xi in kde2.cov]

# Dump dict to JSON
json.dump(bp1, open(p1 + ".json", "w"), indent=2)
json.dump(bp2, open(p2 + ".json", "w"), indent=2)
