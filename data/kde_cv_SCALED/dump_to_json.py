# coding: utf-8

"""
Put the interesting parts of the CV result in a JSON file, so it is independent
of the python version used to pickle the full model selector object.

The pickled files are using python 3 so this script only runs in python 3.
"""

import pickle
import json

p = "./KDE_model_selector_20_exp_IC86_I_followup_2nd_pass"

ms = pickle.load(open(p + ".pickle", "rb"))

# Get sklearn (0.18.1) model selector objects
kde = ms.best_estimator_

# Get best params and dump dict to JSON
bp = kde.get_params()
json.dump(bp, open(p + ".json", "w"), indent=2)
