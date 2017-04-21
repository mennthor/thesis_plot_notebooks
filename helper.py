"""
Some helper functions to keep the notebook clean.
More important functions are kept in the notebook.
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mpldates
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

import scipy.interpolate as sci
import scipy.optimize as sco
import scipy.stats as scs

import json
import datetime
import pickle
from astropy.time import Time as astrotime

from corner_hist import corner_hist
from anapymods3.plots.general import split_axis, get_binmids, hist_marginalize


def load_data():
    """
    Load some local data/MC from IC86-I from epinat (pullcorrected)
    Returns data, MC and livetime. Data, MC are in recarray format.
    """
    exp = np.load("data/IC86_I_data.npy")
    mc = np.load("data/IC86_I_mc.npy")
    # Use the officially stated livetime, not the ones from below
    livetime = 332.61
    return exp, mc, livetime


def get_run_list():
    """
    Generate list of dict from iclive run list in json format.
    Returns a list of dicts, where each item is a dict for a single run.
    """
    # Grab from json
    jsonFile = open('data/ic86-i-goodrunlist.json', 'r')
    grlist = json.load(jsonFile)
    jsonFile.close()

    # This is a list of dicts (one dict per run)
    runs = grlist["runs"]

    return runs


def get_run_dict(runs):
    """
    Convert the run list of dicts to a dict of lists.
    Returns a single dict, where each stat is the key to a list with the stats
    for each run.
    """
    # This is a dict of arrays (all run values in an array per keyword)
    run_dict = dict(zip(runs[0].keys(), zip(*[r.values() for r in runs])))
    for k in run_dict.keys():
        run_dict[k] = np.array(run_dict[k])

    return run_dict


def get_good_runs(run_dict):
    """
    Filter the runs as stated on jfeintzeigs wiki page to get the used runs.
    Returns the used run information in a recarray, the start_mjd, stop_mjd for
    each run in MJD format and the total livetime from this runlist.
    """
    # Now compile runs as stated on jfeintzeigs page

    # Transform livetimes to MJD floats
    start_mjd = astrotime(run_dict["good_tstart"]).mjd
    stop_mjd = astrotime(run_dict["good_tstop"]).mjd

    # Create recarry to apply mask, only keep start, stop and runID
    dtype = [("start_mjd", np.float), ("stop_mjd", np.float),
             ("runID", np.int)]
    run_arr = np.array(list(zip(start_mjd, stop_mjd, run_dict["run"])),
                       dtype=dtype)

    # Note: The last 2 runs aren't included anyway, so he left them out in
    # the reported run list. This fits here, as the other 4 runs are found
    # in the list.
    exclude_rate = [120028, 120029, 120030, 120087, 120156, 120157]
    i3good = (run_dict["good_i3"] == True)
    itgood = (run_dict["good_it"] == True)
    ratebad = np.in1d(run_dict["run"], exclude_rate)

    # Include if it & i3 good and rate is good
    include = i3good & itgood & ~ratebad
    inc_run_arr = run_arr[include]

    # Get the total and per run livetimes in mjd
    runtimes_mjd = inc_run_arr["stop_mjd"] - inc_run_arr["start_mjd"]
    _livetime = np.sum(runtimes_mjd)

    return inc_run_arr, _livetime
