"""
Some helper functions to keep the notebook clean.
More important functions are kept in the notebook.
"""

import numpy as np

import json
from astropy.time import Time as astrotime
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


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


def create_goodrun_dict(runlist, filter_runs):
    """
    Create a dict of lists. Each entry in each list is one run.

    Parameters
    ----------
    runlist : str
        Path to a valid good run runlist snapshot from [1]_ in JSON format.
        Must have keys 'latest_snapshot' and 'runs'.
    filter_runs : function
        Filter function to remove unwanted runs from the goodrun list.
        Called as `filter_runs(run)`. Function must operate on a single
        dictionary `run`, with keys:

            ['good_i3', 'good_it', 'good_tstart', 'good_tstop', 'run',
             'reason_i3', 'reason_it', 'source_tstart', 'source_tstop',
             'snapshot', 'sha']

    Returns
    -------
    goodrun_dict : dict
        Dictionary with run attributes as keys. The values are stored in
        lists in each key. One list item is one run.

    Notes
    -----
    .. [1] https://live.icecube.wisc.edu/snapshots/
    """
    with open(runlist, 'r') as jsonFile:
        goodruns = json.load(jsonFile)

    if not all([k in goodruns.keys() for k in ["latest_snapshot", "runs"]]):
        raise ValueError("Runlist misses 'latest_snapshot' or 'runs'")

    # This is a list of dicts (one dict per run)
    goodrun_list = goodruns["runs"]

    # Filter to remove unwanted runs
    goodrun_list = list(filter(filter_runs, goodrun_list))

    # Convert the run list of dicts to a dict of arrays for easier handling
    goodrun_dict = dict(zip(goodrun_list[0].keys(),
                            zip(*[r.values() for r in goodrun_list])))
    for k in goodrun_dict.keys():
        goodrun_dict[k] = np.array(goodrun_dict[k])

    # Add times to MJD floats
    goodrun_dict["good_start_mjd"] = astrotime(
        goodrun_dict["good_tstart"]).mjd
    goodrun_dict["good_stop_mjd"] = astrotime(
        goodrun_dict["good_tstop"]).mjd

    # Add runtimes in MJD days
    goodrun_dict["runtime_days"] = (goodrun_dict["good_stop_mjd"] -
                                    goodrun_dict["good_start_mjd"])

    livetime = np.sum(goodrun_dict["runtime_days"])

    return goodrun_dict, livetime


def hist_comp(sam1, sam2, **kwargs):
    figsize = kwargs.pop("figsize", (12, 6))
    fig, (al, ar) = plt.subplots(1, 2, figsize=figsize)

    normed = kwargs.pop("normed", True)
    cmap = kwargs.pop("cmap", "inferno")
    vmin, vmax = kwargs.pop("crnge", [None, None])
    bins = kwargs.pop("bins", 50)
    log = kwargs.pop("log", True)

    histargs = dict(bins=bins,
                    cmap=cmap,
                    normed=normed,
                    norm=LogNorm() if log else Normalize(),
                    vmin=vmin,
                    vmax=vmax,
                   )

    _, bx, by, img = al.hist2d(sam1[:, 0], sam1[:, 1], **histargs)
    plt.colorbar(ax=al, mappable=img)

    # Choose exact same binning here
    histargs.pop("bins")
    _, _, _, img = ar.hist2d(sam2[:, 0], sam2[:, 1], bins=[bx, by], **histargs)
    plt.colorbar(ax=ar, mappable=img)

    return fig, (al, ar)
