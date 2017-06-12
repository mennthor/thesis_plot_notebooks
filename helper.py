# coding: utf-8

"""
Some helper functions to keep the notebook clean.
More important functions are kept in the notebook.
"""

import numpy as np
import json
from astropy.time import Time as astrotime
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


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


def _create_runtime_bins(X, goodrun_dict, remove_zero_runs=False):
    """
    Creates time bins [start_MJD_i, stop_MJD_i] for each run i and bin the
    experimental data to calculate the rate for each run.

    Parameters
    ----------
    X : array_like, shape (n_samples)
        MJD times of experimental data.
    goodrun_dict : dict
        Dictionary with run attributes as keys. The values are stored in
        lists in each key. One list item is one run.
    remove_zero_runs : bool, optional
        If True, remove all runs with zero events and adapt the livetime.
        (default: False)

    Returns
    -------
    rate_rec : recarray, shape(nruns)
        Record array with keys:
        - "run" : int, ID of the run.
        - "rate" : float, rate in Hz in this run.
        - "runtime" : float, livetime of this run in MJD days.
        - "start_mjd" : float, MJD start time of the run.
        - "stop_mjd" : float, MJD end time of the run.
        - "nevts" : int, numver of events in this run.
        - "rates_std" : float, sqrt(N) stddev of the rate in Hz in this run.
    """
    _secinday = 24. * 60. * 60.
    # Store events in bins with run borders
    start_mjd = goodrun_dict["good_start_mjd"]
    stop_mjd = goodrun_dict["good_stop_mjd"]
    run = goodrun_dict["run"]

    tot_evts = 0
    # Histogram time values in each run manually
    evts = np.zeros_like(run, dtype=int)
    for i, (start, stop) in enumerate(zip(start_mjd, stop_mjd)):
        mask = (X >= start) & (X < stop)
        evts[i] = np.sum(mask)
        tot_evts += np.sum(mask)

    # Crosscheck, if we got all events and didn't double count
    if not tot_evts == len(X):
        print("Events selected : ", tot_evts)
        print("Events in X     : ", len(X))
        raise ValueError("Not all events in 'X' were sorted in bins. If " +
                         "this is intended, please remove them beforehand.")

    if remove_zero_runs:
        # Remove all zero event runs and update livetime
        m = (evts > 0)
        _livetime = np.sum(stop_mjd - start_mjd)
        evts, run = evts[m], run[m]
        start_mjd, stop_mjd = start_mjd[m], stop_mjd[m]
        print("Removing runs with zero events")
        print("  Number of runs with 0 events : {:d}".format(np.sum(~m)))
        print("  Total livetime of those runs : {} d".format(_livetime))

    # Normalize to rate in Hz
    runtime = stop_mjd - start_mjd
    rate = evts / (runtime * _secinday)

    # Calculate 1 / sqrt(N) stddev for scaled rates
    rate_std = np.sqrt(rate) / np.sqrt(runtime * _secinday)

    # Create record-array
    names = ["run", "rate", "runtime", "start_mjd",
             "stop_mjd", "nevts", "rate_std"]
    types = [int, np.float, np.float, np.float, np.float, int, np.float]
    dtype = [(n, t) for n, t in zip(names, types)]

    a = np.vstack((run, rate, runtime, start_mjd, stop_mjd, evts, rate_std))
    rate_rec = np.core.records.fromarrays(a, dtype=dtype)

    return rate_rec


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
