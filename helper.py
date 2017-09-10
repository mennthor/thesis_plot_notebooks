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
from anapymods3.healpy import rotator


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
    """
    Takes 2 2d samples, sam1, sam2 and creates 2d histograms next to each other
    in a single figure.
    Return figure and both axis objects.
    """
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


def circle_on_skymap(ra0, dec0, r, ax, flat=False, **kwargs):
    """
    Draws correct circles in spherical coordinates by drawing a circle around
    the pole and then rotation it to the desired center.
    Has a bug when drawing on skymaps, then stray lines may appear when the
    circle is split in right ascension.

    ra0, dec0 : circle center in euqatorial coords in radians
    r : circle radius in radians
    ax : the axis to draw onto
    flat : if true, do not convert to skymap coordinates (x, y). Can be used
        when plotting ra, dec on a flat plot
    kwargs are passed to plt.plot.
    """
    # Make correct circle points around the pole, rotate all back to real
    # position of circle center: (ra_from, dec_from) -> (ra0, dec0)
    npts = 100
    ra_rot = np.linspace(0, 2 * np.pi, npts)
    dec_rot = np.pi / 2. - np.ones(npts) * r

    # Center of the circle at the pole
    ra_from = np.zeros(npts, dtype=np.float)
    dec_from = np.zeros(npts, dtype=np.float) + np.pi / 2.

    ra0 = np.repeat(ra0, npts)
    dec0 = np.repeat(dec0, npts)

    ra, dec = rotator(ra_from, dec_from, ra0, dec0, ra_rot, dec_rot)

    # Back to map coordinates
    if not flat:
        x, y = np.pi - ra, dec
    else:
        x, y = ra, dec

    # Find circles which are over the 2pi periodic border in x [0, 2pi]
    # Spit sample and draw seperately
    m = np.abs(np.diff(x)) > np.deg2rad(90)
    idx = np.argmax(m)

    if np.sum(m) > 0:
        ax.plot(x[:idx + 1], y[:idx + 1], **kwargs)
        ax.plot(x[idx + 1:], y[idx + 1:], **kwargs)
    else:
        ax.plot(x, y, **kwargs)
    return ax


def corner_hist(h, bins, label=None, color="k", cmap="inferno",
                hist_args={}, hist2D_args={}):
    """
    Plot marginalized 1D and 2D disgtributions of given histogram.

    Parameter
    ---------
    h : array
        nD array with shape (nbins 1st dim, ..., nbins last dim).
    bins : list
        List of len nD. Each item is an arrays containing the bin borders in
        that dimension
    label : list of strings
        Axis label

    kwargs
    ------
    hist_args : dict
        Arguments passed to matplotlib 1D hist function
    hist2D_args : dict
        Arguments passed to matplotlib 2D hist function

    Returns
    -------
    fig, ax : matplotlib figure and axis object
        The figure object and array (nD x nD) of axes objects.
    """
    h = np.atleast_1d(h)
    dim = len(h.shape)
    if dim != len(bins):
        raise ValueError("For each dimension a list of bins must be " +
                         "provided. Dimensionality of hist and bins doesn't " +
                         "match.")

    # Get bin mids for the "plot existing hist as weights trick"
    mids = []
    for b in bins:
        mids.append(0.5 * (b[:-1] + b[1:]))

    # Labels are manually handled further below
    fig, ax = plt.subplots(dim, dim, sharex=False,
                           sharey=False, figsize=(4 * dim, 4 * dim))

    # First set correct axes limits
    for row in range(dim):
        for col in range(dim):
            if col > row:  # Uper diagonal is turned off
                ax[row, col].axis("off")
            else:
                ax[row, col].set_xlim(bins[col][0], bins[col][-1])
                if row != col:
                    # 2D case: y limits are set with respect to bins
                    ax[row, col].set_ylim(bins[row][0], bins[row][-1])
                else:
                    # Set ticks right in 1D to distinguish from 2D yaxis
                    ax[row, col].yaxis.tick_right()

    # Diagonal is 1D, else are 2D marginalization = sum over all remaining dims
    for row in range(dim):
        for col in range(row + 1):
            if row == col:
                # For the 1D case we sum over n-1 dimensions
                axis = np.ones(dim, dtype=bool)
                axis[row] = False

                axis = tuple(np.arange(dim)[axis])
                hist = np.sum(h, axis=axis)

                ax[row, col].hist(mids[row], bins=bins[row], weights=hist,
                                  **hist_args)
            else:
                # For the 2D case we sum over n-2 dimensions
                xx, yy = np.meshgrid(mids[col], mids[row])
                XX = xx.flatten()
                YY = yy.flatten()
                axis = np.ones(dim, dtype=bool)

                axis[row] = False
                axis[col] = False
                axis = tuple(np.arange(dim)[axis])
                # We need to transpose, because dimensions are swapped between
                # meshgrid and numpy.histogrammdd
                hflat = np.sum(h, axis=axis).T.flatten()

                ax[row, col].hist2d(XX, YY, bins=[bins[col], bins[row]],
                                    weights=hflat, **hist2D_args)

    # Set axis label
    if label is not None:
        for col in range(dim):
            ax[-1, col].set_xlabel(label[col])
            ax[col, col].set_ylabel("counts")
            ax[col, col].yaxis.set_label_position("right")
        for row in range(1, dim):
            ax[row, 0].set_ylabel(label[row])

    # Rotate lower xticklabel so they don't interfere
    for col in range(dim):
        for label in ax[-1, col].get_xticklabels():
            label.set_rotation(60)

    # Unset x label and ticks manually for internal axes
    if dim > 1:
        for row in range(0, dim - 1):
            for col in range(0, row + 1):
                ax[row, col].get_xaxis().set_ticklabels([])

        # Unset y label and ticks manually for internal axes.
        for row in range(1, dim):
            for col in range(1, row):
                ax[row, col].get_yaxis().set_ticklabels([])

    # Make plots square. Set aspect needs the correct ratio.
    # "Equal" just equals the axis range
    for row in range(dim):
        for col in range(dim):
            ax[row, col].set_aspect(1. / ax[row, col].get_data_ratio())

    fig.tight_layout(h_pad=-1, w_pad=-3)

    return fig, ax
