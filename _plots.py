# coding: utf-8

"""
Some helper functions to keep the notebook clean.
More important functions are kept in the notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from tdepps.utils import rotator


def idx2rowcol(idx, ncols):
    """
    Convert a 1d running index to ``[row, col]`` indices. ``numpy`` broadcasting
    rules apply to ``idx``

    Parameters
    ----------
    idx : int or array-like
        Current index / indices, ``idx >= 0``.
    ncols : int
        Number of columns to index, ``ncols > 0``.

    Returns
    -------
    row, col : int
        Row and column indices.
    """
    row = np.floor(idx / ncols).astype(int)
    col = np.mod(idx, ncols).astype(int)
    return row, col


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
    the pole and then rotate it to the desired center.
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
