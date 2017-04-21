"""
Class for sampling the number of expected background events in a time frame.

Background expectation is estimated from data resolved in time and declination.
These are the two parameters, that are relevant for a given source.
"""

import numpy as np
import sklearn.neighbors as skn
from sklearn.utils import check_random_state


class BGRateInjector(object):
    """
    Background Rate Injector

    Create a 2D smooth estimate of the data PDF in time and declination.
    The smoothing removes any influence from over- or underfluctuations on data
    to estimate the background rate robustly.

    The rate PDF is created by fitting a `scipy.interpolate.RectBivariateSpline`
    to a 2D histogram.
    Outside the data range, the spline is set to zero, suitable for the hard cut
    in time.

    Parameters
    ----------
    bins : int or array-like
        Binning of the 2D data histogram. Each bin is z-value for the splines.

    """

    def __init__(self):
        print("Interface only. Defines functions: ", self._IMPLEMENTS)
        return

    def fit(self, X):
        """
        Build the injection model with the provided data

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """
        raise NotImplementedError("BGInjector is an interface.")

    def sample(self, n_samples=1, range=None, random_state=None):
        """
        Generate random samples from the fitted model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. (defaults: 1)
        range : array-like, shape (2, n_features)
            Give a range [low, hig] in which events shall be sampled.
            (default: None)
        random_state : RandomState, optional
            A random number generator instance. (default: None)

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            Generated samples from the fitted model.
        """
        raise NotImplementedError("BGInjector is an interface.")