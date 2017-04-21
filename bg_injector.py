"""
Class for random background event injection.

Provides different models to sample events in a given timeframe and for a
given event position from experimental data.
"""

import numpy as np
import sklearn.neighbors as skn


VALID_METHODS = ["fixed_kde", "adaptive_kde", "data"]


class BGInjector(object):
    """
    Background Injector Interface

    Injects background events obtained from a given data sample.
    Implements a `fit` and a `sample` method.
    """
    def fit(self, x):
        """
        Build the injection model with the provided data

        Parameters
        ----------
        x : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """
        raise NotImplementedError("BGInjector is an interface.")

    def sample(self, n_samples=1, random_state=None):
        """
        Generate random samples from the fitted model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. (defaults: 1)
        random_state : RandomState, optional
            A random number generator instance. (default: None)

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            Generated samples from the fitted model.
        """
        raise NotImplementedError("BGInjector is an interface.")

class FixedBandwidthKDEInjector(BGInjector):
    """
    Background Injector using a kernel density estimator with a fixed bandwidth
    to describe the background pdf.

    This is a wrapper for `sklearn.neighbours.KernelDensity` which implements
    a `fit` and a `sample`method.
    The sample method is only supported for 'gaussian' and 'tophat' kernels.
    """
    def __init__(self):
        # Set the docstrings of the interface to the implemented classes.
        self.fit.__func__.__doc__ = super(FixedBandwidthKDEInjector,
                                          self).fit.__doc__
        self.sample.__func__.__doc__ = super(FixedBandwidthKDEInjector,
                                             self).sample.__doc__

        return

    def fit(self, x):
        return

    def sample(self, n_samples=1, random_state=None):
        return
