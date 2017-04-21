"""
Class for random background event injection.

Provides different models to sample events in a given timeframe and for a
given event position from experimental data.
"""

import numpy as np
import sklearn.neighbors as skn
from sklearn.utils import check_random_state


class BGInjector(object):
    """
    Background Injector Interface

    Injects background events obtained from a given data sample.
    Implements a `fit` and a `sample` method.
    """
    _IMPLEMENTS = ["fit", "sample"]

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


class FixedBandwidthKDEBGInjector(BGInjector):
    """
    Fixed Bandwidth Kernel Density Background Injector

    Background Injector using a kernel density estimator with a fixed bandwidth
    to describe the background pdf.
    This is a wrapper for `sklearn.neighbours.KernelDensity` which implements
    a `fit` and a `sample`method.
    The sample method is only supported for 'gaussian' and 'tophat' kernels.

    Parameters
    ----------
    XtoY : function, call signature XtoY(X), returns Y
        Function, that takes the data X in and transforms the to be suitable for
        a global KDE. Returns Y with same shape but transformed values. The KDE
        is build on Y afterwards.
    YtoX : function, call signature YtoX(Y), returns X
        Backtransformation of `XtoY`. Returns X in the original paramter
        space after sampling.
    bandwidth : float
        The bandwidth of the kernel. Should be optimized in a seperate step in
        the same space as XtoY implies. (default: 1.)
    kernel : string
        The kernel to use. Valid kernels are ['gaussian'|'tophat'], the only
        two that have the sample method. (default: 'gaussian')
    rtol : float
        The desired relative tolerance of the result.  A larger tolerance will
        generally lead to faster execution. (default: 1e-8)

    KDEargs
    -------
    Other parameters are passed directly to the the
    `sklearn.neighbours.KernelDensity` class.
    """
    _VALID_KERNELS = ["gaussian", "tophat"]

    def __init__(self, XtoY, YtoX, bandwidth=1.,
                 kernel="gaussian", rtol=1e-8, **KDEargs):
        inherit_docstrings_from_interface(self, "FixedBandwidthKDEBGInjector")

        self.XtoY = XtoY
        self.YtoX = YtoX
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.rtol = rtol

        if XtoY is None or YtoX is None:
            raise ValueError("Transformation functions can't be None.")

        if kernel not in self._VALID_KERNELS:
            raise ValueError("Invalid kernel: '{}'".fromat(kernel))

        # Create KDE model
        self.kde_model = skn.KernelDensity(bandwidth=bandwidth,
                                           kernel=kernel,
                                           rtol=rtol,
                                           **KDEargs
                                           )

        return

    def fit(self, X):
        self._n_features = X.shape[1]

        # Transform data X to KDE space Y
        Y = self.XtoY(X)

        self.kde_model.fit(Y)
        return

    def sample(self, n_samples=1, range=None, random_state=None):
        range, random_state = check_sample_params(self, n_samples, range,
                                                  random_state)

        # Scale ranges to KDE space
        range = self.XtoY(range)

        # Check which samples are in range, redraw those that are not
        Y = np.empty((0, self._n_features))
        while n_samples > 0:
            gen = self.kde_model.sample(n_samples, random_state)
            accepted = np.all(np.logical_and(gen >= range[0],
                                             gen <= range[1]), axis=1)
            n_samples = np.sum(~accepted)
            print("Not accepted : ", n_samples)
            # Append accepted to final sample
            Y = np.append(Y, gen[accepted], axis=0)

        # Unscale sample to get original feature space
        return self.YtoX(Y)


class DataBGInjector(BGInjector):
    """
    Data Background Injector

    Background Injector selecting random data events from the given sample.
    """
    def __init__(self):
        inherit_docstrings_from_interface(self, "DataBGInjector")
        return

    def fit(self, X):
        # The 'model' is the data itself
        self.X = np.copy(X)
        self._n_features = X.shape[1]
        return

    def sample(self, n_samples=1, range=None, random_state=None):
        range, random_state = check_sample_params(self, n_samples, range,
                                                  random_state)

        # Draw indices uniformly from the data in range and return the events
        X = self.X
        accepted = np.all(np.logical_and(X >= range[0],
                                         X <= range[1]), axis=1)
        X = X[accepted]
        idx = random_state.randint(X.shape[0], size=n_samples)
        return X[idx]


class UniformBGInjector(BGInjector):
    # "Close" to real data
    _logE_mean = 3.
    _logE_sigma = 1.
    _sigma_scale = 2.
    """
    Uniform Background Injector

    Background Injector creating uniform events on the whole sky.
    Created features are logE from a gaussian with given mean and covariance,
    declination in radian uniformly distributed on the sphere and sigma in
    radianfrom an exponential distribution with scale beta.

    Note: Uses emcee to sample a nice sigma distribtuion x * exp(-x)
    """
    def __init__(self):
        inherit_docstrings_from_interface(self, "UniformBGInjector")
        return

    def fit(self, X=None):
        self._n_features = 3
        return

    def sample(self, n_samples=1, range=None, random_state=None):
        range, random_state = check_sample_params(self, n_samples, range,
                                                  random_state)

        X = np.zeros((n_samples, 3), dtype=np.float)

        X[:, 0] = random_state.normal(self._logE_mean,
                                      self._logE_sigma, size=n_samples)
        X[:, 1] = np.arccos(random_state.uniform(-1, 1, size=n_samples))

        # Sample a nice sigma with x * exp(-x)
        def logpdf(x):
            if x < 0:
                return -np.inf
            return np.log(x) - self._sigma_scale * x

        import emcee
        n_walkers = 6
        n_mcmc = int(np.ceil(n_samples / n_walkers))
        sampler = emcee.EnsembleSampler(dim=1, nwalkers=n_walkers,
                                        lnpostfn=logpdf)
        sampler.run_mcmc(N=n_mcmc,
                         pos0=np.random.uniform(0.4, 0.6, size=(n_walkers, 1)))
        X[:, 2] = np.deg2rad(sampler.flatchain.flatten()[:n_samples])

        return X


# #############################################################################
# ## Utils
# #############################################################################


def check_sample_params(self, n_samples, range, random_state):
    """
    Just performs checks on parameter sanity for the BGInjector sample method.

    The `random_state` is passed to `sklearn.util.check_random_state` which
    creates a new RandomState depending on the input type or just returns the
    RandomState if given.

    The range is multiplied with the scale to match the data scale.
    """
    if n_samples < 1:
        raise ValueError("'n_samples' must be at least 1.")

    if range is None:
        range = np.repeat([[-np.inf, np.inf], ],
                          repeats=self._n_features, axis=0).T
    range = np.array(range)
    if range.shape[0] != 2:
        raise ValueError("Invalid 'range'. Must be shape (2, n_features).")
    elif np.any(np.equal(range, None)):
        raise ValueError("'None' not allowed, use `+-np.inf` instead.")

    # Just selects depending on None, Int or RandomState type
    random_state = check_random_state(random_state)

    return range, random_state


def inherit_docstrings_from_interface(self, descr, func=None):
    """
    Inherits all function docstrings from superclass functions defined in its
    `_IMPLEMENTS` list.

    Parameters
    ----------
    self : object
        Current instance of child class.
    descr : string
        Name of child class.
    func : list of strings
        If not None, only the functions given here get the docstring attached.
        If None, all listed in super class' `_IMPLEMENTS` are used.
        (default: None)
    """
    if func is None:
        func = eval("super({0}, self)._IMPLEMENTS".format(descr))

    for f in func:
        exec("self.{0}.__func__.__doc__ = ".format(f) +
             "super({0}, self).{1}.__doc__".format(descr, f))

    return
