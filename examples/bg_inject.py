# coding: utf-8

import bg_injector as bgi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


exp = np.load("../data/IC86_I_data.npy")
X = np.vstack([exp["logE"], exp["dec"], exp["sigma"]]).T


MODE = "UNIFORM"
print("Sampling Mode is : ", MODE)


if MODE == "KDE":
    # Apply the 5° sigma cut
    mask = X[:, 2] <= 5
    X = X[mask]

    # Define the transformed coordinates on which the KDE should be fitted
    def XtoY(X):
        """Transform X in real space to Y in KDE space."""
        scale = [1.5, 2.5, 2.0]
        X[:, 2] = np.rad2deg(X[:, 2])
        X *= scale
        return X

    def YtoX(Y):
        """Transform back from Y in KDE space to X in real space."""
        scale = [1.5, 2.5, 2.0]
        Y /= scale
        Y[:, 2] = np.deg2rad(Y[:, 2])
        return Y

    bw = 0.114  # From cross validation with the above transformations
    dinj = bgi.FixedBandwidthKDEBGInjector(XtoY, YtoX, bandwidth=bw)

    # KDE kernel was optimized on a certain scale and with a cut on sigma
    dinj.fit(X)

    # Restrict to phyisical range, because KDE smears out a little
    range = np.vstack([[-np.inf, np.inf],
                       [-np.pi / 2., np.pi / 2.],
                       [0, np.inf]]).T

    sample = dinj.sample(int(1e6), range=range)

    _ = plt.hist2d(sample[:, 0], sample[:, 1], bins=50,
                   normed=True, norm=LogNorm())
    plt.show()
    _ = plt.hist(sample[:, 2], bins=50, range=[0, np.deg2rad(5)], normed=True)
    plt.show()

elif MODE == "DATA":
    dinj = bgi.DataBGInjector()
    dinj.fit(X)
    sample = dinj.sample(int(1e6))

    _ = plt.hist2d(sample[:, 0], sample[:, 1], bins=50,
                   normed=True, norm=LogNorm())
    plt.show()
    _ = plt.hist(sample[:, 2], bins=50, range=[0, np.deg2rad(5)], normed=True)
    plt.show()

elif MODE == "UNIFORM":
    dinj = bgi.UniformBGInjector()
    dinj.fit()
    sample = dinj.sample(int(1e5))

    _ = plt.hist2d(sample[:, 0], sample[:, 1], bins=50,
                   normed=True, norm=LogNorm())
    plt.show()
    _ = plt.hist(sample[:, 2], bins=50, range=[0, np.deg2rad(5)], normed=True)
    plt.show()
