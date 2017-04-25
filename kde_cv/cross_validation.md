# Transformation of the data for the fixed width KDE

The kernel width was optimized using a 20-fold cross validation using the sklearn GridSearchCV method.
Three scans with increasing resolution were used to narrow in on the optimum width.
Each pickle file holds the result of the grid search with the scanned paramter space and the best result.

Data was transformed before optimizing the gaussian kernel width, because the KDE from sklearn uses a single bandwidth and a symmetric kernel.
When having different scales in data, one kernel simply can't catch'em all.
So we scale the data (by hand and eye at the moment) to have the features on an almost equal scale.

The transformation simply scales each values with the factors given below.

    logE  : 1.5
    dec   : 2.5
    sigma : 2.0  &  sigma (deg) <= 5°

- logE is used in log10(E / GeV).
    + logE is good from the beginning. It falls of smoothly to zero on both ends. We simply scale the peak width to match the other features a bit.
- declination is used in radian and normal space.
    + This has better properties, because the edges are falling to zero, which suits the gaussian kernel much better than a sharp cut at the boundaries as in sin(dec) for example.
- sigma is taken in degree to scale it to the other parameters naturally.
    + It still needs a bit manual scaling because it drops quite hard to zero at the lower edge sigma=0 so we broaden this to better fit a gaussian kernel.
    + Furthermore a cut on sigma is applied **before** fitting, only using events with sigma <= 5°.
    + This prevents Kernel creation on badly reconstructed outlier events. The kernel properties smear in that region anyway but in a much smoother way.

All explicit settings used are:

    kernel = "gaussian"
    rtol   = 1e-8

everything else left to default values.
