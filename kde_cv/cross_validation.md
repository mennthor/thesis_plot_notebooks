# Transformation of the data for the fixed width KDE

The kernel width was optimized using a 20-fold cross validation using the sklearn GridSearchCV method.
Three scans with increasing resolution were used to narrow in on the optimum width.
Each pickle file holds the result of the grid search with the scanned paramter space and the best result.

Data was transformed before optimizing the gaussian kernel width, because the KDE from sklearn uses a single bandwidth and a symmetric kernel.
When having different scales in data, one kernel simply can't catch'em all.
So we scale the data (by hand and eye at the moment) to have the features on an almost equal scale.

The transformation simply scales each values with the factors given below.

    logE  : 1.5,
    dec   : 2.5,
    sigma : 2.0

- logE is used in log10(E / GeV).
    + logE is good from the beginning. It falls of smoothly to zero on both ends. We simply scale the peak width to match the other features a bit.
- declination is used in radian and normalspace
    + This has better properties, because the edges are falling to zero, whcih suits the gaussian kernel much better than a sharp cut as in sin(dec) for example.
- sigma is taken in degree, because otherwise the values are all well below 1 (>5°)
    + It need a bit scaling because it drops quite hard to zero at the lower edge sigma=0.

Furthermore a cut on sigma is applied, only using events with sigma <= 5°.
This prevents Kernel creation on badly reconstructed outlier events.
The kernel properties smear in that region anyway but in a much smoother way.



