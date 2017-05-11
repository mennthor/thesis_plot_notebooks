# Cross validation results from adaptive width kernels

Two CVs have been done, one with `glob_bw` parameter, controlling initial fixed bandwidth, only.
The other with both parameters `glob_bw` and `alpha` to crosscheck.
The single parameter case is much faster and can be run within hours, while the 2D grid space needs to test much more settings.

**Note:** The result file contains the whole `sklearn.model_selector` instance and has the best estimator ready to use.
It can be used to sample from or evaluate other points, as long as the trained data sample stays the same (as given below).
`alpha` can still be changed, but the global bandwidth needs to remain fixed.
Otherwise the KDE has to be fitted anew.

Usage:

    import pickle
    with open("CV10_glob_bw_EXP_IC86I_CUT_sig.ll.90" + \
              "_PARS_diag_True_alpha_0.5_pass2.pickle", "rb") as f:
        ms = pickle.load(f)
    best_kde = ms.best_estimator_


## glob_bw only

Tested with coarse and fine scanning.

### Passes

First pass:

- Script: `CV10_glob_bw_EXP_IC86I_CUT_sig.ll.90_PARS_diag_True_alpha_0.5.py`
- Result file: `CV10_glob_bw_EXP_IC86I_CUT_sig.ll.90_PARS_diag_True_alpha_0.5.pickle`
- Tested ranges:
    + `glob_bw` `np.arange(0.01, 0.1, 0.01)`
- Best parameter:
    + `glob_bw`: 0.040

Second pass:

- Script: `CV10_glob_bw_EXP_IC86I_CUT_sig.ll.90_PARS_diag_True_alpha_0.5_pass2.py`
- Result file: `CV10_glob_bw_EXP_IC86I_CUT_sig.ll.90_PARS_diag_True_alpha_0.5_pass2.pickle`
- Tested ranges:
    + `glob_bw`: `np.arange(0.03, 0.05, 0.001)`
- Best parameters: 
    + `glob_bw`: 0.044

### Common Settings

Fixed KDE params:

- `alpha`: 0.5
- `diag_cov`: True

CV settings:

- Number of splits: 10

Data settings:

- Sample: `IC86-I_data.npy` from `iccobalt:/home/epinat/PointSource/data_final`
- Order: `np.vstack((logE, dec, sigma)).T`
    + `logE` in GeV
    + `dec` in radian
    + `sigma` in radian
- Data set cuts:
    + `sigma` < 90°


## glob_bw and alpha

Tested with coarse and fine scanning.

### Passes

First pass:

- Script: `CV10_glob_bw_alpha_EXP_IC86I_CUT_sig.ll.20_PARS_diag_True.py`
- Result file: `CV10_glob_bw_alpha_EXP_IC86I_CUT_sig.ll.20_PARS_diag_True.pickle`
- Tested ranges:
    + `glob_bw`: `np.linspace(0.025, 0.125, 10)`
    + `alpha`  : `np.linspace(0.1, 0.9, 10)`
- Best parameters:
    + `glob_bw`: 0.0472
    + `alpha`  : 0.4555

Second pass:

- Script: `CV10_glob_bw_alpha_EXP_IC86I_CUT_sig.ll.20_PARS_diag_True_pass2.py`
- Result file: `CV10_glob_bw_alpha_EXP_IC86I_CUT_sig.ll.20_PARS_diag_True_pass2.pickle`
- Tested ranges:
    + `glob_bw` `np.arange(0.03, 0.05, 0.001)`
- Best parameters:
    + `glob_bw`: 0.0475
    + `alpha`  : 0.4499

### Common Settings

Fixed KDE params:

- `diag_cov`: True

CV settings:

- Number of splits: 10

Data settings:

- Sample: `IC86-I_data.npy` from `iccobalt:/home/epinat/PointSource/data_final`
- Order: `np.vstack((logE, dec, sigma)).T`
    + `logE` in GeV
    + `dec` in radian
    + `sigma` in radian
- Data set cut: `sigma` < 20°
    + Doesn't really matter much, above 20° there are only few very badly reconstructed events
