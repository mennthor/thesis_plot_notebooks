#!/home/tmenne/software/miniconda3/bin/python

"""
Usage:
  my_program.py -n <ntrials> -s <seed> -d <data> -m <mc> -r <runlist> -o <out>

Arguments:
  -n <ntrials>   number of trials to generate
  -s <seed>      random seed for trial generation
  -d <data>      npy file with data used to build the PDFs
  -m <mc>        npy file with mc data used to build the PDFs
  -r <runlist>   snapshot goodrunlist in json format
  -o <out>       outfile for saved trials

-h --help     show this help message
"""

from docopt import docopt
import os
import pickle
import numpy as np

import tdepps.bg_injector as BGInj
import tdepps.bg_rate_injector as BGRateInj
import tdepps.rate_function as RateFunc
import tdepps.llh as LLH
import tdepps.analysis as Analysis


if __name__ == "__main__":
    """
    Run background only trials in tdepps framework.

    More explanation in docs. This is condensed as it's a job file.
    """
    args = docopt(__doc__)
    ntrials = args["-n"]
    rnd_seed = args["-s"]
    data_file = os.path.abspath(args["-d"])
    mc_file = os.path.abspath(args["-m"])
    runlist_file = os.path.abspath(args["-r"])
    out_file = os.path.abspath(args["-o"])

    # Test arguments
    for key, val in args.items():
        print(key, val)

    # Load data
    exp = np.load(data_file)
    mc = np.load(mc_file)
    # Make a global sigma cut (removes a handful of badly reconstructed evts)
    mc = mc[mc["sigma"] < np.deg2rad(20)]
    exp = exp[exp["sigma"] < np.deg2rad(20)]

    # #########################################################################
    # 1. Create a bg rate injector model
    def filter_runs(run):
        """
        Filter runs as stated in jfeintzig's doc [1]_.

        Notes
        -----
        .. [1] https://wiki.icecube.wisc.edu/index.php/IC86_I_Point_Source_Analysis/Data_and_Simulation # noqa
        """
        exclude_runs = [120028, 120029, 120030, 120087, 120156, 120157]
        if ((run["good_i3"] is True) & (run["good_it"] is True) &
                (run["run"] not in exclude_runs)):
            return True
        else:
            return False

    # Create an injector using a goodrun list, use a sinus rate function with
    # fixed period of 1yr
    rate_func = RateFunc.Sinus1yrRateFunction()
    runlist_inj = BGRateInj.RunlistBGRateInjector(runlist_file, filter_runs,
                                                  rate_func)
    # Fit the injector to make it usable
    rate_func = runlist_inj.fit(T=exp["timeMJD"], x0=None,
                                remove_zero_runs=True)

    # #########################################################################
    # 2. Create our bg injector, here we use the data resampler
    data_inj = BGInj.DataBGInjector()
    data_inj.fit(exp)

    # #########################################################################
    # 3. Create the GRBLLH object with all the PDF settings
    sin_dec_bins = np.linspace(-1, 1, 50)

    min_logE = 1
    max_logE = 10
    logE_bins = np.linspace(min_logE, max_logE, 40)

    spatial_pdf_args = {"bins": sin_dec_bins, "k": 3, "kent": True}

    energy_pdf_args = {"bins": [sin_dec_bins, logE_bins],
                       "gamma": 2., "fillval": "col", "interpol_log": False}

    time_pdf_args = {"nsig": 4., "sigma_t_min": 2., "sigma_t_max": 30.}

    grbllh = LLH.GRBLLH(X=exp, MC=mc, srcs=None,
                        spatial_pdf_args=spatial_pdf_args,
                        energy_pdf_args=energy_pdf_args,
                        time_pdf_args=time_pdf_args)

    # #########################################################################
    # 4. Create test src setup we want to test, with some different properties.
    nsrcs = 5
    dt = np.vstack((np.repeat([-20.], nsrcs), 100. * np.arange(1, nsrcs + 1))).T

    names = ["t", "dt0", "dt1", "ra", "dec", "w_theo"]
    types = len(names) * [np.float]
    dtype = [(_n, _t) for _n, _t in zip(names, types)]
    srcs = np.empty((nsrcs, ), dtype=dtype)

    # Choose times equally spaced, but away from borders
    mint, maxt = np.amin(exp["timeMJD"]), np.amax(exp["timeMJD"])
    srcs["t"] = np.linspace(mint, maxt, nsrcs + 2)[1:-1]

    srcs["dt0"] = dt[:, 0]
    srcs["dt1"] = dt[:, 1]

    # Don't let them overlap at 0,2pi
    srcs["ra"] = np.linspace(0, 2 * np.pi, nsrcs + 1)[:-1]

    # Don't select directly at poles
    srcs["dec"] = np.arcsin(np.linspace(-1, 1, nsrcs + 2)[1:-1])

    # These are just ones, they shouldn't cause problems
    srcs["w_theo"] = np.ones(nsrcs, dtype=np.float)

    # #########################################################################
    # And now the analysis object to run trials
    ana = Analysis.TransientsAnalysis(srcs=srcs, llh=grbllh)
    theta0 = {"ns": 10}
    res, nzeros = ana.do_trials(ntrials, theta0,
                                bg_inj=data_inj, bg_rate_inj=runlist_inj,
                                random_state=rnd_seed, minimizer_opts=None)

    # Just pickle away, combine seperately later
    out_dict = {"res": res, "nzeros": nzeros}

    # Create dir if not existing
    os.makedirs(os.path.dirname(out_file))

    with open(out_file, "wb") as outf:
        pickle.dump(file=outf, obj=out_dict)
