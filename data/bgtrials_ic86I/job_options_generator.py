"""
Generate JOB, VARS list for each job to submit.
"""

import os
import numpy as np


# Steering parameters for the jobs
jobnames = "tdepps_ic86Ibg_trials"
oj_submit_path = os.path.abspath("./onejob.submit")
job_options_fpath = os.path.abspath("./job_options.dag")
njobs = int(1e4)

# Arguments for each job
ntrials = int(1e5)  # Total trials = njobs * ntrials
seed = 100 * np.arange(1, njobs + 1, dtype=int)
data_path = os.path.abspath("./data/IC86_I_data.npy")
mc_path = os.path.abspath("./data/IC86_I_mc.npy")
runlist_path = os.path.abspath("./data/ic86-i-goodrunlist.json")
out_file = os.path.abspath("./results")

# File job and arg string templates to be filled with args
JOB = "JOB {} {}\n"
VARS = ("VARS {} ntrials=\"{}\" seed=\"{}\" data=\"{}\" " +
        "mc=\"{}\" runlist=\"{}\" out=\"{}\"\n")

# Write dagman job argument file
with open(job_options_fpath, "w") as jobf:
    for i in range(njobs):
        jobname = jobnames + "_{}".format(i)
        out_file_i = os.path.join(out_file, "trials_{}.pickle".format(i))

        jobf.write(JOB.format(jobname, oj_submit_path))

        jobf.write(VARS.format(jobname, ntrials, seed[i],
                               data_path, mc_path, runlist_path,
                               out_file_i))
