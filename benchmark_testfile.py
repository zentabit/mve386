'''benchmark_testfile
Main benchmarking script. Creates an instance of Benchmark using the settings
defined here.
'''
import numpy as np
from bayes_opt import acquisition

from lib.bo_common import RGP_UCB, GP_UCB_2, CB
import lib.benchmark as benchmark

# sklearn produces a warning each time a GP is fitted. Turn them off.
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning) 

# Gaussian parameters
dim = 3
nu = 1.5
alpha = 1e-3

# Select an acquisition function here
aq_base = acquisition.UpperConfidenceBound

# Select tunable values here
# aq_arg = {"xi":[3,15,10]} # EI optimum 8
aq_arg = {"kappa":[14, 14, 0]} # UCB optimum 8
# aq_arg = {"theta":[5.5, 5.5, 0]} # RGP_UCB optimum 5.5
# aq_arg={"delta": [0.304, 0.304, 0]} # GP_UCB_2 optimum 0.6
# Vill ni köra utan att benchmarka över aq-parametrar använd
# aq_arg = {"xi":[P,P,0]}
# där P är värdet på parametern ni vill köra


n_samples = 360 # Total number of samples
function_number = 1 # The number of randomised functions to run the process on
iter_repeats = 0 # Repeat optimisation for the same function

# Setting this for sequential BEDO computes the KLD every nth iteration
# n_sample_step_size = 3

# Settings for uniform sampling
# aq_base = dummy_acqf # Uniform refinement
# uniform_refinement = 7 #Antalet punkter

batch_sizes = [5]
verbose = 1

for i in range(np.size(batch_sizes)):
    batch_size = batch_sizes[i]
    init_points = batch_size # first batch is LH

    b = benchmark.Benchmark(
        dim,
        nu,
        alpha,
        aq_base,
        n_samples,
        init_points,
        aq_arg,
        function_number= function_number,
        # unif_refinement=uniform_refinement,
        batch_size=batch_size,
        verbose=verbose
    )

    b.run()
    b.save(fname=f"benchmarks/stage2/{aq_base.__name__}/{aq_base.__name__}-{batch_size}")
    # b.save(fname=f"benchmarks/UCB/UCB-{batch_sizes[i]}")

    print(b)
