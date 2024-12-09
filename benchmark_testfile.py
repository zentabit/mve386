import benchmark

from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import math
import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from bo_common import *
from scipy.stats import entropy
import landscape
from contextlib import redirect_stdout


# Gaussian parameters
nu = 1.5
alpha = 1e-3


aq_base = acquisition.ExpectedImprovement
aq_arg = {"xi":[5,7,2]}

# Vill ni köra utan att benchmarka över aq-parametrar använd e
# aq_arg = {"xi":[P,P,1]}
# där P är värdet på parametern ni vill köra

iter_max = 30
iter_repeats = 2
        

b = benchmark.Benchmark(
    nu,
    alpha,
    aq_base,
    iter_max,
    3,
    aq_params=aq_arg,
    iteration_repeats= iter_repeats,
    function_number= 2
)

b.run()

b.save()

b._print()

print(b)