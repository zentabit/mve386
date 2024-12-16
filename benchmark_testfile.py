import benchmark
import numpy as np
from bayes_opt import acquisition
from bo_common import RGP_UCB, GP_UCB_2, CB

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Gaussian parameters
dim = 3
nu = 1.5
alpha = 1e-3

# Aquisition
aq_base = RGP_UCB
# print(aq_base.__name__)
# aq_arg = {"xi":[3,15,10]} # EI optimum 8
# aq_arg = {"kappa":[10, 10, 0]} # UCB optimum 8
aq_arg = {"theta":[5.5, 5.5, 0]} # RGP_UCB optimum 5.5
# aq_arg={"delta": [0.304, 0.304, 0]} # GP_UCB_2 optimum 0.6
# Vill ni köra utan att benchmarka över aq-parametrar använd
# aq_arg = {"xi":[P,P,0]}
# där P är värdet på parametern ni vill köra


n_samples = 360 # Lågt för att testa snabbt
# init_points = 3 # TODO: Fråga: Bör vara samma som batch_size ?
function_number = 100
iter_repeats = 0

# Välj en av step size/batches
# Behöver sedan konsekvensändra i funktionsdefinitionen
# n_sample_step_size = 3


# aq_base = None # Uniform refinement
# uniform_refinement = 7 #Antalet punkter

batch_sizes = [90]
verbose = 1
for i in range(np.size(batch_sizes)):
    batch_size = batch_sizes[i]
    init_points = batch_size

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
    b.save(fname=f"benchmarks/test_stage2/{aq_base.__name__}/{aq_base.__name__}-{batch_size}")

    print(b)
