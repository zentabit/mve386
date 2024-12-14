import benchmark
import numpy as np
from bayes_opt import acquisition
from bo_common import RGP_UCB, GP_UCB_2, CB

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Gaussian parameters
dim = 2
nu = 1.5
alpha = 1e-3

# Aquisition
aq_base = CB
# aq_arg = {"xi":[3,15,10]} # EI optimum 8
aq_arg = {"kappa":[0, 20, 10]} # UCB optimum 8
# aq_arg = {"theta":[1, 10, 10]} # RGP_UCB optimum 5.5
# aq_arg={"delta": [0.01, 0.99, 10]} # GP_UCB_2 optimum 0.6
# Vill ni köra utan att benchmarka över aq-parametrar använd
# aq_arg = {"xi":[P,P,0]}
# där P är värdet på parametern ni vill köra


n_samples = 60 # Lågt för att testa snabbt
# init_points = 3 # TODO: Fråga: Bör vara samma som batch_size ?
function_number = 25
iter_repeats = 0

# Välj en av step size/batches
# Behöver sedan konsekvensändra i funktionsdefinitionen
# n_sample_step_size = 3


# aq_base = None # Uniform refinement
# uniform_refinement = 4 #Antalet punkter

batch_sizes = [5,10,15,20,30]
verbose = 0
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
    b.save(fname=f"benchmarks/CB/CB-{batch_size}")

    print(b)
