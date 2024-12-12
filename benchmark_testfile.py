import benchmark
from bayes_opt import acquisition
from bo_common import RGP_UCB, GP_UCB_2

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Gaussian parameters
dim = 2
nu = 1.5
alpha = 1e-3

# Aquisition
#aq_base = RGP_UCB
# aq_arg = {"xi":[1,10,10]} # EI optimum 8
# aq_arg = {"kappa":[5, 10, 5]} # UCB optimum 8
#aq_arg = {"theta":[5.5, 5.5, 0]} # RGP_UCB optimum 5.5
# aq_arg={"delta": [0.6, 0.6, 0]} # GP_UBP_2 optimum 0.6
# Vill ni köra utan att benchmarka över aq-parametrar använd
# aq_arg = {"xi":[P,P,0]}
# där P är värdet på parametern ni vill köra


n_samples = 100 # Lågt för att testa snabbt
init_points = 3 # TODO: Fråga: Bör vara samma som batch_size ?
function_number = 100
iter_repeats = 1

# Välj en av step size/batches
# Behöver sedan konsekvensändra i funktionsdefinitionen
# n_sample_step_size = 3
batch_size = 10

aq_base = None # Uniform refinement
uniform_refinement = 4 #Antalet punkter


verbose = 0

b = benchmark.Benchmark(
    dim,
    nu,
    alpha,
    aq_base,
    n_samples,
    init_points,
    function_number= function_number,
    unif_refinement=uniform_refinement,
    verbose=verbose
)

b.run()
b.save()

print(b)
