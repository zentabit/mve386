import benchmark
from bayes_opt import acquisition

# Gaussian parameters
dim = 2
nu = 1.5
alpha = 1e-3

# Aquisition
aq_base = acquisition.ExpectedImprovement
aq_arg = {"xi":[7,7,1]}
# Vill ni köra utan att benchmarka över aq-parametrar använd
# aq_arg = {"xi":[P,P,0]}
# där P är värdet på parametern ni vill köra

#
n_samples = 30 # Lågt för att testa snabbt
init_points = 3 # TODO: Fråga: Bör vara samma som batch_size ?
function_number = 2
iter_repeats = 1

# Välj en av step size/batches
# Behöver sedan konsekvensändra i funktionsdefinitionen
# n_sample_step_size = 3
batch_size = 10

verbose = 1

b = benchmark.Benchmark(
    dim,
    nu,
    alpha,
    aq_base,
    n_samples,
    init_points,
    aq_params=aq_arg,
    iteration_repeats= iter_repeats,
    function_number= function_number,
    batch_size= batch_size,
    verbose=verbose
)

print(b)

print(b._computeAqParams())

#b.run()
#b.save()
