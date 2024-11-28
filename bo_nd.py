from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import landscape
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.stats import entropy, gamma
import sampling_randUnif
import test_functions
from acquisitionfunctions import *
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sampling_unifrefine import unifrefine

batch_sz = 3 # batch size in LHS
landscape.nin = 2
landscape.peakedness = 10 # set the peakedness to get more extremes
mus, covs = landscape.gen_gauss(5, landscape.nin, 1) # fix an f throughout the run

def f_aux(X):
    # return test_functions.trough2d(x)
    return landscape.f_sca(np.moveaxis(X,0,landscape.nin), mus, covs)

# def f(x,y,z):
#     return landscape.f_sca((x,y,z), mus, covs)

def create_function(arg_names):
    # Create a string defining the function with the required signature
    args = ", ".join(arg_names)
    func_def = f"""
def f({args}):
    return landscape.f_sca(({args}), mus, covs)
"""
    # Execute this string in the global namespace
    exec(func_def, globals())


def posterior(optimizer, grid):
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def extract_mu(optimizer, x):
    xs = [x for _ in range(landscape.nin)]
    Xgrid = np.meshgrid(*xs)
    
    out = np.transpose(np.vstack([X.ravel() for X in Xgrid]))
    mu, sigma = posterior(optimizer, out)
    mu = np.reshape(mu, np.shape(X[0]))
    # sigma = np.reshape(sigma, np.shape(X))

    return mu

def presample_lh(npoints, optimizer): # Create a LHS and update the optimizer
    lh = LatinHypercube(landscape.nin)
    xs = lh.random(npoints)

    for x in xs:
        optimizer.probe(x)

def presample_unif(npoints, optimizer): # Sample uniformly and update the optimizer
    xs = sampling_randUnif.randUnifSample(landscape.nin, npoints)

    for x in xs:
        optimizer.probe(x)

def presample_unifrefine(refine, optimizer): # Sample in a grid, specified by refine
    xs = np.array(unifrefine(landscape.d, landscape.nin, refine))

    for idx in np.ndindex(*(np.shape(xs)[1:])):
        optimizer.probe(xs[(slice(None),) + idx])

# Some acquisition functions
# acqf = acquisition.UpperConfidenceBound(kappa=10) 
# acqf = CB(beta=0, kappa=1)
# acqf = GP_UCB_2()
# acqf = RGP_UCB(theta = 3)
acqf = acquisition.ExpectedImprovement(xi = 6)

# Set opt bounds and create target
var_names = [ f"x{i}" for i in range(0, landscape.nin) ]
pbounds = { var: (0,1) for var in var_names }
create_function(var_names)
x = np.arange(0,1,0.01).reshape(-1,1)
X = np.array(np.meshgrid(*[x for _ in range(landscape.nin)]))
# print(X)
print(np.shape(X))
Z = f_aux(X)
npts = 49
nu = 1.5
alpha = 1e-3
# landscape.plot2d(mus, covs)

# This is just a dummy for unif sampling
optimizer = BayesianOptimization(
    f = f,
    pbounds=pbounds,
    acquisition_function=acqf,
    verbose = 0,
    random_state=0
)
optimizer._gp = GaussianProcessRegressor(
    kernel=Matern(nu=nu),
    alpha=alpha,
    normalize_y=True,
    n_restarts_optimizer=9,
    random_state=optimizer._random_state,
    )

# presample_unif(npts - 1, optimizer)
presample_unifrefine(3, optimizer)

optimizer.maximize(init_points=0, n_iter=1) # by optimising once, we get a nice posterior
mu = extract_mu(optimizer, x)
h_unif = entropy(Z.flatten(), np.abs(mu).flatten())
print(f"Entropy of unif search: {h_unif}")

# This is the real run
optimizer = BayesianOptimization(
    f = f,
    pbounds=pbounds,
    acquisition_function=acqf,
    verbose = 0,
    random_state=0
)
optimizer._gp = GaussianProcessRegressor(
    kernel=Matern(nu=nu),
    alpha=alpha,
    normalize_y=True,
    n_restarts_optimizer=9,
    random_state=optimizer._random_state,
    )

presample_lh(batch_sz, optimizer)
optimizer.maximize(init_points=0, n_iter=npts)
# mu = plot_gp_2d(optimizer, x, x, Z)
mu = extract_mu(optimizer, x)
h_reg = entropy(Z.flatten(), np.abs(mu).flatten())
print(f"Entropy of regression: {h_reg}")
print(f"h_unif/h_reg = {h_unif/h_reg}")

plt.show()